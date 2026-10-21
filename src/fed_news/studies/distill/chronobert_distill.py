"""BUILD: Fed tone from ChronoBERT (2009 vintage) trained on Claude Haiku's sentence probabilities ("Hd"; needs a
CUDA GPU, about 3-4 hours on an RTX 2060).

Haiku's class probabilities for the sentences before 2010 are the soft labels; the model is frozen at the end of 2009
and alone scores the sentences from 2010 on, so Haiku's possible memory of outcomes reaches only the training labels.
Reads:   <data>/coral/haiku_sentence_probs.pkl, <data>/corpus.pkl; ChronoBERT weights from Hugging Face
         (manelalab/chrono-bert-v1-20091231).
Writes:  results/distill/tuning.csv, settings.json, fidelity.json;
         <data>/distill_models/<final|foldA|foldB>/ (encoder, tokenizer, CORAL head, info.json);
         <data>/distill/scores_<model>.pkl (level probabilities per scored sentence);
         <data>/distill/doc_stance_distill.pkl (tone per document).
Run:     python -m fed_news.studies.distill.chronobert_distill tune     (grid on 1994-2006, validation 2007-2009)
         python -m fed_news.studies.distill.chronobert_distill final    (final model and the two cross-fit folds)
         python -m fed_news.studies.distill.chronobert_distill tone     (document tone and fidelity)

Details: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, Part B. The CORAL head and settings of the CORAL study
(batch 16, max length 256, AdamW weight decay 0.01, fp16 autocast, clipping 1.0, biases at learning rate 1e-2 without
weight decay); soft level targets (P_N + P_H, P_H); one seed. Every stage saves its output and is skipped when the
output exists, so an interrupted run resumes.
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from fed_news import config
from fed_news.studies.coral import chronobert_coral as cc
from fed_news.studies.coral.core import class_probs, coral_loss, level_probs, tone_from_levels
from fed_news.features.chronobert_tone import STANCE_TYPES, model_name

SENTENCES = config.DATA / "coral" / "haiku_sentence_probs.pkl"
MODELS = config.DATA / "distill_models"
WORK = config.DATA / "distill"
OUT_PATH = WORK / "doc_stance_distill.pkl"
RESULTS = config.RESULTS / "distill"
PROBS = ["p_dovish", "p_neutral", "p_hawkish"]
VINTAGE, CUT_YEAR = 2009, 2010
TUNE_LAST, VALID_YEARS = 2006, (2007, 2009)
TUNE_SHARE, N_VALID = 0.5, 6000
LRS, MAX_EPOCHS = (1e-5, 2e-5, 4e-5), 2
FOLDS = {"foldA": (1994, 2001), "foldB": (2002, 2009)}
SEED = 0
TIE = 1e-4


def with_years(df: pd.DataFrame) -> pd.DataFrame:
    """The sentences with their document's release date and year (doc ids are <type>:<YYYYMMDD>:<hash>)."""
    release = pd.to_datetime(df["doc_id"].str.split(":").str[1], format="%Y%m%d")
    return df.assign(release=release.to_numpy(), year=release.dt.year.to_numpy())


def load_sentences() -> pd.DataFrame:
    """Every sentence with Haiku probabilities (NaN where Haiku gave none), dated by its document."""
    return with_years(pd.read_pickle(SENTENCES)).reset_index(drop=True)


def soft_levels(q) -> np.ndarray:
    """Soft CORAL level targets from class probabilities (dovish, neutral, hawkish): (P_N + P_H, P_H)."""
    q = np.asarray(q, dtype=float)
    return np.column_stack([q[:, 1] + q[:, 2], q[:, 2]])


def init_biases_soft(head, lv) -> None:
    """Start the biases at the logits of the mean soft levels of the training rows (clipped to [0.001, 0.999])."""
    m = np.clip(np.nanmean(np.asarray(lv, dtype=float), axis=0), 1e-3, 1 - 1e-3)
    with torch.no_grad():
        head.bias.copy_(torch.as_tensor(np.log(m / (1 - m)), dtype=head.bias.dtype))


def soft_cross_entropy(q, lp) -> float:
    """Mean of -sum_c q_c log p_c: Haiku's class probabilities q against the model's class probabilities p."""
    p = np.clip(class_probs(lp), 1e-12, None)
    return float(np.mean(-(np.asarray(q, dtype=float) * np.log(p)).sum(1)))


def tuning_split(s: pd.DataFrame, n_valid: int = N_VALID, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Row labels for tuning: a random TUNE_SHARE of the labelled sentences up to TUNE_LAST, and n_valid random
    labelled sentences of VALID_YEARS."""
    lab = s[s[PROBS].notna().all(axis=1)]
    rng = np.random.default_rng(seed)
    pool = lab.index[lab["year"] <= TUNE_LAST].to_numpy()
    train = np.sort(rng.choice(pool, round(TUNE_SHARE * len(pool)), replace=False))
    vpool = lab.index[lab["year"].between(*VALID_YEARS)].to_numpy()
    valid = np.sort(rng.choice(vpool, min(n_valid, len(vpool)), replace=False))
    return train, valid


def plan(s: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Which labelled sentences each model trains on and which sentences it scores: `final` trains on everything
    before CUT_YEAR and scores the rest; each fold model trains on its own years and scores the other fold."""
    lab = s[PROBS].notna().all(axis=1)
    yr = s["year"]
    rows = lambda m: s.index[m].to_numpy()
    folds = {k: yr.between(*v) for k, v in FOLDS.items()}
    return {"final": {"train": rows(lab & (yr < CUT_YEAR)), "score": rows(yr >= CUT_YEAR)},
            "foldA": {"train": rows(lab & folds["foldA"]), "score": rows(folds["foldB"])},
            "foldB": {"train": rows(lab & folds["foldB"]), "score": rows(folds["foldA"])}}


def pick_best(rows: list[dict]) -> dict:
    """Lowest validation loss; settings within TIE of the lowest go to fewer epochs, then the lower learning rate.
    A setting whose loss is NaN (training diverged) never wins."""
    rows = [r for r in rows if np.isfinite(r["val_ce"])]
    best = min(r["val_ce"] for r in rows)
    near = [r for r in rows if r["val_ce"] <= best + TIE]
    r = min(near, key=lambda r: (r["epochs"], r["lr"]))
    return {"lr": r["lr"], "epochs": r["epochs"]}


def fine_tune_soft(net, tok, sents: list[str], lv, seed: int, device: str, lr: float, epochs: int,
                   batch: int = config.STANCE_BATCH, on_epoch=None, amp: bool | None = None):
    """AdamW on the CORAL loss with soft level targets: shuffled mini-batches, gradient clipping 1.0, fp16 autocast
    on the GPU (amp=False for float32); biases start at the mean soft levels. on_epoch(epoch, net) runs after each
    epoch. A batch whose loss is not finite is skipped; FloatingPointError if the weights end up non-finite."""
    rng = np.random.default_rng(seed)
    amp = device == "cuda" if amp is None else amp
    init_biases_soft(net.head, lv)
    net.to(device)
    opt = cc.make_optimizer(net, lr)
    scaler = torch.amp.GradScaler(enabled=amp)
    lv = torch.as_tensor(np.asarray(lv), dtype=torch.float32, device=device)
    skipped = 0
    for epoch in range(1, epochs + 1):
        net.train()
        for j, ids, mask in cc._batches(tok, sents, batch, rng.permutation(len(sents)), device):
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                logits = net(ids, mask)
            loss = coral_loss(logits.float(), lv[torch.as_tensor(j, device=device)])
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        if on_epoch is not None:
            on_epoch(epoch, net)
    net.zero_grad(set_to_none=True)  # the gradients would otherwise stay on the GPU with the model
    if skipped:
        print(f"skipped {skipped} batches with a non-finite loss", flush=True)
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise FloatingPointError("training diverged: the weights are no longer finite")
    return net


def score_sorted(net, tok, texts: list[str], device: str, batch: int = 128) -> np.ndarray:
    """Level probabilities for each text, batched in order of length (less padding), returned in input order. Rows
    that overflow in fp16 (non-finite) are scored again in float32."""
    net.to(device).eval()
    out = np.zeros((len(texts), 2), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for j, ids, mask in cc._batches(tok, texts, batch, order, device):
            out[j] = level_probs(net(ids, mask).float())
    bad = np.flatnonzero(~np.isfinite(out).all(1))
    if len(bad):
        with torch.no_grad():
            for j, ids, mask in cc._batches(tok, texts, 16, bad, device):
                out[j] = level_probs(net(ids, mask).float())
        print(f"rescored {len(bad)} rows in float32", flush=True)
    return out


def doc_tone(docs: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    """Tone per document (mean sentence tone; NaN and n_sent 0 without sentences), sorted by release."""
    agg = (sent.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    out = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    out["n_sent"] = out["n_sent"].fillna(0).astype(int)
    return out.sort_values(["release", "doc_id"]).reset_index(drop=True)


def fidelity(q, lp) -> dict:
    """How closely the model follows Haiku: tone correlation, soft cross-entropy, most-likely-class agreement."""
    q, p = np.asarray(q, dtype=float), class_probs(lp)
    return {"n": int(len(q)), "tone_correlation": float(np.corrcoef(q[:, 2] - q[:, 0], tone_from_levels(lp))[0, 1]),
            "soft_cross_entropy": soft_cross_entropy(q, lp), "agreement": float((q.argmax(1) == p.argmax(1)).mean())}


def _new_net():
    """A fresh 2009-vintage ChronoBERT with a CORAL head, and its tokenizer, on the GPU."""
    torch.manual_seed(SEED)
    name = model_name(VINTAGE)
    return cc.StanceNetCoral(AutoModel.from_pretrained(name)).cuda(), AutoTokenizer.from_pretrained(name)


def _release() -> None:
    """Return freed GPU memory to the device. The caller must drop its own references (del net, tok) first: on a
    6 GB card a second model left in memory spills into shared system memory and slows training several-fold."""
    gc.collect()
    torch.cuda.empty_cache()


def tune() -> dict:
    """The grid on 1994-2006 with validation on 2007-2009; rows saved after each learning rate (resumable)."""
    s = load_sentences()
    tr, va = tuning_split(s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "tuning.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    q_va = s.loc[va, PROBS].to_numpy(float)
    texts_va = list(s.loc[va, "sentence"])
    for lr in LRS:
        if any(np.isclose(r["lr"], lr) for r in rows):
            continue
        net, tok = _new_net()
        new = []

        def evaluate(epoch, net):
            lp = score_sorted(net, tok, texts_va, "cuda")
            torch.cuda.empty_cache()  # drop the scoring batches' cache before training continues
            new.append({"lr": lr, "epochs": epoch, "val_ce": soft_cross_entropy(q_va, lp),
                        **{f"val_{k}": v for k, v in fidelity(q_va, lp).items() if k != "soft_cross_entropy"},
                        "n_train": len(tr), "n_valid": len(va)})
            print(new[-1], flush=True)

        fine_tune_soft(net, tok, list(s.loc[tr, "sentence"]), soft_levels(s.loc[tr, PROBS]), SEED, "cuda", lr,
                       MAX_EPOCHS, on_epoch=evaluate)
        rows += new
        pd.DataFrame(rows).to_csv(path, index=False)
        del net, tok, evaluate
        _release()
    best = pick_best(rows)
    settings = {**best, "vintage": VINTAGE, "base_model": model_name(VINTAGE), "seed": SEED,
                "batch": config.STANCE_BATCH, "max_length": config.STANCE_MAXLEN, "weight_decay": 0.01,
                "bias_lr": cc.BIAS_LR, "loss": "CORAL with soft level targets (P_N + P_H, P_H) from Haiku",
                "grid": {"lr": list(LRS), "epochs": list(range(1, MAX_EPOCHS + 1))}}
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2))
    return settings


def _train_model(s: pd.DataFrame, rows, settings: dict):
    """Train one model with the tuned settings in fp16; if it diverges, train it again in float32."""
    tr = rows["train"]
    texts, lv = list(s.loc[tr, "sentence"]), soft_levels(s.loc[tr, PROBS])
    net, tok = _new_net()
    try:
        return fine_tune_soft(net, tok, texts, lv, SEED, "cuda", settings["lr"], settings["epochs"]), tok, "fp16"
    except FloatingPointError as e:
        print(f"{e}; training again in float32", flush=True)
    del net
    _release()
    net, tok = _new_net()
    return fine_tune_soft(net, tok, texts, lv, SEED, "cuda", settings["lr"], settings["epochs"], amp=False), tok, "fp32"


def final() -> None:
    """Train the final model and the two fold models with the tuned settings, save each, and score its sentences.
    A saved score file with non-finite rows is repaired: its rows are scored again with the saved model, or the model
    is trained again when its saved weights are not finite."""
    settings = json.loads((RESULTS / "settings.json").read_text())
    s = load_sentences()
    WORK.mkdir(parents=True, exist_ok=True)
    for name, rows in plan(s).items():
        out = WORK / f"scores_{name}.pkl"
        if out.exists():
            part = pd.read_pickle(out)
            bad = ~np.isfinite(part[["lp1", "lp2"]].to_numpy(float)).all(1)
            if not bad.any():
                continue
            net, tok = cc.load_model(MODELS / name)
            if all(bool(torch.isfinite(p).all()) for p in net.parameters()):
                lp = score_sorted(net, tok, list(s.loc[part.loc[bad, "row"], "sentence"]), "cuda")
                part.loc[bad, ["lp1", "lp2"]] = lp
                _write_scores(part, out)
                print(f"{name}: repaired {int(bad.sum())} rows", flush=True)
                del net, tok
                _release()
                continue
            print(f"{name}: saved weights are not finite; training again", flush=True)
            del net, tok
            _release()
        net, tok, precision = _train_model(s, rows, settings)
        tr = rows["train"]
        cc.save_model(net, tok, MODELS / name, {"model": name, "settings": settings, "n_train": len(tr),
                                                "precision": precision,
                                                "train_years": [int(s.loc[tr, "year"].min()),
                                                                int(s.loc[tr, "year"].max())],
                                                "label_source": "Claude Haiku 5.5 class probabilities"})
        lp = score_sorted(net, tok, list(s.loc[rows["score"], "sentence"]), "cuda")
        part = pd.DataFrame({"row": rows["score"], "lp1": lp[:, 0], "lp2": lp[:, 1]})
        _write_scores(part, out)
        print(f"{name}: trained on {len(tr)} ({precision}), scored {len(part)}", flush=True)
        del net, tok
        _release()


def _write_scores(part: pd.DataFrame, out) -> None:
    """Write a score file atomically; refuse non-finite level probabilities."""
    if not np.isfinite(part[["lp1", "lp2"]].to_numpy(float)).all():
        raise FloatingPointError(f"{out.name}: non-finite level probabilities")
    tmp = out.with_name(out.name + ".tmp")
    part.to_pickle(tmp)
    tmp.replace(out)


def tone() -> pd.DataFrame:
    """Join the three models' scores, write the tone per document and the fidelity report."""
    from fed_news.download.fed_documents import build_corpus
    s = load_sentences()
    parts = {name: pd.read_pickle(WORK / f"scores_{name}.pkl") for name in plan(s)}
    lp = pd.concat([p.assign(model=name) for name, p in parts.items()]).set_index("row").sort_index()
    if not lp.index.equals(s.index):
        raise ValueError("the three models must score every sentence exactly once")
    lv = lp[["lp1", "lp2"]].to_numpy(float)
    sent = s[["doc_id"]].assign(stance=tone_from_levels(lv))
    corpus = build_corpus()
    docs = corpus[corpus["type"].isin(STANCE_TYPES)]
    out = doc_tone(docs, sent)
    out.to_pickle(OUT_PATH)
    lab = s[PROBS].notna().all(axis=1).to_numpy()
    report = {}
    for label, mask in (("from_2010_final_model", lab & (s["year"] >= CUT_YEAR).to_numpy()),
                        ("before_2010_cross_fitted", lab & (s["year"] < CUT_YEAR).to_numpy())):
        report[label] = fidelity(s.loc[mask, PROBS].to_numpy(float), lv[mask])
    (RESULTS / "fidelity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "final", "tone"])
    step = ap.parse_args().step
    if step != "tone" and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required: the distillation trains only on the GPU")
    {"tune": tune, "final": final, "tone": tone}[step]()
