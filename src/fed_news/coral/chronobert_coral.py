"""BUILD: Fed tone scored by ChronoBERT fine-tuned with a CORAL output layer (needs a CUDA GPU, about 8 hours).

Same per-year protocol as features/chronobert_tone.py; only the output layer, the loss and the label order differ.
Every fine-tuned model is saved.
Reads:   <data>/labels.pkl, <data>/corpus.pkl; ChronoBERT weights from Hugging Face (manelalab/chrono-bert-v1-<year>1231).
Writes:  <data>/coral/stance/<year>.pkl (sentence level probabilities and tone per year),
         <data>/coral/doc_stance_chronobert_coral.pkl (tone per document),
         <data>/chronobert_coral_models/<year>/seed<k>/ (encoder, tokenizer, CORAL head, info.json),
         results/coral/chronobert_coral_accuracy.csv.
Run:     python -m fed_news.coral.chronobert_coral

Details: CORAL study, section 4. For each classifier year y (2000 onward; documents from 1994-1999 use y = 2000),
fine-tune ChronoBERT vintage y-1 (clipped to 1999-2024) on human labels dated before y, seeds 0, 1, 2, with the
Study 4 settings (learning rate 2e-5, 3 epochs, batch 16, max length 256, AdamW weight decay 0.01, fp16 autocast,
gradient clipping 1.0); design amendment 2: the CORAL biases start at the logits of the training labels'
cumulative shares and train at learning rate 1e-2 without weight decay. Labels are re-coded to ranks dovish 0 < neutral 1 < hawkish 2; the head is a CORAL layer on
the mean of the token vectors and the loss is the CORAL loss. The seeds' level probabilities are averaged; sentence
tone = P(hawkish) - P(dovish). Finished years are reused only if their fingerprint matches, so an interrupted run
resumes.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from fed_news import config
from fed_news.coral.core import TONE_RANK, CoralHead, biases_ordered, coral_loss, init_biases, level_probs, levels
from fed_news.coral.core import class_probs, param_groups, predict_rank
from fed_news.coral.core import tone_from_levels
from fed_news.download.fed_documents import build_corpus
from fed_news.download.stance_labels import LABELS, load_labels
from fed_news.features.chronobert_tone import STANCE_TYPES, _fingerprint, _sentence_table, _write_part
from fed_news.features.chronobert_tone import classifier_year, eligible, model_name, vintage_year

MODELS = config.DATA / "chronobert_coral_models"
WORK = config.DATA / "coral"
OUT_PATH = WORK / "doc_stance_chronobert_coral.pkl"
RESULTS = config.RESULTS / "coral"
BIAS_LR = 1e-2  # design amendment 2: the two thresholds get their own learning rate
LABEL_ORDER = "dovish 0 < neutral 1 < hawkish 2 (levels: dovish (0,0), neutral (1,0), hawkish (1,1))"


class StanceNetCoral(nn.Module):
    """A language model with a CORAL head on the mean of its token vectors: two level logits per sentence."""

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm
        self.head = CoralHead(lm.config.hidden_size)

    def forward(self, ids, mask):
        """Level logits for a batch of tokenised sentences."""
        h = self.lm(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def to_ranks(codes) -> np.ndarray:
    """Stored stance codes (dovish 0, hawkish 1, neutral 2) as ordinal ranks (dovish 0, neutral 1, hawkish 2)."""
    return np.array([TONE_RANK[int(c)] for c in codes], dtype=int)


def _batches(tok, texts, size, order, device):
    """Tokenised mini-batches of `texts` on `device`."""
    for i in range(0, len(order), size):
        j = order[i:i + size]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=config.STANCE_MAXLEN,
                  return_tensors="pt")
        yield j, enc["input_ids"].to(device), enc["attention_mask"].to(device)


def make_optimizer(net, lr: float = config.STANCE_LR):
    """AdamW: the CORAL biases at BIAS_LR without weight decay; every other parameter at `lr`, weight decay 0.01."""
    return torch.optim.AdamW(param_groups(net, lr, 0.01, bias_lr=BIAS_LR), lr=lr)


def fine_tune(net, tok, sents: list[str], ranks, seed: int, device: str, epochs: int = config.STANCE_EPOCHS,
              lr: float = config.STANCE_LR, batch: int = config.STANCE_BATCH):
    """AdamW on the CORAL loss, shuffled mini-batches, gradient clipping 1.0, fp16 autocast on the GPU; the biases
    start at the training labels' cumulative shares (design amendment 2)."""
    rng = np.random.default_rng(seed)
    amp = device == "cuda"
    init_biases(net.head, ranks)
    net.to(device)
    opt = make_optimizer(net, lr)
    scaler = torch.amp.GradScaler(enabled=amp)
    lv = levels(ranks).to(device)
    for _ in range(epochs):
        net.train()
        for j, ids, mask in _batches(tok, sents, batch, rng.permutation(len(sents)), device):
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                logits = net(ids, mask)
            loss = coral_loss(logits.float(), lv[torch.as_tensor(j, device=device)])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    return net


def score(net, tok, texts: list[str], device: str, batch: int = 128) -> np.ndarray:
    """Level probabilities P(rank >= 1), P(rank >= 2) for each text."""
    net.to(device).eval()
    out = np.zeros((len(texts), 2), dtype=np.float32)
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for j, ids, mask in _batches(tok, texts, batch, np.arange(len(texts)), device):
            out[j] = level_probs(net(ids, mask).float())
    return out


def save_model(net, tok, path: Path, info: dict) -> None:
    """Save the encoder (Hugging Face format, float32), the tokenizer, the CORAL head and info.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    net.lm.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    torch.save(net.head.state_dict(), path / "coral_head.pt")
    meta = {**info, "label_order": LABEL_ORDER, "biases": net.head.bias.detach().cpu().tolist(),
            "biases_ordered": biases_ordered(net.head)}
    (path / "info.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path: Path):
    """A saved model and its tokenizer (on the CPU)."""
    path = Path(path)
    net = StanceNetCoral(AutoModel.from_pretrained(path / "encoder"))
    net.head.load_state_dict(torch.load(path / "coral_head.pt", map_location="cpu"))
    return net, AutoTokenizer.from_pretrained(path / "tokenizer")


def train_and_score(train: pd.DataFrame, texts: list[str], vintage: int, seed: int, save_dir: Path | None,
                    info: dict | None = None) -> tuple[np.ndarray, bool, list[float]]:
    """Fine-tune one CORAL classifier (one seed) on `train` on the GPU, save it, and return the level probabilities
    for `texts`, whether its biases are ordered, and the biases."""
    torch.manual_seed(seed)
    name = model_name(vintage)
    tok = AutoTokenizer.from_pretrained(name)
    net = StanceNetCoral(AutoModel.from_pretrained(name)).cuda()
    fine_tune(net, tok, list(train["sentence"]), to_ranks(train["label"]), seed, "cuda")
    lp = score(net, tok, texts, "cuda")
    ordered, biases = biases_ordered(net.head), net.head.bias.detach().cpu().tolist()
    if save_dir is not None:
        settings = {"lr": config.STANCE_LR, "epochs": config.STANCE_EPOCHS, "batch": config.STANCE_BATCH,
                    "max_length": config.STANCE_MAXLEN, "weight_decay": 0.01, "loss": "CORAL",
                    "bias_lr": BIAS_LR, "bias_start": "logits of the training labels' cumulative shares"}
        save_model(net, tok, save_dir, {**(info or {}), "base_model": name, "vintage": vintage, "seed": seed,
                                        "n_train": len(train), "settings": settings})
    # Free everything per seed so repeated calls do not accumulate memory.
    del net, tok
    gc.collect()
    torch.cuda.empty_cache()
    return lp, ordered, biases


def combine_seeds(lps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean level probabilities over seeds (an average of ordered pairs stays ordered) and the tone they imply."""
    lp = np.mean(lps, axis=0)
    return lp, tone_from_levels(lp)


def ordinal_mae(lp: np.ndarray, ranks) -> float:
    """The paper's ordinal error: mean absolute distance between predicted and true ranks."""
    return float(np.mean(np.abs(predict_rank(lp) - np.asarray(ranks, dtype=int))))


def fingerprint(train: pd.DataFrame, holdout: pd.DataFrame, cyear: int, doc_ids: list[str]) -> dict:
    """The Study 4 fingerprint plus a CORAL marker, so a Study 4 year is never reused as a CORAL year."""
    return {**_fingerprint(train, holdout, cyear, doc_ids), "coral": 1}


def year_report(lp_hold: np.ndarray, truth, lp_all: np.ndarray, runs: list) -> dict:
    """Holdout accuracy, macro F1 and ordinal error, the mean class probabilities (holdout and all scored
    sentences) and each seed's biases: what to look at after the first year before letting the run continue."""
    pred, truth = predict_rank(lp_hold), np.asarray(truth, dtype=int)
    rep = {"accuracy": float((pred == truth).mean()), "macro_f1": _macro_f1(pred, truth),
           "ordinal_mae": ordinal_mae(lp_hold, truth)}
    for part, lp in (("holdout", lp_hold), ("all", lp_all)):
        p = class_probs(lp).mean(0)
        rep.update({f"mean_p_{c}_{part}": float(v) for c, v in zip(("dovish", "neutral", "hawkish"), p)})
    rep["biases_by_seed"] = [list(r[2]) for r in runs]
    rep["n_seeds_biases_unordered"] = int(sum(not r[1] for r in runs))
    return rep


def _macro_f1(pred, truth) -> float:
    """Macro F1 over the three ranks (the Study 4 formula)."""
    return float(np.mean([2 * ((pred == k) & (truth == k)).sum() / max((pred == k).sum() + (truth == k).sum(), 1)
                          for k in range(3)]))


def run() -> pd.DataFrame:
    """Fine-tune and score every year (reusing finished years), then write the tone per document and the holdout
    accuracy table."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required: ChronoBERT-CORAL fine-tuning runs only on the GPU")
    labels, corpus = load_labels(), build_corpus()
    docs = corpus[corpus["type"].isin(STANCE_TYPES)].copy()
    docs["cyear"] = docs["release"].dt.year.map(classifier_year)
    parts_dir = WORK / "stance"
    parts_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    acc_rows, all_sent = [], []
    for cyear in sorted(docs["cyear"].unique()):
        part = parts_dir / f"{cyear}.pkl"
        ydocs = docs[docs["cyear"] == cyear]
        pool = eligible(labels, cyear)
        hold = np.random.default_rng(0).random(len(pool)) < config.STANCE_HOLDOUT
        train, holdout = pool[~hold], pool[hold]
        if set(train["label"]) != set(LABELS.values()) or len(holdout) == 0:
            raise ValueError(f"cyear {cyear}: train must contain all 3 labels and holdout must be non-empty")
        fp = fingerprint(train, holdout, cyear, list(ydocs["doc_id"]))
        cached = pd.read_pickle(part) if part.exists() else None
        if cached is not None and cached[2] != fp:
            print(f"{cyear}: cached part stale, recomputing", flush=True)
            cached = None
        if cached is not None:
            sent, acc, _ = cached
        else:
            sent = _sentence_table(ydocs)
            texts = list(holdout["sentence"]) + list(sent["sentence"])
            info = {"cyear": int(cyear), "train_years": [int(train["year"].min()), int(train["year"].max())]}
            runs = [train_and_score(train, texts, vintage_year(cyear), seed, MODELS / str(cyear) / f"seed{seed}", info)
                    for seed in config.STANCE_SEEDS]
            lp, tone = combine_seeds([r[0] for r in runs])
            n = len(holdout)
            acc = {"cyear": int(cyear), "vintage": vintage_year(cyear), "n_train": len(train), "n_holdout": n,
                   **year_report(lp[:n], to_ranks(holdout["label"]), lp[n:], runs)}
            sent = sent.assign(lp1=lp[n:, 0], lp2=lp[n:, 1], stance=tone[n:])
            _write_part(part, (sent, acc, fp))
        acc_rows.append(acc)
        all_sent.append(sent)
        print(acc, flush=True)
    sentences = pd.concat(all_sent, ignore_index=True)
    agg = (sentences.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    doc_stance = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    doc_stance["n_sent"] = doc_stance["n_sent"].fillna(0).astype(int)
    doc_stance = doc_stance.sort_values(["release", "doc_id"]).reset_index(drop=True)
    doc_stance.to_pickle(OUT_PATH)
    pd.DataFrame(acc_rows).to_csv(RESULTS / "chronobert_coral_accuracy.csv", index=False)
    return doc_stance


if __name__ == "__main__":
    out = run()
    print(out.groupby(out["release"].dt.year)["stance"].describe().round(3).to_string())
