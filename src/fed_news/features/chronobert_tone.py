"""BUILD: Fed tone scored by ChronoBERT (needs a CUDA GPU).

For each year, a ChronoBERT classifier is trained only on human labels from earlier years, then scores every sentence of
that year's statements, minutes, press conferences and speeches (tone = P(hawkish) - P(dovish)). A document's tone is
the mean over its sentences.
Reads:   <data>/labels.pkl, <data>/corpus.pkl; ChronoBERT weights from Hugging Face
         (manelalab/chrono-bert-v1-<year>1231).
Writes:  <data>/stance/<year>.pkl (sentence scores per year), <data>/doc_stance.pkl (tone per document),
         results/stance_accuracy.csv.
Run:     python -m fed_news.features.chronobert_tone

Details: Per-year, lookahead-free stance classifiers (spec section 6).

For each classifier year y (2000 onward; documents from 1994-1999 use y = 2000), fine-tune
ChronoBERT vintage y-1 (clipped to 1999-2024) on labels with year < y, three seeds, and score
every sentence of that year's statements, minutes, press conferences and speeches. Sentence
stance = P(hawkish) - P(dovish); document stance = mean over its sentences. Models are not
saved (disk); per-year results are, so an interrupted run resumes.
"""
import gc
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from fed_news import config
from fed_news.download.fed_documents import build_corpus, split_sentences
from fed_news.download.stance_labels import LABELS, load_labels

STANCE_TYPES = ("statement", "minutes", "presconf", "speech")


def vintage_year(row_year: int) -> int:
    """Which ChronoBERT checkpoint a year uses: the previous year's (clipped to 1999-2024)."""
    return min(max(row_year - 1, config.FIRST_VINTAGE), config.LAST_VINTAGE)


def classifier_year(row_year: int) -> int:
    """Which yearly classifier scores a document year (1994-1999 use the 2000 classifier)."""
    return max(row_year, config.FIRST_CLASSIFIER_YEAR)


def model_name(vintage: int) -> str:
    """Hugging Face name of the ChronoBERT checkpoint for a year."""
    return f"manelalab/chrono-bert-v1-{vintage}1231"


def eligible(labels: pd.DataFrame, cyear: int) -> pd.DataFrame:
    """Human labels dated before `cyear` (the only ones that classifier may train on)."""
    out = labels[labels["year"] < cyear]
    if len(out) < config.MIN_LABELS:
        raise ValueError(f"only {len(out)} labels before {cyear}; need {config.MIN_LABELS}")
    return out


class StanceNet(nn.Module):
    """ChronoBERT with a 3-way head (dovish / hawkish / neutral) on the mean of its token vectors."""
    def __init__(self, name: str):
        super().__init__()
        self.lm = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.lm.config.hidden_size, 3)

    def forward(self, ids, mask):
        """Class scores for a batch of tokenised sentences."""
        h = self.lm(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def _batches(tok, texts, size, order):
    """Tokenised mini-batches of `texts`, on the GPU."""
    for i in range(0, len(order), size):
        j = order[i:i + size]
        enc = tok([texts[k] for k in j], padding=True, truncation=True,
                  max_length=config.STANCE_MAXLEN, return_tensors="pt")
        yield j, enc["input_ids"].cuda(), enc["attention_mask"].cuda()


def train_and_score(train: pd.DataFrame, texts: list[str], vintage: int, seed: int) -> np.ndarray:
    """Fine-tune one classifier (one seed) on `train`, then return class probabilities for `texts`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    name = model_name(vintage)
    tok = AutoTokenizer.from_pretrained(name)
    net = StanceNet(name).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=config.STANCE_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    sents, y = list(train["sentence"]), torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    for _ in range(config.STANCE_EPOCHS):
        net.train()
        for j, ids, mask in _batches(tok, sents, config.STANCE_BATCH, rng.permutation(len(sents))):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), y[j].cuda())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    net.eval()
    out = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for j, ids, mask in _batches(tok, texts, 128, np.arange(len(texts))):
            out[j] = torch.softmax(net(ids, mask).float(), dim=-1).cpu().numpy()
    # Free everything per seed so repeated calls do not accumulate memory.
    del net, opt, scaler, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _sentence_table(docs: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence of each document: (doc_id, sentence)."""
    rows = [(d.doc_id, s.lower()) for d in docs.itertuples() for s in split_sentences(d.text)]
    return pd.DataFrame(rows, columns=["doc_id", "sentence"])


def _fingerprint(train: pd.DataFrame, holdout: pd.DataFrame, cyear: int, doc_ids: list[str]) -> dict:
    """Settings and data hashes of a year's run, so a saved year is reused only if nothing changed."""
    def digest(items) -> str:
        """SHA-1 of a list of items."""
        return hashlib.sha1("\n".join(map(str, items)).encode()).hexdigest()
    return {"vintage": vintage_year(cyear), "seeds": tuple(config.STANCE_SEEDS),
            "lr": config.STANCE_LR, "epochs": config.STANCE_EPOCHS,
            "batch": config.STANCE_BATCH, "maxlen": config.STANCE_MAXLEN,
            "holdout_frac": config.STANCE_HOLDOUT, "n_train": len(train),
            "n_holdout": len(holdout),
            "labels": digest(sorted(zip(train["sentence"], train["label"]))
                             + sorted(zip(holdout["sentence"], holdout["label"]))),
            "docs": digest(sorted(doc_ids))}


def _write_part(part, obj) -> None:
    """Save one year's results atomically (write a temp file, then rename)."""
    tmp = part.with_name(part.name + ".tmp")
    pd.to_pickle(obj, tmp)
    os.replace(tmp, part)


def run_stance() -> pd.DataFrame:
    """Score every year (reusing saved years), then write the tone per document and the accuracy table."""
    labels, corpus = load_labels(), build_corpus()
    docs = corpus[corpus["type"].isin(STANCE_TYPES)].copy()
    docs["cyear"] = docs["release"].dt.year.map(classifier_year)
    parts_dir = config.DATA / "stance"
    parts_dir.mkdir(exist_ok=True)
    acc_rows, all_sent = [], []
    for cyear in sorted(docs["cyear"].unique()):
        part = parts_dir / f"{cyear}.pkl"
        ydocs = docs[docs["cyear"] == cyear]
        pool = eligible(labels, cyear)
        hold = np.random.default_rng(0).random(len(pool)) < config.STANCE_HOLDOUT
        train, holdout = pool[~hold], pool[hold]
        if set(train["label"]) != set(LABELS.values()) or len(holdout) == 0:
            raise ValueError(f"cyear {cyear}: train must contain all 3 labels and holdout must be "
                             f"non-empty (train labels {sorted(set(train['label']))}, "
                             f"holdout n={len(holdout)})")
        fp = _fingerprint(train, holdout, cyear, list(ydocs["doc_id"]))
        cached = None
        if part.exists():
            loaded = pd.read_pickle(part)
            old = loaded[2] if len(loaded) == 3 else {}
            changed = [k for k in fp if old.get(k) != fp[k]]
            if changed:
                print(f"{cyear}: cached part stale, recomputing (changed: {', '.join(changed)})",
                      flush=True)
            else:
                cached = loaded
        if cached is not None:
            sent, acc, _ = cached
        else:
            sent = _sentence_table(ydocs)
            texts = list(holdout["sentence"]) + list(sent["sentence"])
            probs = np.mean([train_and_score(train, texts, vintage_year(cyear), seed)
                             for seed in config.STANCE_SEEDS], axis=0)
            pred = probs[:len(holdout)].argmax(1)
            truth = holdout["label"].to_numpy()
            f1 = np.mean([2 * ((pred == k) & (truth == k)).sum()
                          / max((pred == k).sum() + (truth == k).sum(), 1) for k in range(3)])
            acc = {"cyear": cyear, "vintage": vintage_year(cyear), "n_train": len(train),
                   "n_holdout": len(holdout), "accuracy": float((pred == truth).mean()),
                   "macro_f1": float(f1)}
            sp = probs[len(holdout):]
            sent = sent.assign(stance=sp[:, 1] - sp[:, 0])
            _write_part(part, (sent, acc, fp))
        acc_rows.append(acc)
        all_sent.append(sent)
        print(acc, flush=True)
    sentences = pd.concat(all_sent, ignore_index=True)
    agg = (sentences.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    doc_stance = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    doc_stance["n_sent"] = doc_stance["n_sent"].fillna(0).astype(int)
    empty = doc_stance[doc_stance["n_sent"] == 0]
    if len(empty):
        print("documents with zero sentences (kept, stance NaN):", flush=True)
        print(empty.groupby(["type", empty["release"].dt.year]).size().to_string(), flush=True)
    doc_stance = doc_stance.sort_values(["release", "doc_id"]).reset_index(drop=True)
    doc_stance.to_pickle(config.DATA / "doc_stance.pkl")
    pd.DataFrame(acc_rows).to_csv(config.RESULTS / "stance_accuracy.csv", index=False)
    return doc_stance


if __name__ == "__main__":
    out = run_stance()
    print(out.groupby(out["release"].dt.year)["stance"].describe().round(3).to_string())
