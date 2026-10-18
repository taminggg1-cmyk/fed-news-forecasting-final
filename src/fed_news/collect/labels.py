"""Hawkish / dovish / neutral sentence labels (spec section 5).

Trillion Dollar Words (gtfintechlab/fomc_communication, 2,480 sentences, label 0 dovish,
1 hawkish, 2 neutral, with a year) plus the 1,000 Federal Reserve sentences of the World
Central Banks set (gtfintechlab/federal_reserve_system; the configs 5768/78516/944601 are three
splits of the same sentences, so one config's train+val+test is the full set).
Each sentence is dated by the earliest corpus document containing it (the datasets' own year
fields are not document years; spec section 5), so eligibility is lookahead-free.
"""
import html
import io
import unicodedata

import pandas as pd

from fed_news import config
from fed_news.collect.corpus import build_corpus, split_sentences
from fed_news.http import get

LABEL_TYPES = ("statement", "minutes", "presconf", "speech")
MIN_MATCH_RATE = 0.85
MIN_FALLBACK_CHARS = 40
DATED_COLUMNS = ["sentence", "label", "release", "source", "year"]
LABELS = {"dovish": 0, "hawkish": 1, "neutral": 2}
TDW = "https://huggingface.co/datasets/gtfintechlab/fomc_communication/resolve/main/{}.csv"
WCB = ("https://huggingface.co/datasets/gtfintechlab/federal_reserve_system/resolve/main/"
       "5768/{}-00000-of-00001.parquet")


_CHAR_MAP = {**{c: "'" for c in "\u2018\u2019\u201a\u201b\u2032"},
             **{c: '"' for c in "\u201c\u201d\u201e\u201f\u2033"},
             **{c: "-" for c in "\u2013\u2014\u2012\u2015\u2212"}}


def _norm(text: str) -> str:
    """Normalise for matching: unescape HTML, ASCII quotes/dashes, unicode spaces -> space,
    collapse whitespace, lowercase."""
    text = html.unescape(str(text))
    text = "".join(" " if unicodedata.category(c) == "Zs" else _CHAR_MAP.get(c, c) for c in text)
    return " ".join(text.split()).lower()


def _load_raw() -> pd.DataFrame:
    """TDW + WCB sentences; the TDW/WCB `year` columns are kept only as `raw_year` (not for
    eligibility: TDW years are not document years, see spec section 5)."""
    tdw = pd.concat([pd.read_csv(io.StringIO(get(TDW.format(s)))) for s in ("train", "test")])
    tdw = pd.DataFrame({"sentence": tdw["sentence"], "label": tdw["label"].astype(int),
                        "raw_year": tdw["year"].astype(int), "source": "tdw"})
    wcb = pd.concat([pd.read_parquet(io.BytesIO(get(WCB.format(s), binary=True)))
                     for s in ("train", "val", "test")])
    wcb = wcb[wcb["stance_label"].isin(LABELS)]
    wcb = pd.DataFrame({"sentence": wcb["sentences"], "label": wcb["stance_label"].map(LABELS),
                        "raw_year": wcb["year"].astype(int), "source": "wcb"})
    out = pd.concat([tdw, wcb], ignore_index=True)
    out["sentence"] = out["sentence"].map(_norm)
    return out.drop_duplicates("sentence").reset_index(drop=True)


def date_labels(labels: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Date each label sentence by the earliest release of a corpus document containing it.

    Unmatched labels are dropped. Returns sentence, label, release, source, year (= release year).
    """
    labels = labels.reset_index(drop=True)
    docs = corpus[corpus["type"].isin(LABEL_TYPES)].sort_values("release")
    first: dict[str, pd.Timestamp] = {}
    for d in docs.itertuples():
        for sent in split_sentences(d.text):
            first.setdefault(_norm(sent), d.release)
    rel = labels["sentence"].map(_norm).map(first)
    missing = rel.isna()
    if missing.any():
        texts = [(d.release, _norm(d.text)) for d in docs.itertuples()]
        for idx in labels.index[missing]:
            needle = _norm(labels.at[idx, "sentence"])
            if len(needle) < MIN_FALLBACK_CHARS:
                continue
            for release, text in texts:
                if needle in text:
                    rel.at[idx] = release
                    break
    out = labels.assign(sentence=labels["sentence"].map(_norm), release=rel)
    out = out[out["release"].notna()].reset_index(drop=True)
    out["release"] = pd.to_datetime(out["release"])
    out["year"] = out["release"].dt.year.astype(int)
    return out[DATED_COLUMNS]


def load_labels(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "labels.pkl"
    if path.exists() and not refresh:
        cached = pd.read_pickle(path)
        if list(cached.columns) == DATED_COLUMNS:
            return cached
        print("labels.pkl has a stale schema (no release dates); rebuilding", flush=True)
    raw = _load_raw()
    out = date_labels(raw, build_corpus())
    for source in ("tdw", "wcb"):
        n_raw, n_out = (raw["source"] == source).sum(), (out["source"] == source).sum()
        rate = n_out / max(n_raw, 1)
        print(f"label match rate {source}: {n_out}/{n_raw} = {rate:.3f}", flush=True)
        if rate < MIN_MATCH_RATE:
            raise ValueError(f"only {rate:.1%} of {source} labels matched corpus documents; "
                             f"need {MIN_MATCH_RATE:.0%}")
    for cyear in range(config.FIRST_CLASSIFIER_YEAR, 2027):
        print(f"eligible labels for classifier {cyear}: {(out['year'] < cyear).sum()}", flush=True)
    out.to_pickle(path)
    return out
