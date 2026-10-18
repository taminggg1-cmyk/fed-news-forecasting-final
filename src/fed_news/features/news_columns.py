"""BUILD: add the news columns to a panel.

n_stance and n_stance_chg (Fed tone and its change), n_beige (Beige Book tone), n_news, n_gpr, n_epu (news indices).
Reads:   <data>/<panel>.pkl, <data>/doc_stance.pkl, <data>/corpus.pkl, <data>/indices.pkl.
Writes:  <data>/<panel>_full.pkl.
Run:     python -m fed_news.features.news_columns           (t1 and t2)
         python -m fed_news.features.news_columns t1b t2    (the panels this study uses)

Details: The six news columns (spec section 6).

n_stance:     mean of group means (statement+press conference, minutes, speeches) over documents
              released in [win_start, cutoff); carried forward when the window is empty.
n_stance_chg: n_stance minus the previous row's n_stance.
n_beige:      Loughran-McDonald polarity of the latest Beige Book released before the cutoff.
n_news, n_gpr, n_epu: 30-day means ending 3 days before the cutoff (log levels for GPR, EPU).
"""
import numpy as np
import pandas as pd
import pysentiment2 as ps

from fed_news import config
from fed_news.download.fed_documents import build_corpus
from fed_news.download.news_indices import load_indices, window_mean
from fed_news.features.panels import N_COLS

GROUP = {"statement": "policy", "presconf": "policy", "minutes": "minutes", "speech": "speech"}
_LM = ps.LM()


def fed_stance(stance: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Fed tone over a window: mean of the group means (policy statements, minutes, speeches)."""
    win = stance[(stance["release"] >= start) & (stance["release"] < end)]
    if win.empty:
        return np.nan
    return float(win.groupby(win["type"].map(GROUP))["stance"].mean().mean())


def lm_tone(text: str) -> float:
    """Loughran-McDonald polarity of a text."""
    return float(_LM.get_score(_LM.tokenize(text))["Polarity"])


def beige_tones(corpus: pd.DataFrame) -> pd.Series:
    """Tone of every Beige Book, by release date."""
    beige = corpus[corpus["type"] == "beige"]
    return pd.Series([lm_tone(t) for t in beige["text"]], index=beige["release"]).sort_index()


def _latest_before(series: pd.Series, cutoff: pd.Timestamp) -> float:
    """Last value dated before `cutoff`."""
    sub = series[series.index < cutoff]
    return float(sub.sort_index().iloc[-1]) if len(sub) else np.nan


def safe_log(series: pd.Series) -> pd.Series:
    """Log of a positive index; values <= 0 (the GPR index has exact zeros) are missing, never -inf."""
    return np.log(series.where(series > 0))


def check_finite(frame: pd.DataFrame) -> None:
    """Raise if any non-missing value is +-inf (NaN is allowed: those rows drop in common_rows)."""
    bad = np.isinf(frame.to_numpy(float))
    if bad.any():
        cols = [c for c, b in zip(frame.columns, bad.any(0)) if b]
        raise ValueError(f"non-finite values in {cols}")


def build_text_features(rows: pd.DataFrame, stance: pd.DataFrame, beige: pd.Series,
                        idx: pd.DataFrame) -> pd.DataFrame:
    """The six news columns for each row of a panel."""
    if not rows.index.is_monotonic_increasing:
        raise ValueError("rows must be in chronological order (ffill/diff depend on it)")
    out = pd.DataFrame(index=rows.index)
    log_gpr, log_epu = safe_log(idx["gpr"]), safe_log(idx["epu"])
    out["n_stance"] = pd.Series([fed_stance(stance, s, c) for s, c in zip(rows["win_start"], rows["cutoff"])],
                                index=rows.index).ffill()
    out["n_stance_chg"] = out["n_stance"].diff()
    out["n_beige"] = [_latest_before(beige, c) for c in rows["cutoff"]]
    out["n_news"] = [window_mean(idx["news"], c) for c in rows["cutoff"]]
    out["n_gpr"] = [window_mean(log_gpr, c) for c in rows["cutoff"]]
    out["n_epu"] = [window_mean(log_epu, c) for c in rows["cutoff"]]
    return out[N_COLS]


def attach_all(names=("t1", "t2")) -> None:
    """Add the news columns to each named panel and save it as <data>/<name>_full.pkl."""
    stance = pd.read_pickle(config.DATA / "doc_stance.pkl")
    beige = beige_tones(build_corpus())
    idx = load_indices()
    for name in names:
        rows = pd.read_pickle(config.DATA / f"{name}.pkl")
        feats = build_text_features(rows, stance, beige, idx)
        check_finite(feats)
        full = rows.join(feats)
        full.to_pickle(config.DATA / f"{name}_full.pkl")
        print(f"{name}: N non-missing\n{full[N_COLS].notna().sum().to_string()}", flush=True)


if __name__ == "__main__":
    import sys
    attach_all(tuple(sys.argv[1:]) or ("t1", "t2"))
