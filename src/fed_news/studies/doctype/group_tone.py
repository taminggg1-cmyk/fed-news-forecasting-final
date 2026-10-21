"""BUILD (attempt 14): the distilled tone split by document type ("Hs", 6 columns).

For each panel row: the mean tone of the statements and press conferences, of the minutes and of the speeches
released in [win_start, cutoff), and the change of each from the previous row.
Reads:   <data>/t2.pkl, <data>/t2_full.pkl and a per-document tone table (attempt 14: the distilled tone,
         <data>/distill/doc_stance_distill.pkl).

Details: docs/studies/5_doctype/2026-10-21-doctype-tone-design.md, section 1. A group with no document in a window
keeps its last value; before a group's first document, the row's combined tone (n_stance, the mean of the group
means) is used.
"""
import numpy as np
import pandas as pd

from fed_news import config
from fed_news.features.news_columns import GROUP, fed_stance
from fed_news.features.swap_tone import naive_panel

GROUPS = ("policy", "minutes", "speech")
LEVELS = [f"n_stance_{g}" for g in GROUPS]
HS = LEVELS + [f"{c}_chg" for c in LEVELS]


def _group_mean(stance: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, group: str) -> float:
    """Mean tone of one group's documents released in [start, end); NaN if there are none."""
    win = stance[(stance["release"] >= start) & (stance["release"] < end) & (stance["type"].map(GROUP) == group)]
    return float(win["stance"].mean()) if len(win) else np.nan


def group_tones(rows: pd.DataFrame, doc_stance: pd.DataFrame) -> pd.DataFrame:
    """The 6 Hs columns for each row (rows need win_start and cutoff, in date order)."""
    if not rows.index.is_monotonic_increasing:
        raise ValueError("rows must be in chronological order (carry-forward and changes depend on it)")
    windows = list(zip(rows["win_start"], rows["cutoff"]))
    combined = pd.Series([fed_stance(doc_stance, s, c) for s, c in windows], index=rows.index).ffill()
    out = pd.DataFrame(index=rows.index)
    for group, col in zip(GROUPS, LEVELS):
        tone = pd.Series([_group_mean(doc_stance, s, c, group) for s, c in windows], index=rows.index)
        out[col] = tone.ffill().fillna(combined)
    for col in LEVELS:
        out[f"{col}_chg"] = out[col].diff()
    return out[HS]


def split_panel(doc_stance: pd.DataFrame, name: str = "t2") -> pd.DataFrame:
    """Panel <name>_full with n_stance and n_stance_chg rebuilt from `doc_stance`, plus the 6 Hs columns."""
    rows = pd.read_pickle(config.DATA / f"{name}.pkl")
    full = naive_panel(name, doc_stance)
    return full.join(group_tones(rows, doc_stance).reindex(full.index))
