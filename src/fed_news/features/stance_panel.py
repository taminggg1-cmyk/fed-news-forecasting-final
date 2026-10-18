"""Swap the stance columns of a cached panel (t1b / t2) for those built from another document-stance table, such
as the Claude Haiku index or ChronoBERT's. Copied unchanged from fed_news/studies/plan5_event_study/naive_stance.py
(fed-news-forecasting ad2183c).
"""
import pandas as pd

from fed_news import config
from fed_news.features.text_features import fed_stance


def naive_panel(name: str, doc_stance: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_pickle(config.DATA / f"{name}.pkl")
    full = pd.read_pickle(config.DATA / f"{name}_full.pkl").copy()
    st = pd.Series([fed_stance(doc_stance, s, c) for s, c in zip(rows["win_start"], rows["cutoff"])],
                   index=rows.index).ffill()
    full["n_stance"] = st.reindex(full.index)
    full["n_stance_chg"] = st.diff().reindex(full.index)
    return full
