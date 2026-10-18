"""Three daily newspaper-based indices (spec section 6, F4-F6).

news: SF Fed Daily News Sentiment Index (sheet "Data", columns date / News Sentiment).
gpr:  Caldara-Iacoviello daily Geopolitical Risk index (column GPRD).
epu:  Baker-Bloom-Davis daily Economic Policy Uncertainty (day, month, year, daily_policy_index).
Each feature is a mean over a 30-day window that ends three days before the cutoff.
"""
import io

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.http import get

SF_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"


def load_indices(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "indices.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    sf = pd.read_excel(io.BytesIO(get(SF_URL, binary=True, cache=False)), sheet_name="Data")
    news = pd.Series(sf["News Sentiment"].to_numpy(float), index=pd.to_datetime(sf["date"]))
    gpr_raw = pd.read_excel(io.BytesIO(get(GPR_URL, binary=True, cache=False)), engine="xlrd")
    gpr_raw["date"] = pd.to_datetime(gpr_raw["date"], errors="coerce")
    gpr_raw = gpr_raw.dropna(subset=["date"])
    gpr = pd.Series(gpr_raw["GPRD"].to_numpy(float), index=gpr_raw["date"])
    epu_raw = pd.read_csv(io.StringIO(get(EPU_URL, cache=False)))
    epu_idx = pd.to_datetime(epu_raw[["year", "month", "day"]])
    epu = pd.Series(epu_raw["daily_policy_index"].to_numpy(float), index=epu_idx)
    frame = pd.DataFrame({
        "news": news[~news.index.duplicated(keep="last")],
        "gpr": gpr[~gpr.index.duplicated(keep="last")],
        "epu": epu[~epu.index.duplicated(keep="last")],
    }).sort_index()
    frame.to_pickle(path)
    return frame


def window_mean(series: pd.Series, cutoff: pd.Timestamp, days: int = config.TEXT_WINDOW_DAYS,
                lag: int = config.INDEX_LAG_DAYS) -> float:
    end = cutoff - pd.Timedelta(days=lag)
    start = end - pd.Timedelta(days=days)
    sub = series[(series.index >= start) & (series.index < end)].dropna()
    return float(sub.mean()) if len(sub) else np.nan
