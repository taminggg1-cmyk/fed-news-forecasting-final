"""Daily FRED series that are never revised: target rate, effective rate, yields, VIX, oil.

The decision variable is the FOMC target, a step function: DFEDTAR (point target) to
2008-12-15, then the midpoint of DFEDTARL/DFEDTARU. FEDFUNDS (monthly average effective
rate) is the paper's T1 target.
"""
import io

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.http import get

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
DAILY = {
    "DFEDTAR": "target_point", "DFEDTARL": "target_lower", "DFEDTARU": "target_upper",
    "DFF": "eff_rate", "DTB3": "tbill_3m", "DGS6MO": "y6m", "DGS1": "y1", "DGS2": "y2",
    "DGS10": "y10", "VIXCLS": "vix", "DCOILWTICO": "wti",
}


def fred_csv(series_id: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(get(FRED_CSV.format(series_id), cache=False)))
    frame.columns = ["date", "value"]
    values = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(frame["date"]), name=series_id)


def splice_target(daily: pd.DataFrame) -> pd.Series:
    midpoint = (daily["target_lower"] + daily["target_upper"]) / 2.0
    return daily["target_point"].where(daily["target_point"].notna(), midpoint).ffill()


def load_daily(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "market_daily.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    daily = pd.DataFrame({name: fred_csv(sid) for sid, name in DAILY.items()}).sort_index()
    daily = daily.loc["1990-01-01":]
    daily["target"] = splice_target(daily)
    daily.to_pickle(path)
    return daily


def load_fedfunds(refresh: bool = False) -> pd.Series:
    path = config.DATA / "fedfunds.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    series = fred_csv("FEDFUNDS").loc["1990-01-01":]
    series.to_pickle(path)
    return series


def asof(series: pd.Series, cutoff: pd.Timestamp) -> float:
    """Last non-missing value dated strictly before `cutoff`."""
    sub = series.loc[series.index < cutoff].dropna()
    return float(sub.iloc[-1]) if len(sub) else np.nan
