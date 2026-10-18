"""Real-time macro data: every published vintage of each series, from ALFRED (spec section 5).

A value is visible at a cutoff only if its vintage was released strictly before the cutoff.
Before a series' first archived vintage, the earliest vintage is used and each period is gated
by a conservative release lag after the period ends; callers count those fallbacks.
"""
import json

import numpy as np
import pandas as pd
import requests

from fed_news import config
from fed_news.http import get

API = ("https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}"
       "&file_type=json&realtime_start=1776-07-04&realtime_end=9999-12-31"
       "&limit=100000&offset={offset}")

# name -> (FRED id, frequency, release lag in days after the period ends)
SERIES = {
    "m2": ("M2SL", "M", 27),
    "pce": ("PCEPI", "M", 31),
    "pce_core": ("PCEPILFE", "M", 31),
    "payems": ("PAYEMS", "M", 9),
    "earn": ("CES0500000030", "M", 9),
    "gdp": ("GDP", "Q", 31),
    "u6": ("U6RATE", "M", 9),
    "u3": ("UNRATE", "M", 9),
    "ppi": ("PPIACO", "M", 15),
    "cpi": ("CPIAUCSL", "M", 14),
    "cpi_core": ("CPILFESL", "M", 14),
    "hpi": ("HPIPONM226S", "M", 62),
}


def api_key() -> str:
    return (config.SECRETS / "fred_api_key.txt").read_text().strip()


def parse_observations(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["observations"])
    frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    frame["realtime_end"] = frame["realtime_end"].replace("9999-12-31", "2262-04-01")
    for col in ("date", "realtime_start", "realtime_end"):
        frame[col] = pd.to_datetime(frame[col])
    return frame[["date", "realtime_start", "realtime_end", "value"]]


def fetch_vintages(sid: str, refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "alfred" / f"{sid}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    key, parts, offset = api_key(), [], 0
    while True:
        try:
            text = get(API.format(sid=sid, key=key, offset=offset), cache=False)
        except (requests.RequestException, FileNotFoundError):
            # the original error message embeds the URL, which contains the API key
            raise RuntimeError(f"ALFRED fetch failed for {sid}") from None
        payload = json.loads(text)
        parts.append(parse_observations(payload))
        offset += len(payload["observations"])
        if not payload["observations"] or offset >= int(payload["count"]):
            break
    frame = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def period_end(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index + (pd.offsets.QuarterEnd(0) if freq == "Q" else pd.offsets.MonthEnd(0))


def asof_series(obs: pd.DataFrame, cutoff: pd.Timestamp, freq: str, lag_days: int):
    """The series as published strictly before `cutoff`, and whether the fallback was used."""
    day = cutoff - pd.Timedelta(days=1)
    first = obs["realtime_start"].min()
    if day >= first:
        live = obs[(obs["realtime_start"] <= day) & (obs["realtime_end"] >= day)]
        return live.set_index("date")["value"].sort_index(), False
    earliest = obs[obs["realtime_start"] == first].set_index("date")["value"].sort_index()
    released = period_end(earliest.index, freq) + pd.Timedelta(days=lag_days)
    return earliest[released < cutoff], True


def load_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {name: fetch_vintages(sid, refresh) for name, (sid, _, _) in SERIES.items()}
