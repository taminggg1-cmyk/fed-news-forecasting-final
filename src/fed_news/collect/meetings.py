"""One row per FOMC decision from 1994 (spec section 4, T2).

Scheduled decision dates are the minutes-link dates (within two days: merged to the later one).
Any target change not within 0-3 days after a minutes date adds an unscheduled date (the
statement date 0-1 days before the change if there is one, else the change date). Statement
dates are used for nothing else.
The decision is the target in force up to four days after the meeting (or the day before the
next meeting) minus the target in force before it. Cutoff = the day before the decision date.
"""
import numpy as np
import pandas as pd

from fed_news import config
from fed_news.collect import market
from fed_news.collect.fomc_calendar import load_calendar


def _split_dates(cal: pd.DataFrame, target: pd.Series) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Scheduled dates (minutes dates, merged within 2 days to the later one) and unscheduled
    dates (target changes not within 0-3 days after a scheduled date, dated by a statement
    0-1 days before the change if there is one, else by the change itself)."""
    minutes = sorted(set(cal.loc[cal["kind"] == "minutes", "date"]))
    statements = sorted(set(cal.loc[cal["kind"] == "statement", "date"]))
    scheduled: list[pd.Timestamp] = []
    for day in minutes:
        if scheduled and (day - scheduled[-1]).days <= 2:
            scheduled[-1] = day
        else:
            scheduled.append(day)
    unscheduled: set[pd.Timestamp] = set()
    steps = target.diff()
    for change in steps.index[steps.abs() > 1e-9]:
        if any(0 <= (change - m).days <= 3 for m in minutes):
            continue
        near = [s for s in statements if 0 <= (change - s).days <= 1]
        unscheduled.add(max(near) if near else change)
    return scheduled, sorted(unscheduled - set(scheduled))


def _decision_dates(cal: pd.DataFrame, target: pd.Series) -> list[pd.Timestamp]:
    scheduled, unscheduled = _split_dates(cal, target)
    return sorted(scheduled + unscheduled)


def build_meetings(cal: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    unscheduled_dates = set(_split_dates(cal, target)[1])
    dates = [d for d in _decision_dates(cal, target)
             if d >= pd.Timestamp(config.SAMPLE_START) and d + pd.Timedelta(days=4) <= target.index.max()]
    rows = []
    for i, day in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else day + pd.Timedelta(days=60)
        horizon = min(day + pd.Timedelta(days=5), nxt)
        before = market.asof(target, day)
        after = market.asof(target, horizon)
        change = after - before
        rows.append({
            "date": day, "cutoff": day - pd.Timedelta(days=1),
            "target_before": before, "target_after": after, "y_change": change,
            "y_class": 0 if change < -config.CLASS_EPS else (2 if change > config.CLASS_EPS else 1),
            "unscheduled": int(day in unscheduled_dates),
        })
    return pd.DataFrame(rows).set_index("date")


def load_meetings(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "meetings.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frame = build_meetings(load_calendar(), market.load_daily()["target"])
    frame.to_pickle(path)
    return frame


if __name__ == "__main__":
    m = load_meetings(refresh=True)
    print(f"meetings {len(m)}  {m.index.min().date()} .. {m.index.max().date()}")
    print(m["y_class"].value_counts().sort_index().to_string())
    print("unscheduled:", list(m.index[m["unscheduled"] == 1].date))
