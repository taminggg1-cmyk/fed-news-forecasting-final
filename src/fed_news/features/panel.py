"""Rows for the two tasks (spec sections 4-5).

T1: one row per month t from 1994; cutoff = first day of month t; y_level = FEDFUNDS(t);
    s_rate_lag = FEDFUNDS(t-1); text window [first day of t-1, cutoff).
T2: one row per FOMC decision; cutoff = day before the decision; y_level = target after;
    s_rate_lag = target before; text window [previous meeting's cutoff, cutoff).
S = the paper's Table 1 statistics, real-time; M = our market block; t_u3 feeds the Taylor rule.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from fed_news.collect import alfred, market
from fed_news import config
from fed_news.collect.meetings import load_meetings

S_COLS = ["s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_pce_yoy", "s_pce_core_yoy", "s_payems_chg",
          "s_earn_yoy", "s_gdp_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy", "s_cpi_yoy",
          "s_cpi_core_yoy", "s_hpi_yoy", "s_slope_10y2y", "s_slope_1y6m"]
M_COLS = ["m_sp_tbill_3m", "m_sp_y6m", "m_sp_y1", "m_sp_y2", "m_target_level", "m_zero_bound",
          "m_vix", "m_wti"]
N_COLS = ["n_stance", "n_stance_chg", "n_beige", "n_news", "n_gpr", "n_epu"]
TAYLOR_COLS = ["s_rate_lag", "s_pce_core_yoy", "t_u3"]
BLOCKS = {"S": S_COLS, "M": M_COLS, "N": N_COLS}
YOY = {"pce": "s_pce_yoy", "pce_core": "s_pce_core_yoy", "earn": "s_earn_yoy", "ppi": "s_ppi_yoy",
       "cpi": "s_cpi_yoy", "cpi_core": "s_cpi_core_yoy", "hpi": "s_hpi_yoy"}


def feature_columns(blocks: str) -> list[str]:
    cols = [c for b in blocks for c in BLOCKS[b]]
    if any(c.startswith("y_") for c in cols):
        raise ValueError("target column in feature list")
    return cols


def _last(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.iloc[-1]) if len(series) else np.nan


def macro_at(alfred_obs: dict, cutoff: pd.Timestamp) -> tuple[dict, int]:
    out, fallbacks = {}, 0

    def get_series(name: str) -> pd.Series:
        nonlocal fallbacks
        _, freq, lag = alfred.SERIES[name]
        series, used_fallback = alfred.asof_series(alfred_obs[name], cutoff, freq, lag)
        fallbacks += int(used_fallback)
        return series.asfreq("QS" if freq == "Q" else "MS") if len(series) else series

    m2 = get_series("m2")
    out["s_m2_mom"] = _last(m2.pct_change() * 100)
    out["s_m2_yoy"] = _last(m2.pct_change(12) * 100)
    for name, col in YOY.items():
        out[col] = _last(get_series(name).pct_change(12) * 100)
    out["s_payems_chg"] = _last(get_series("payems").diff())
    out["s_gdp_yoy"] = _last(get_series("gdp").pct_change(4) * 100)
    u6 = get_series("u6")
    out["s_u6_mom"] = _last(u6.diff())
    out["s_u6_yoy"] = _last(u6.diff(12))
    out["t_u3"] = _last(get_series("u3"))
    return out, fallbacks


def market_at(daily: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    v = {col: market.asof(daily[col], cutoff)
         for col in ["target", "tbill_3m", "y6m", "y1", "y2", "y10", "vix", "wti"]}
    return {
        "s_slope_10y2y": v["y10"] - v["y2"], "s_slope_1y6m": v["y1"] - v["y6m"],
        "m_sp_tbill_3m": v["tbill_3m"] - v["target"], "m_sp_y6m": v["y6m"] - v["target"],
        "m_sp_y1": v["y1"] - v["target"], "m_sp_y2": v["y2"] - v["target"],
        "m_target_level": v["target"], "m_zero_bound": int(v["target"] <= 0.30),
        "m_vix": v["vix"], "m_wti": v["wti"],
    }


def _row(cutoff, win_start, y_level, rate_lag, daily, alfred_obs) -> dict:
    macro, fallbacks = macro_at(alfred_obs, cutoff)
    return {"cutoff": cutoff, "win_start": win_start, "y_level": y_level,
            "y_change": y_level - rate_lag, "s_rate_lag": rate_lag,
            **macro, **market_at(daily, cutoff), "fallback_n": fallbacks}


def _classes(change: pd.Series) -> pd.Series:
    return pd.Series(np.where(change > config.CLASS_EPS, 2, np.where(change < -config.CLASS_EPS, 0, 1)),
                     index=change.index)


def build_t1(daily, fedfunds, alfred_obs) -> pd.DataFrame:
    rows = {}
    months = fedfunds.loc[config.SAMPLE_START:].index
    for t in months:
        prev = t - pd.DateOffset(months=1)
        if prev not in fedfunds.index:
            continue
        rows[t] = _row(t, prev, float(fedfunds[t]), float(fedfunds[prev]), daily, alfred_obs)
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame["y_class"] = _classes(frame["y_change"])
    return frame


_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def second_business_day(month_start: pd.Timestamp) -> pd.Timestamp:
    """Study 4 v2 monthly cutoff: 00:00 New York time at the start of the second US business day of the month."""
    return _BDAY.rollforward(pd.Timestamp(month_start)) + _BDAY


def build_t1b(daily, fedfunds, alfred_obs) -> pd.DataFrame:
    """T1 rows with the Study 4 v2 cutoff (design section 12.1): inputs dated before the second business day of
    month t, by which time FEDFUNDS(t-1) has been published; the text window starts at month t-1's cutoff."""
    rows = {}
    for t in fedfunds.loc[config.SAMPLE_START:].index:
        prev = t - pd.DateOffset(months=1)
        if prev not in fedfunds.index:
            continue
        rows[t] = _row(second_business_day(t), second_business_day(prev), float(fedfunds[t]), float(fedfunds[prev]),
                       daily, alfred_obs)
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame["y_class"] = _classes(frame["y_change"])
    return frame


def build_t1b_panel() -> pd.DataFrame:
    t1b = build_t1b(market.load_daily(), market.load_fedfunds(), alfred.load_all())
    t1b.to_pickle(config.DATA / "t1b.pkl")
    return t1b


def build_t2(meet, daily, alfred_obs) -> pd.DataFrame:
    rows, prev_cutoff = {}, None
    for day, m in meet.iterrows():
        win_start = prev_cutoff if prev_cutoff is not None else m["cutoff"] - pd.Timedelta(days=45)
        rows[day] = _row(m["cutoff"], win_start, m["target_after"], m["target_before"], daily, alfred_obs)
        rows[day]["y_class"] = int(m["y_class"])
        rows[day]["unscheduled"] = int(m["unscheduled"])
        prev_cutoff = m["cutoff"]
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, fedfunds, alfred_obs = market.load_daily(), market.load_fedfunds(), alfred.load_all()
    t1 = build_t1(daily, fedfunds, alfred_obs)
    t2 = build_t2(load_meetings(), daily, alfred_obs)
    t1.to_pickle(config.DATA / "t1.pkl")
    t2.to_pickle(config.DATA / "t2.pkl")
    return t1, t2


if __name__ == "__main__":
    t1, t2 = build_panels()
    for name, frame in (("T1", t1), ("T2", t2)):
        complete = frame[S_COLS + M_COLS + ["t_u3"]].notna().all(axis=1)
        print(f"{name}: rows {len(frame)}, complete S+M+Taylor {complete.sum()}, first complete "
              f"{frame.index[complete.to_numpy()].min().date()}, fallback rows {(frame.fallback_n > 0).sum()}")
