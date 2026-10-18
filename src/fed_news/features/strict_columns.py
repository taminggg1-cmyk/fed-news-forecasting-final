"""BUILD helper: the "strict data" columns.

Details: The strictly real-time statistics: the 8 inputs whose ALFRED vintages cover the whole sample, so no revised
fallback is ever used. Study 4 version 2 runs its "strict" arms on these columns (plus the market block M for
meetings). Copied unchanged from fed_news/studies/study5_horizon/study5.py (fed-news-forecasting ad2183c), where
they were first defined.
"""
import pandas as pd

from fed_news.features.panels import M_COLS

REALTIME_S = ["s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_payems_chg", "s_gdp_yoy", "s_cpi_yoy",
              "s_slope_10y2y", "s_slope_1y6m"]
KEPT_SERIES = ("m2", "payems", "gdp", "cpi")
B = REALTIME_S + M_COLS


def check_vintages(first: dict, kept, first_cutoff: pd.Timestamp) -> None:
    """Raise if a kept series has no first-published vintage before the first cutoff."""
    late = [k for k in kept if first[k] > first_cutoff - pd.Timedelta(days=1)]
    if late:
        raise ValueError(f"series would use the ALFRED fallback: {late}")
