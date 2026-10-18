import numpy as np
import pandas as pd
import pytest

from fed_news.features import panel

T = pd.Timestamp


def flat_vintage(level_by_month: dict, released: str):
    dates = list(level_by_month)
    return pd.DataFrame({"date": [T(d) for d in dates], "realtime_start": T(released),
                         "realtime_end": T("2262-04-01"), "value": list(level_by_month.values())})


def test_feature_columns_never_include_targets():
    cols = panel.feature_columns("SMN")
    assert cols == panel.S_COLS + panel.M_COLS + panel.N_COLS
    assert not any(c.startswith("y_") for c in cols)


def test_s_cols_match_paper_table_1():
    assert panel.S_COLS == [
        "s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_pce_yoy", "s_pce_core_yoy", "s_payems_chg",
        "s_earn_yoy", "s_gdp_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy", "s_cpi_yoy",
        "s_cpi_core_yoy", "s_hpi_yoy", "s_slope_10y2y", "s_slope_1y6m"]


def test_macro_at_computes_yoy_within_one_vintage():
    months = pd.date_range("1999-01-01", "2000-01-01", freq="MS")
    obs = flat_vintage({m: 100.0 + i for i, m in enumerate(months)}, "2000-02-15")
    alfred_obs = {name: obs for name in ["m2", "pce", "pce_core", "payems", "earn", "u6", "u3",
                                         "ppi", "cpi", "cpi_core", "hpi"]}
    q = pd.date_range("1999-01-01", "2000-01-01", freq="QS")
    alfred_obs["gdp"] = flat_vintage({d: 200.0 + i for i, d in enumerate(q)}, "2000-02-15")
    out, fallbacks = panel.macro_at(alfred_obs, T("2000-02-16"))
    assert out["s_cpi_yoy"] == pytest.approx((112.0 / 100.0 - 1) * 100)
    assert out["s_payems_chg"] == pytest.approx(1.0)
    assert out["s_u6_yoy"] == pytest.approx(12.0)
    assert out["t_u3"] == pytest.approx(112.0)
    assert out["s_gdp_yoy"] == pytest.approx((204.0 / 200.0 - 1) * 100)
    assert fallbacks == 0
    # cutoff on the first vintage's release day: that vintage is invisible, so all 12 series
    # fall back to release-lag gating (strictness itself is tested in test_alfred.py)
    _, fallbacks = panel.macro_at(alfred_obs, T("2000-02-15"))
    assert fallbacks == 12


def test_market_at_uses_target_before_cutoff():
    idx = pd.date_range("2020-01-01", "2020-01-10")
    daily = pd.DataFrame({"target": 1.0, "tbill_3m": 1.5, "y6m": 1.6, "y1": 1.7, "y2": 1.8,
                          "y10": 2.0, "vix": 15.0, "wti": 60.0}, index=idx)
    daily.loc["2020-01-10":, "target"] = 9.0
    out = panel.market_at(daily, T("2020-01-10"))
    assert out["m_sp_tbill_3m"] == pytest.approx(0.5)
    assert out["s_slope_10y2y"] == pytest.approx(0.2)
    assert out["s_slope_1y6m"] == pytest.approx(0.1)
    assert out["m_zero_bound"] == 0


def test_second_business_day_skips_weekends_and_federal_holidays():
    from fed_news.features.panel import second_business_day
    assert second_business_day(pd.Timestamp("2025-09-01")) == pd.Timestamp("2025-09-03")  # Mon 1st Labor Day
    assert second_business_day(pd.Timestamp("2022-01-01")) == pd.Timestamp("2022-01-04")  # Sat 1st
    assert second_business_day(pd.Timestamp("2025-02-01")) == pd.Timestamp("2025-02-04")  # Sat 1st
    assert second_business_day(pd.Timestamp("2026-10-01")) == pd.Timestamp("2026-10-02")  # Thu 1st
    assert second_business_day(pd.Timestamp("2024-01-01")) == pd.Timestamp("2024-01-03")  # Mon 1st New Year


def test_monthly_v2_rows_use_the_second_business_day_cutoff(monkeypatch):
    from fed_news.features import panel
    calls = []
    monkeypatch.setattr(panel, "_row", lambda cutoff, win_start, y, lag, d, a: calls.append((cutoff, win_start)) or
                        {"cutoff": cutoff, "win_start": win_start, "y_level": y, "y_change": y - lag, "s_rate_lag": lag})
    ff = pd.Series([1.0, 1.1, 1.2], index=pd.to_datetime(["2025-07-01", "2025-08-01", "2025-09-01"]))
    monkeypatch.setattr(panel.config, "SAMPLE_START", "2025-07-01")
    out = panel.build_t1b(None, ff, None)
    assert calls[-1] == (pd.Timestamp("2025-09-03"), pd.Timestamp("2025-08-04"))  # Fri 1 Aug, so Mon 4 Aug
    assert list(out.index) == list(ff.index[1:])
