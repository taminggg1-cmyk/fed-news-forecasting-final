import numpy as np
import pandas as pd
import pytest

from fed_news.collect import market


def test_splice_target_uses_point_then_range_midpoint():
    idx = pd.to_datetime(["2008-12-15", "2008-12-16", "2008-12-17"])
    daily = pd.DataFrame({"target_point": [1.0, np.nan, np.nan],
                          "target_lower": [np.nan, 0.0, 0.0],
                          "target_upper": [np.nan, 0.25, 0.25]}, index=idx)
    assert market.splice_target(daily).tolist() == [1.0, 0.125, 0.125]


def test_asof_is_strictly_before_cutoff():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert market.asof(s, pd.Timestamp("2020-01-03")) == 2.0
    assert np.isnan(market.asof(s, pd.Timestamp("2020-01-01")))


def test_asof_skips_missing_values():
    s = pd.Series([1.0, np.nan], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert market.asof(s, pd.Timestamp("2020-01-05")) == 1.0


@pytest.mark.network
def test_target_around_february_1994_hike():
    daily = market.load_daily(refresh=True)
    assert market.asof(daily["target"], pd.Timestamp("1994-02-04")) == pytest.approx(3.0)
    assert daily["target"].loc["1994-02-07"] == pytest.approx(3.25)
    assert daily["target"].loc["2008-12-17"] == pytest.approx(0.125)
