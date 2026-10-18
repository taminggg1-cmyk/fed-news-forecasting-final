import numpy as np
import pandas as pd
import pytest

from fed_news.collect import indices


def test_window_mean_excludes_last_lag_days_and_cutoff():
    s = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10))
    # cutoff Jan 10, lag 3 -> end Jan 7 (exclusive), days 3 -> Jan 4, 5, 6
    assert indices.window_mean(s, pd.Timestamp("2020-01-10"), days=3, lag=3) == pytest.approx(4.0)


def test_window_mean_empty_is_nan():
    s = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    assert np.isnan(indices.window_mean(s, pd.Timestamp("2019-01-01")))


@pytest.mark.network
def test_live_indices_cover_sample():
    frame = indices.load_indices(refresh=True)
    assert list(frame.columns) == ["news", "gpr", "epu"]
    assert frame.index.min() <= pd.Timestamp("1993-01-01")
    assert frame.loc["1994":, ["news", "gpr", "epu"]].notna().mean().min() > 0.95
