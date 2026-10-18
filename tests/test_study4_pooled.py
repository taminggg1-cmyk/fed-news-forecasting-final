import numpy as np
import pandas as pd
import pytest

from fed_news.studies.study4_networks_v2 import study4_pooled as sp


def _preds(dates, y_true, y_pred):
    return pd.DataFrame({"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred})


def test_pooled_differential_is_rowwise_mean_of_base_minus_news_squared_errors():
    d = ["2001-01-01", "2001-02-01"]
    base_a, new_a = _preds(d, [1.0, 1.0], [1.5, 1.0]), _preds(d, [1.0, 1.0], [1.0, 1.2])
    base_b, new_b = _preds(d, [1.0, 1.0], [1.0, 2.0]), _preds(d, [1.0, 1.0], [1.1, 1.0])
    out = sp.pooled_differential([(base_a, new_a), (base_b, new_b)])
    # pair a: [0.25 - 0, 0 - 0.04]; pair b: [0 - 0.01, 1 - 0]
    assert out.to_numpy() == pytest.approx([(0.25 - 0.01) / 2, (-0.04 + 1.0) / 2])
    assert list(out.index) == list(pd.to_datetime(d))


def test_pooled_differential_aligns_rows_on_date():
    base = _preds(["2001-01-01", "2001-02-01"], [1.0, 1.0], [2.0, 1.0])
    new = _preds(["2001-02-01", "2001-01-01"], [1.0, 1.0], [1.0, 1.0])
    out = sp.pooled_differential([(base, new)])
    assert out.loc[pd.Timestamp("2001-01-01")] == pytest.approx(1.0)
    assert out.loc[pd.Timestamp("2001-02-01")] == pytest.approx(0.0)


def test_pooled_differential_rejects_mismatched_rows():
    base = _preds(["2001-01-01", "2001-02-01"], [1.0, 1.0], [1.0, 1.0])
    new = _preds(["2001-01-01"], [1.0], [1.0])
    with pytest.raises(ValueError):
        sp.pooled_differential([(base, new)])


def test_summary_counts_positive_pairs_and_tests_the_mean():
    rng = np.random.default_rng(0)
    d = pd.Series(rng.normal(0.01, 0.001, 200))
    res = sp.summary(d, [0.2, -0.1, 0.3, 0.05], lags=8)
    assert res["pairs_positive"] == "3 of 4"
    assert res["p_one_sided"] < 0.001
    assert res["boot_ci90"][0] > 0
