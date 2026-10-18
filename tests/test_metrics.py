import numpy as np
import pandas as pd
import pytest

from fed_news.evaluation import metrics

OK = [0.5, 0.5, 0.5]


def test_metrics_on_toy_predictions():
    preds = pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.5, 2.0],
                          "class_true": [2, 1], "class_pred": [2, 0]})
    m = metrics.metrics(preds)
    assert m["mse"] == pytest.approx(0.125) and m["mae"] == pytest.approx(0.25)
    assert m["mbe"] == pytest.approx(0.25) and m["accuracy"] == pytest.approx(0.5)


def test_decide_requires_all_three():
    assert metrics.decide([0.01, 0.02, 0.03], [0.5, 0.6, 0.7], OK) == "supported"
    assert metrics.decide([0.01, 0.20, 0.03], [0.5, 0.6, 0.7], OK) == "not supported"


def test_decide_void_when_placebo_passes_any():
    assert metrics.decide([0.01, 0.02, 0.03], [0.5, 0.04, 0.7], OK) == "void"
    assert metrics.decide([0.30, 0.02, 0.03], [0.5, 0.04, 0.7], OK) == "void"


def test_decide_rejects_nan():
    """decide() must raise ValueError on NaN p-values."""
    with pytest.raises(ValueError, match="NaN p-value"):
        metrics.decide([0.01, 0.02, np.nan], [0.5, 0.6, 0.7], OK)
    with pytest.raises(ValueError, match="NaN p-value"):
        metrics.decide([0.01, 0.02, 0.03], [0.5, np.nan, 0.7], OK)


def test_decide_boundary_is_strict():
    """p == 0.05 (alpha) is not significant; p < 0.05 is."""
    assert metrics.decide([0.05, 0.05, 0.05], [0.5, 0.6, 0.7], OK) == "not supported"


def test_decide_void_when_shuffled_below_001():
    assert metrics.decide([0.01, 0.02, 0.03], OK, [0.5, 0.005, 0.5]) == "void"


def test_decide_not_void_when_shuffled_002():
    assert metrics.decide([0.01, 0.02, 0.03], OK, [0.5, 0.02, 0.5]) == "supported"


def test_decide_rejects_nan_in_shuffled():
    with pytest.raises(ValueError, match="NaN p-value"):
        metrics.decide([0.01, 0.02, 0.03], OK, [0.5, np.nan, 0.5])
