import numpy as np
import pandas as pd
import pytest

from fed_news.evaluation import evaluate

OK = [0.5, 0.5, 0.5]


def test_metrics_on_toy_predictions():
    preds = pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.5, 2.0],
                          "class_true": [2, 1], "class_pred": [2, 0]})
    m = evaluate.metrics(preds)
    assert m["mse"] == pytest.approx(0.125) and m["mae"] == pytest.approx(0.25)
    assert m["mbe"] == pytest.approx(0.25) and m["accuracy"] == pytest.approx(0.5)


def test_decide_requires_all_three():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.6, 0.7], OK) == "supported"
    assert evaluate.decide([0.01, 0.20, 0.03], [0.5, 0.6, 0.7], OK) == "not supported"


def test_decide_void_when_placebo_passes_any():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.04, 0.7], OK) == "void"
    assert evaluate.decide([0.30, 0.02, 0.03], [0.5, 0.04, 0.7], OK) == "void"


def test_decide_rejects_nan():
    """decide() must raise ValueError on NaN p-values."""
    with pytest.raises(ValueError, match="NaN p-value"):
        evaluate.decide([0.01, 0.02, np.nan], [0.5, 0.6, 0.7], OK)
    with pytest.raises(ValueError, match="NaN p-value"):
        evaluate.decide([0.01, 0.02, 0.03], [0.5, np.nan, 0.7], OK)


def test_decide_boundary_is_strict():
    """p == 0.05 (alpha) is not significant; p < 0.05 is."""
    assert evaluate.decide([0.05, 0.05, 0.05], [0.5, 0.6, 0.7], OK) == "not supported"


def test_three_tests_orders_base_and_new():
    """Test that _three_tests correctly identifies when S+N beats S."""
    import pandas as pd
    from datetime import datetime

    # Create synthetic data: S+N clearly better than S
    dates = pd.date_range("2020-01-01", periods=10)
    # S has high error and low accuracy
    s_data = pd.DataFrame({
        "date": dates,
        "y_true": [1.0] * 10,
        "y_pred": [3.0] * 10,  # large error
        "class_true": [0, 1, 0, 1] * 2 + [0, 1],
        "class_pred": [1, 0, 1, 0] * 2 + [1, 0],  # opposite, 0% accuracy
    })
    # S+N has low error and high accuracy
    sn_data = pd.DataFrame({
        "date": dates,
        "y_true": [1.0] * 10,
        "y_pred": [1.1] * 10,  # small error
        "class_true": [0, 1, 0, 1] * 2 + [0, 1],
        "class_pred": [0, 1, 0, 1] * 2 + [0, 1],  # correct, 100% accuracy
    })

    t1 = {"S": s_data, "S+N": sn_data}
    t2 = {"S": s_data, "S+N": sn_data}

    p_values = evaluate._three_tests(t1, t2, "S+N")
    assert all(p < 0.05 for p in p_values), f"Expected all p-values < 0.05, got {p_values}"

    # Swapping S and S+N should reverse the advantage
    p_values_swapped = evaluate._three_tests(
        {"S": sn_data, "S+N": s_data},
        {"S": sn_data, "S+N": s_data},
        "S+N"
    )
    # Both DM-HLN tests should have p > 0.5 when S+N (actually S) is worse
    assert p_values_swapped[0] > 0.5, f"DM T1 p-value should be > 0.5 when base is better"
    assert p_values_swapped[1] > 0.5, f"DM T2 p-value should be > 0.5 when base is better"


def test_three_tests_rejects_misaligned_dates():
    """_three_tests must reject frames with different date columns."""
    dates1 = pd.date_range("2020-01-01", periods=10)
    dates2 = pd.date_range("2020-01-11", periods=10)

    s_data = pd.DataFrame({
        "date": dates1,
        "y_true": [1.0] * 10,
        "y_pred": [1.1] * 10,
        "class_true": [0, 1] * 5,
        "class_pred": [0, 1] * 5,
    })
    sn_data = pd.DataFrame({
        "date": dates2,
        "y_true": [1.0] * 10,
        "y_pred": [1.1] * 10,
        "class_true": [0, 1] * 5,
        "class_pred": [0, 1] * 5,
    })

    t1 = {"S": s_data, "S+N": sn_data}
    t2 = {"S": s_data, "S+N": sn_data}

    with pytest.raises(ValueError, match="prediction dates differ"):
        evaluate._three_tests(t1, t2, "S+N")


def test_decide_void_when_shuffled_below_001():
    assert evaluate.decide([0.01, 0.02, 0.03], OK, [0.5, 0.005, 0.5]) == "void"


def test_decide_not_void_when_shuffled_002():
    assert evaluate.decide([0.01, 0.02, 0.03], OK, [0.5, 0.02, 0.5]) == "supported"


def test_decide_rejects_nan_in_shuffled():
    with pytest.raises(ValueError, match="NaN p-value"):
        evaluate.decide([0.01, 0.02, 0.03], OK, [0.5, np.nan, 0.5])


def test_deviations_section_copies_file_or_default(tmp_path):
    assert evaluate.deviations_section(tmp_path) == "None recorded."
    (tmp_path / "DEVIATIONS.md").write_text("- changed X because Y\n", encoding="utf-8")
    assert evaluate.deviations_section(tmp_path) == "- changed X because Y"


def test_provenance_without_tag(tmp_path):
    import subprocess
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    assert "no tag" in evaluate.provenance_section(tmp_path)


def test_three_tests_base_rw():
    d = pd.date_range("2020-01-01", periods=10)
    rw = pd.DataFrame({"date": d, "y_true": [1.0] * 10, "y_pred": [3.0] * 10,
                       "class_true": [0, 1] * 5, "class_pred": [1, 0] * 5})
    new = rw.assign(y_pred=1.1, class_pred=rw["class_true"])
    p = evaluate._three_tests({"rw": rw, "x": new}, {"rw": rw, "x": new}, "x", base="rw")
    assert all(v < 0.05 for v in p)
