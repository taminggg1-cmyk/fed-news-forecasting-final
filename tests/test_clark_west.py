import numpy as np
import pandas as pd

from fed_news.evaluation import clark_west as cw


def test_clark_west_matches_formula():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    small = y + rng.normal(scale=1.0, size=200)
    large = y + rng.normal(scale=0.5, size=200)
    t, p = cw.clark_west(y, small, large)
    f = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
    assert np.isclose(t, f.mean() / (f.std(ddof=1) / np.sqrt(len(f))))
    assert t > 2 and p < 0.05


def test_clark_west_identical_forecasts_give_no_evidence():
    y = np.array([1.0, 2.0, 3.0, 2.5])
    pred = np.array([1.1, 1.9, 3.2, 2.4])
    assert cw.clark_west(y, pred, pred) == (0.0, 0.5)


def test_compare_checks_dates(tmp_path):
    dates = pd.date_range("2001-01-01", periods=5, freq="MS")
    base = pd.DataFrame({"date": dates, "y_true": 1.0, "y_pred": 1.1})
    other = base.copy()
    other["date"] = pd.date_range("2002-01-01", periods=5, freq="MS")
    try:
        cw.compare(base, other)
    except ValueError as err:
        assert "dates" in str(err)
    else:
        raise AssertionError("mismatched dates must raise")


def test_main_writes_table(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    for task, n in (("T1", 40), ("T2", 30)):
        dates = pd.date_range("2001-01-01", periods=n, freq="MS")
        y = rng.normal(size=n)
        for name, scale in (("S", 1.0), ("S+N", 0.5)):
            pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(scale=scale, size=n)}).to_csv(
                tmp_path / f"preds_{task}_{name}.csv", index=False)
    monkeypatch.setattr(cw.config, "RESULTS", tmp_path)
    table = cw.main()
    assert list(table["task"]) == ["T1", "T2"]
    assert {"n", "dm_hln_p", "clark_west_t", "clark_west_p"} <= set(table.columns)
    assert (tmp_path / "exploratory" / "clark_west.csv").exists()
