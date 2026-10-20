import numpy as np
import pandas as pd

from fed_news.coral import run_logit as rl
from fed_news.features.panels import M_COLS, S_COLS


def test_holm_over_the_two_primary_clark_west_tests():
    assert rl.primary_holm(0.02, 0.04) == {"S+M+Co": 0.04, "S+M+Ho": 0.04}
    assert rl.primary_holm(0.01, 0.30) == {"S+M+Co": 0.02, "S+M+Ho": 0.30}


def test_void_rule_uses_clark_west_thresholds():
    assert rl.void(0.04, 0.5) is True       # placebo beats S+M at p < 0.05
    assert rl.void(0.2, 0.005) is True      # shuffled control beats S+M at p < 0.01
    assert rl.void(0.2, 0.02) is False


def test_specs_pair_each_tone_with_its_own_controls():
    s = rl.specs()
    assert set(s) == {"S+M", "S+M+Co", "S+M+Ho", "placebo-Co", "placebo-Ho", "shuffled-Co", "shuffled-Ho"}
    assert s["S+M"][1].cols == S_COLS + M_COLS
    for tone in ("Co", "Ho"):
        assert s[f"S+M+{tone}"][0] == s[f"placebo-{tone}"][0] == s[f"shuffled-{tone}"][0] == tone
        assert s[f"placebo-{tone}"][1].placebo and s[f"shuffled-{tone}"][1].shuffle_labels
        assert s[f"S+M+{tone}"][1].cols == S_COLS + M_COLS + ["n_stance", "n_stance_chg"]


def test_verdict_follows_the_design_wording():
    ok = rl.verdict("ChronoBERT-CORAL", 0.03, False)
    assert ok["decision"] == "supported" and "exploratory" in ok["wording"] and "p = 0.03" in ok["wording"]
    assert rl.verdict("Haiku ordinal", 0.20, False)["decision"] == "not supported"
    assert "not significant at 5%" in rl.verdict("Haiku ordinal", 0.20, False)["wording"]
    assert rl.verdict("Haiku ordinal", 0.01, True)["decision"] == "void"


def test_common_rows_keeps_meetings_present_in_every_panel():
    idx = pd.date_range("2001-01-01", periods=5, freq="MS")
    pans = {"H": pd.DataFrame({"x": range(5)}, index=idx), "Co": pd.DataFrame({"x": range(4)}, index=idx[1:]),
            "Ho": pd.DataFrame({"x": range(4)}, index=idx[:4])}
    out = rl.common(pans)
    assert all(p.index.equals(idx[1:4]) for p in out.values())


def test_comparison_row_reports_clark_west_and_log_loss_tests():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2001-01-01", periods=60, freq="MS")
    y = rng.normal(5, 1, 60)
    cls = rng.integers(0, 3, 60)
    def preds(noise):
        p = rng.dirichlet([2, 2, 2], 60)
        return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, noise, 60), "lag": y,
                             "class_true": cls, "class_pred": p.argmax(1), "p_cut": p[:, 0], "p_hold": p[:, 1],
                             "p_hike": p[:, 2]})
    row = rl.compare_row(preds(0.5), preds(0.3))
    for key in ("clark_west_t", "clark_west_p", "dm_logloss_p", "nw4_logloss_p", "nw8_logloss_p", "log_loss",
                "accuracy", "brier", "mcnemar_p"):
        assert key in row and np.isfinite(row[key])


def test_haiku_verdict_carries_the_memory_caveat():
    assert "memory cannot be ruled out" in rl.verdict("Haiku ordinal", 0.03, False, haiku=True)["wording"]
    assert "memory cannot be ruled out" in rl.verdict("Haiku ordinal", 0.30, False, haiku=True)["wording"]
    assert "memory" not in rl.verdict("ChronoBERT-CORAL", 0.03, False)["wording"]


def test_comparison_row_signs_favour_the_better_news_model():
    rng = np.random.default_rng(3)
    n = 120
    dates = pd.date_range("2001-01-01", periods=n, freq="MS")
    y = rng.normal(5, 1, n)
    cls = rng.integers(0, 3, n)
    def preds(noise, sharp):
        p = np.full((n, 3), (1 - sharp) / 2)
        p[np.arange(n), cls] = sharp
        return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, noise, n), "lag": y,
                             "class_true": cls, "class_pred": cls, "p_cut": p[:, 0], "p_hold": p[:, 1],
                             "p_hike": p[:, 2]})
    row = rl.compare_row(preds(1.0, 0.4), preds(0.2, 0.8))
    assert row["clark_west_t"] > 0 and row["clark_west_p"] < 0.05
    assert row["dm_logloss_p"] < 0.05 and row["nw4_logloss_p"] < 0.05


def test_evaluate_gives_the_table_primary_tests_controls_and_memory_flag(monkeypatch):
    rng = np.random.default_rng(4)
    n = 80
    dates = pd.date_range("2001-01-01", periods=n, freq="MS")
    y = rng.normal(5, 1, n)
    cls = rng.integers(0, 3, n)
    def preds():
        p = rng.dirichlet([2, 2, 2], n)
        return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, 0.4, n), "lag": y,
                             "class_true": cls, "class_pred": p.argmax(1), "p_cut": p[:, 0], "p_hold": p[:, 1],
                             "p_hike": p[:, 2]})
    monkeypatch.setattr(rl, "_memory_indicators", lambda d: pd.DataFrame(
        {"recalled": np.arange(n) % 2 == 0, "dated": True, "has_exam": True}, index=pd.DatetimeIndex(d)))
    table, res = rl.evaluate({name: preds() for name in rl.specs()})
    assert list(table["model"]) == list(rl.specs())
    assert set(res["primary_holm_p"]) == {"S+M+Co", "S+M+Ho"}
    assert {"placebo_cw_p", "shuffled_cw_p", "decision", "wording"} <= set(res["S+M+Ho"])
    assert res["memory_flag_haiku"]["n"] == n
