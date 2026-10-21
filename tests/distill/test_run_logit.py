import numpy as np
import pandas as pd

from fed_news.distill import run_logit as dl
from fed_news.features.panels import M_COLS, S_COLS


def _preds(dates, rng, y_pred_noise):
    n = len(dates)
    y = np.linspace(5, 6, n)
    cls = np.arange(n) % 3
    p = rng.dirichlet([2, 2, 2], n)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, y_pred_noise, n), "lag": y,
                         "class_true": cls, "class_pred": p.argmax(1), "p_cut": p[:, 0], "p_hold": p[:, 1],
                         "p_hike": p[:, 2]})


def test_specs_use_the_haiku_panel_for_s_m_and_the_distilled_panel_for_the_rest():
    s = dl.specs()
    assert set(s) == {"S+M", "S+M+Hd", "placebo-Hd", "shuffled-Hd"}
    assert s["S+M"][0] == "H" and s["S+M"][1].cols == S_COLS + M_COLS
    assert all(s[k][0] == "Hd" and s[k][1].cols == S_COLS + M_COLS + ["n_stance", "n_stance_chg"]
               for k in ("S+M+Hd", "placebo-Hd", "shuffled-Hd"))
    assert s["placebo-Hd"][1].placebo and s["shuffled-Hd"][1].shuffle_labels


def test_verdict_is_one_test_without_correction():
    ok = dl.verdict(0.03, False)
    assert ok["decision"] == "supported" and "p = 0.03" in ok["wording"] and "exploratory" in ok["wording"]
    assert "Holm" not in ok["wording"]
    no = dl.verdict(0.2, False)
    assert no["decision"] == "not supported" and "not significant at 5%" in no["wording"]
    assert dl.verdict(0.01, True)["decision"] == "void"


def test_evaluate_scores_only_the_meetings_from_the_start_and_reads_the_controls():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2008-01-01", periods=48, freq="MS")
    preds = {"S+M": _preds(dates, rng, 0.5), "S+M+Hd": _preds(dates, rng, 0.2),
             "placebo-Hd": _preds(dates, rng, 0.5), "shuffled-Hd": _preds(dates, rng, 0.9)}
    table, res = dl.evaluate(preds, "2010-01-01")
    assert res["n_test_meetings"] == 24 and res["first"] == "2010-01-01"
    t = table.set_index("model")
    assert np.isclose(res["primary_clark_west_p"], t.loc["S+M+Hd", "clark_west_p"])
    assert res["placebo_cw_p"] == t.loc["placebo-Hd", "clark_west_p"]
    assert res["shuffled_cw_p"] == t.loc["shuffled-Hd", "clark_west_p"]
    assert res["decision"] in {"supported", "not supported", "void"}
    _, full = dl.evaluate(preds, None)
    assert full["n_test_meetings"] == 48
