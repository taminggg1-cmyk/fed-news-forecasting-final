import numpy as np
import pandas as pd

from fed_news.features.panels import M_COLS, S_COLS
from fed_news.studies.coral import run_logit as rl
from fed_news.studies.coral.from2010 import subset
from fed_news.studies.doctype import run_logit as tl
from fed_news.studies.doctype.group_tone import HS


def _preds(dates, rng, y_pred_noise):
    n = len(dates)
    y = np.linspace(5, 6, n)
    cls = np.arange(n) % 3
    p = rng.dirichlet([2, 2, 2], n)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, y_pred_noise, n), "lag": y,
                         "class_true": cls, "class_pred": p.argmax(1), "p_cut": p[:, 0], "p_hold": p[:, 1],
                         "p_hike": p[:, 2]})


def test_specs_use_the_haiku_panel_for_s_m_and_the_split_panel_for_the_rest():
    s = tl.specs()
    assert set(s) == {"S+M", "S+M+Hs", "placebo-Hs", "shuffled-Hs"}
    assert s["S+M"][0] == "H" and s["S+M"][1].cols == S_COLS + M_COLS
    assert all(s[k][0] == "Hs" and s[k][1].cols == S_COLS + M_COLS + HS for k in ("S+M+Hs", "placebo-Hs", "shuffled-Hs"))
    assert s["placebo-Hs"][1].placebo and s["placebo-Hs"][1].news_cols == HS
    assert s["shuffled-Hs"][1].shuffle_labels and not s["shuffled-Hs"][1].placebo


def test_verdict_is_one_test_without_correction():
    ok = tl.verdict(0.03, False)
    assert ok["decision"] == "supported" and "p = 0.03" in ok["wording"] and "exploratory" in ok["wording"]
    assert "split by document type" in ok["wording"] and "Holm" not in ok["wording"]
    no = tl.verdict(0.2, False)
    assert no["decision"] == "not supported" and "not significant at 5%" in no["wording"]
    assert tl.verdict(0.01, True)["decision"] == "void"


def test_evaluate_reads_the_primary_test_the_controls_and_the_split_against_the_single_tone():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2008-01-01", periods=48, freq="MS")
    preds = {"S+M": _preds(dates, rng, 0.5), "S+M+Hs": _preds(dates, rng, 0.2),
             "placebo-Hs": _preds(dates, rng, 0.5), "shuffled-Hs": _preds(dates, rng, 0.9),
             "S+M+Hd": _preds(dates, rng, 0.3)}
    table, res = tl.evaluate(preds, "2010-01-01")
    assert res["n_test_meetings"] == 24 and res["first"] == "2010-01-01"
    t = table.set_index("model")
    assert np.isclose(res["primary_clark_west_p"], t.loc["S+M+Hs", "clark_west_p"])
    assert res["placebo_cw_p"] == t.loc["placebo-Hs", "clark_west_p"]
    assert res["shuffled_cw_p"] == t.loc["shuffled-Hs", "clark_west_p"]
    sub = subset(preds, "2010-01-01")
    want = rl.compare_row(sub["S+M+Hd"], sub["S+M+Hs"])
    assert np.isclose(res["split_vs_single"]["clark_west_p"], want["clark_west_p"])
    assert np.isclose(res["split_vs_single"]["dm_logloss_p"], want["dm_logloss_p"])
    _, full = tl.evaluate(preds, None)
    assert full["n_test_meetings"] == 48
