import numpy as np
import pandas as pd

from fed_news.studies.study4_haiku_logit import haiku_compare as hc


def test_log_loss_and_brier_use_the_true_class_probability():
    preds = pd.DataFrame({"class_true": [0, 1, 2], "p_cut": [0.5, 0.2, 0.1], "p_hold": [0.3, 0.6, 0.1],
                          "p_hike": [0.2, 0.2, 0.8]})
    assert np.allclose(hc.log_loss(preds), -np.log([0.5, 0.6, 0.8]))
    assert np.isclose(hc.brier(preds)[0], 0.5 ** 2 + 0.3 ** 2 + 0.2 ** 2)


def test_memory_flag_fires_when_haikus_extra_gain_sits_on_recalled_meetings():
    rng = np.random.default_rng(0)
    recalled = rng.random(200) < 0.85
    gain_c = rng.normal(0, 0.1, 200)
    gain_h = gain_c + np.where(recalled, 0.15, 0.0) + rng.normal(0, 0.05, 200)
    assert hc.memory_flag(gain_h, gain_c, recalled)["flag"] is True


def test_memory_flag_quiet_when_the_extra_gain_does_not_depend_on_recall():
    rng = np.random.default_rng(1)
    recalled = rng.random(200) < 0.85
    gain_c = rng.normal(0, 0.1, 200)
    gain_h = gain_c + 0.05 + rng.normal(0, 0.05, 200)
    assert hc.memory_flag(gain_h, gain_c, recalled)["flag"] is False


def test_verdict_wording_follows_the_design():
    assert hc.verdict(0.01, 0.4, 0.5, flag=False)["decision"] == "supported"
    assert "no memory pattern" in hc.verdict(0.01, 0.4, 0.5, flag=False)["wording"]
    assert "consistent with memory" in hc.verdict(0.01, 0.4, 0.5, flag=True)["wording"]
    assert hc.verdict(0.2, 0.4, 0.5, flag=False)["decision"] == "not supported"
    assert hc.verdict(0.01, 0.01, 0.5, flag=False)["decision"] == "void"


def test_same_rows_keeps_meetings_present_in_both_panels():
    idx = pd.date_range("2001-01-01", periods=4, freq="MS")
    a, b = pd.DataFrame({"x": range(4)}, index=idx), pd.DataFrame({"x": range(3)}, index=idx[1:])
    ra, rb = hc.same_rows(a, b)
    assert ra.index.equals(idx[1:]) and rb.index.equals(idx[1:])
