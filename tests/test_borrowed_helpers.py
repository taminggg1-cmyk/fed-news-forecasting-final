"""Tests for the helpers copied from earlier designs (features/haiku_prompt.py, features/strict_columns.py,
features/swap_tone.py, models/neural.standardise, evaluation/significance.nw_summary). The test functions are
copied unchanged from tests/test_poc_llm_labels.py, test_poc_haiku_leak.py, test_naive_stance.py and
test_study5.py in fed-news-forecasting (ad2183c); only the module names they use are new."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fed_news.evaluation import significance
from fed_news.features import haiku_prompt as pl, strict_columns, swap_tone as ns

hl = pl  # the leak-test masking now lives in haiku_prompt too
s5 = SimpleNamespace(nw_summary=significance.nw_summary, check_vintages=strict_columns.check_vintages)
NAMES = {"bernanke", "kohn"}


def test_prompt_numbers_sentences_and_sends_text_only():
    prompt = pl.batch_prompt(["inflation is elevated.", "growth has slowed."])
    assert "1. inflation is elevated." in prompt and "2. growth has slowed." in prompt


def test_parse_labels_maps_to_dataset_codes():
    # labels.LABELS: dovish 0, hawkish 1, neutral 2
    assert pl.parse_labels('{"1": "H", "2": "D", "3": "N"}', 3) == [1, 0, 2]


def test_parse_labels_accepts_text_around_the_json():
    assert pl.parse_labels('Here you go:\n```json\n{"1": "d", "2": "H"}\n```', 2) == [0, 1]


def test_parse_labels_rejects_missing_or_unknown_labels():
    with pytest.raises(ValueError):
        pl.parse_labels('{"1": "H"}', 2)
    with pytest.raises(ValueError):
        pl.parse_labels('{"1": "X"}', 1)


def test_mask_hides_years_months_numbers_names_and_events():
    s = ("in september 2008, after lehman failed, chairman bernanke said the rate of 2 percent "
         "and inflation of 3-1/2 percent were consistent with easing.")
    m = hl.mask(s, NAMES)
    for gone in ("2008", "september", "lehman", "bernanke", "3-1/2", " 2 "):
        assert gone not in m
    assert "inflation" in m and "easing" in m and "[YEAR]" in m and "[NAME]" in m and "[EVENT]" in m


def test_mask_keeps_may_and_march_as_ordinary_words_but_not_as_dates():
    assert hl.mask("the committee may decide to march ahead", NAMES) == "the committee may decide to march ahead"
    assert "[MONTH]" in hl.mask("in may the committee met", NAMES)
    assert "[MONTH]" in hl.mask("since march [NUM]", NAMES)
    assert "pandemic" not in hl.mask("pandemic-related disruptions eased", NAMES)


def test_naive_panel_replaces_only_stance_columns(monkeypatch, tmp_path):
    idx = pd.date_range("2005-01-01", periods=4, freq="MS")
    rows = pd.DataFrame({"win_start": idx - pd.offsets.MonthBegin(1), "cutoff": idx}, index=idx)
    full = rows.assign(n_stance=9.0, n_stance_chg=9.0, n_beige=1.0, s_rate_lag=2.0)
    rows.to_pickle(tmp_path / "t1.pkl")
    full.to_pickle(tmp_path / "t1_full.pkl")
    monkeypatch.setattr(ns.config, "DATA", tmp_path)
    ds = pd.DataFrame({"doc_id": ["a", "b"], "type": ["speech", "speech"],
                       "release": pd.to_datetime(["2004-12-15", "2005-02-10"]), "stance": [0.2, 0.6], "n_sent": [1, 1]})
    out = ns.naive_panel("t1", ds)
    assert np.allclose(out["n_stance"], [0.2, 0.2, 0.6, 0.6])             # carried forward when a month is empty
    assert np.isnan(out["n_stance_chg"].iloc[0]) and np.allclose(out["n_stance_chg"].iloc[1:], [0.0, 0.4, 0.0])
    assert (out["n_beige"] == 1.0).all()


def test_newey_west_interval_agrees_with_the_one_sided_test():
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.05, 0.2):
        r = s5.nw_summary(shift + rng.normal(0, 0.3, 200), lags=8)
        assert (r["p_one_sided"] < 0.05) == (r["ci90"][0] > 0)


def test_fallback_check_refuses_a_series_whose_vintages_start_too_late():
    first = {"m2": pd.Timestamp("1980-02-08"), "u6": pd.Timestamp("2011-03-04")}
    s5.check_vintages(first, ["m2"], pd.Timestamp("1995-03-27"))
    with pytest.raises(ValueError):
        s5.check_vintages(first, ["m2", "u6"], pd.Timestamp("1995-03-27"))
