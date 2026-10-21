import numpy as np
import pandas as pd
import pytest

from fed_news.studies.doctype import group_tone as gt


def _rows(cutoffs):
    cut = pd.to_datetime(cutoffs)
    start = [pd.Timestamp("2000-01-01")] + list(cut[:-1])
    return pd.DataFrame({"win_start": start, "cutoff": cut}, index=cut + pd.Timedelta(days=1))


def _docs(rows):
    return pd.DataFrame(rows, columns=["type", "release", "stance"]).assign(
        release=lambda d: pd.to_datetime(d["release"]))


def test_columns_are_three_group_tones_then_their_changes():
    assert gt.HS == ["n_stance_policy", "n_stance_minutes", "n_stance_speech",
                     "n_stance_policy_chg", "n_stance_minutes_chg", "n_stance_speech_chg"]


def test_each_group_tone_is_the_mean_of_that_groups_documents_in_the_window():
    rows = _rows(["2000-02-01", "2000-03-01"])
    docs = _docs([("statement", "2000-01-10", 0.2), ("presconf", "2000-01-11", 0.4), ("minutes", "2000-01-20", -0.1),
                  ("speech", "2000-01-05", 0.3), ("speech", "2000-01-25", 0.1),
                  ("statement", "2000-02-10", -0.2), ("minutes", "2000-02-15", 0.0), ("speech", "2000-02-20", 0.5),
                  ("speech", "2000-03-01", 9.0)])  # released on the cutoff day: not in any window
    out = gt.group_tones(rows, docs)
    assert np.allclose(out["n_stance_policy"], [0.3, -0.2])
    assert np.allclose(out["n_stance_minutes"], [-0.1, 0.0])
    assert np.allclose(out["n_stance_speech"], [0.2, 0.5])


def test_a_group_without_a_document_in_the_window_keeps_its_last_value():
    rows = _rows(["2000-02-01", "2000-03-01", "2000-04-01"])
    docs = _docs([("statement", "2000-01-10", 0.2), ("minutes", "2000-01-20", -0.1), ("speech", "2000-01-25", 0.1),
                  ("speech", "2000-02-20", 0.5), ("statement", "2000-03-10", 0.6)])
    out = gt.group_tones(rows, docs)
    assert np.allclose(out["n_stance_policy"], [0.2, 0.2, 0.6])
    assert np.allclose(out["n_stance_minutes"], [-0.1, -0.1, -0.1])
    assert np.allclose(out["n_stance_speech"], [0.1, 0.5, 0.5])


def test_before_a_groups_first_document_the_combined_tone_of_the_meeting_is_used():
    rows = _rows(["2000-02-01", "2000-03-01", "2000-04-01"])
    docs = _docs([("minutes", "2000-01-20", -0.1), ("speech", "2000-01-25", 0.3),
                  ("minutes", "2000-02-15", 0.2),
                  ("statement", "2000-03-10", 0.6)])
    out = gt.group_tones(rows, docs)
    # combined tone = mean of the group means present: 0.1, then 0.2 (minutes only); row 3 has the first statement
    assert np.allclose(out["n_stance_policy"], [0.1, 0.2, 0.6])
    assert np.allclose(out["n_stance_speech"], [0.3, 0.3, 0.3])


def test_changes_are_differences_from_the_previous_meeting():
    rows = _rows(["2000-02-01", "2000-03-01", "2000-04-01"])
    docs = _docs([("statement", "2000-01-10", 0.2), ("minutes", "2000-01-20", -0.1), ("speech", "2000-01-25", 0.1),
                  ("statement", "2000-02-10", 0.5), ("speech", "2000-03-20", -0.3)])
    out = gt.group_tones(rows, docs)
    assert np.isnan(out["n_stance_policy_chg"].iloc[0])
    assert np.allclose(out["n_stance_policy_chg"].iloc[1:], [0.3, 0.0])
    assert np.allclose(out["n_stance_speech_chg"].iloc[1:], [0.0, -0.4])


def test_rows_out_of_date_order_are_refused():
    rows = _rows(["2000-02-01", "2000-03-01"]).iloc[::-1]
    with pytest.raises(ValueError):
        gt.group_tones(rows, _docs([("minutes", "2000-01-20", -0.1)]))


def test_split_panel_is_the_single_tone_panel_plus_the_six_columns(tmp_path, monkeypatch):
    from fed_news import config
    rows = _rows(["2000-02-01", "2000-03-01"])
    rows.to_pickle(tmp_path / "t2.pkl")
    rows.assign(s_rate_lag=1.0, n_stance=99.0, n_stance_chg=99.0).to_pickle(tmp_path / "t2_full.pkl")
    monkeypatch.setattr(config, "DATA", tmp_path)
    docs = _docs([("statement", "2000-01-10", 0.2), ("minutes", "2000-01-20", -0.1), ("speech", "2000-01-25", 0.1),
                  ("speech", "2000-02-20", 0.5)])
    out = gt.split_panel(docs)
    assert np.allclose(out["n_stance"], [np.mean([0.2, -0.1, 0.1]), 0.5])  # rebuilt from docs, as in attempt 12
    assert out[gt.HS].equals(gt.group_tones(rows, docs))
    assert (out["s_rate_lag"] == 1.0).all()
