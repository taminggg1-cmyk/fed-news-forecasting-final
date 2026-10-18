import pandas as pd
import pytest

from fed_news.collect import meetings

T = pd.Timestamp


def cal(rows):
    return pd.DataFrame([{"kind": k, "date": T(d), "release": T(d), "url": ""} for k, d in rows])


def target(changes, start="2005-01-01", end="2005-04-30", level=2.0):
    idx = pd.date_range(start, end, freq="D")
    s = pd.Series(level, index=idx)
    for day, new in changes:
        s.loc[T(day):] = new
    return s


def test_two_day_meeting_is_merged_and_classified():
    c = cal([("minutes", "2005-02-01"), ("minutes", "2005-02-02"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([("2005-02-02", 2.25)]))
    assert list(m.index) == [T("2005-02-02"), T("2005-03-22")]
    assert m.loc["2005-02-02", "y_class"] == 2 and m.loc["2005-03-22", "y_class"] == 1
    assert m.loc["2005-02-02", "cutoff"] == T("2005-02-01")
    assert m.loc["2005-02-02", "y_change"] == pytest.approx(0.25)


def test_target_change_without_calendar_entry_is_unscheduled():
    c = cal([("minutes", "2005-02-02"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([("2005-03-01", 1.75)]))
    assert T("2005-03-01") in m.index
    assert m.loc["2005-03-01", "unscheduled"] == 1 and m.loc["2005-03-01", "y_class"] == 0


def test_sunday_unscheduled_cut():
    c = cal([("minutes", "2020-03-03"), ("statement", "2020-03-15"), ("minutes", "2020-04-29")])
    s = target([("2020-03-04", 1.125), ("2020-03-16", 0.125)], "2020-01-01", "2020-05-31", 1.625)
    m = meetings.build_meetings(c, s)
    assert m.loc["2020-03-15", "y_class"] == 0
    assert m.loc["2020-03-15", "target_before"] == pytest.approx(1.125)
    assert m.loc["2020-03-15", "target_after"] == pytest.approx(0.125)
    assert m.loc["2020-03-15", "unscheduled"] == 1


def test_statement_only_date_is_not_a_meeting():
    c = cal([("minutes", "2005-02-02"), ("statement", "2005-03-10"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([]))
    assert T("2005-03-10") not in m.index
    assert list(m.index) == [T("2005-02-02"), T("2005-03-22")]


def test_unscheduled_move_uses_statement_date():
    c = cal([("minutes", "2005-02-02"), ("statement", "2005-03-01"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([("2005-03-02", 1.75)]))
    assert T("2005-03-01") in m.index and T("2005-03-02") not in m.index
    assert m.loc["2005-03-01", "unscheduled"] == 1
