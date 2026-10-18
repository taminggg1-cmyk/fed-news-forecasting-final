import pandas as pd
import pytest

from fed_news.collect import alfred

T = pd.Timestamp


def vintages():
    return pd.DataFrame({
        "date": [T("2000-01-01"), T("2000-01-01"), T("2000-02-01")],
        "realtime_start": [T("2000-02-10"), T("2000-03-10"), T("2000-03-10")],
        "realtime_end": [T("2000-03-09"), T("2262-04-01"), T("2262-04-01")],
        "value": [1.0, 1.5, 2.0],
    })


def test_value_released_on_cutoff_day_is_invisible():
    s, fallback = alfred.asof_series(vintages(), T("2000-03-10"), "M", 9)
    assert s.to_dict() == {T("2000-01-01"): 1.0}
    assert fallback is False


def test_revision_visible_the_day_after_release():
    s, _ = alfred.asof_series(vintages(), T("2000-03-11"), "M", 9)
    assert s.to_dict() == {T("2000-01-01"): 1.5, T("2000-02-01"): 2.0}


def test_fallback_before_first_vintage_uses_release_lag():
    s, fallback = alfred.asof_series(vintages(), T("2000-02-10"), "M", 9)
    assert fallback is True
    assert s.to_dict() == {T("2000-01-01"): 1.0}          # Jan 31 + 9 days = Feb 9 < Feb 10
    s, _ = alfred.asof_series(vintages(), T("2000-02-09"), "M", 9)
    assert s.empty                                          # Feb 9 is not strictly before Feb 9


def test_quarterly_fallback_uses_quarter_end():
    obs = pd.DataFrame({"date": [T("2000-01-01")], "realtime_start": [T("2001-01-01")],
                        "realtime_end": [T("2262-04-01")], "value": [100.0]})
    s, _ = alfred.asof_series(obs, T("2000-04-30"), "Q", 31)
    assert s.empty                                          # Mar 31 + 31 days = May 1
    s, _ = alfred.asof_series(obs, T("2000-05-02"), "Q", 31)
    assert len(s) == 1


def test_parse_observations_handles_missing_and_open_end():
    payload = {"observations": [
        {"realtime_start": "2000-02-10", "realtime_end": "9999-12-31", "date": "2000-01-01", "value": "."},
        {"realtime_start": "2000-02-10", "realtime_end": "9999-12-31", "date": "2000-02-01", "value": "2.5"},
    ]}
    frame = alfred.parse_observations(payload)
    assert frame["value"].isna().tolist() == [True, False]
    assert frame["realtime_end"].iloc[0] == T("2262-04-01")


@pytest.mark.network
def test_payems_vintages_download():
    obs = alfred.fetch_vintages("PAYEMS", refresh=True)
    assert obs["realtime_start"].nunique() > 100
    assert obs["date"].min() <= T("1994-01-01")


def test_fetch_failure_does_not_leak_api_key(tmp_path, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.HTTPError("400 for url https://x?api_key=SECRETKEY")

    monkeypatch.setattr(alfred, "get", boom)
    monkeypatch.setattr(alfred, "api_key", lambda: "SECRETKEY")
    monkeypatch.setattr(alfred.config, "DATA", tmp_path)
    with pytest.raises(RuntimeError) as info:
        alfred.fetch_vintages("BAD")
    assert "SECRETKEY" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True
