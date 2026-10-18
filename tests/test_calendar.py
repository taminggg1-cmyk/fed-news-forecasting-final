import pandas as pd
import pytest

from fed_news.collect import fomc_calendar

SNIPPET = """
<p><a href="/fomc/minutes/20050322.htm">Minutes</a> (Released Apr 12, 2005)</p>
<a href="http://www.federalreserve.gov/fomc/beigebook/2005/20050119/default.htm">Beige</a>
<a href="/boarddocs/press/monetary/2005/20050202/">Statement</a>
<p>Minutes (Released Jul 16, 2008): <br /><a href="/monetarypolicy/fomc20080625.htm">HTML</a> | <a href="/monetarypolicy/files/fomcminutes20080625.pdf">310 KB PDF</a> </p>
<a href="/newsevents/press/monetary/20081216b.htm">Statement</a>
<div class="col-xs-12 col-md-4 col-lg-4 fomc-meeting__minutes"> <strong>Minutes:</strong><br> <a href="/monetarypolicy/files/fomcminutes20220126.pdf">PDF</a> | <a href="/monetarypolicy/fomcminutes20220126.htm">HTML</a> <br> (Released February 16, 2022) </div>
<a href="/monetarypolicy/fomcpresconf20220126.htm">Press Conference</a>
<a href="/newsevents/pressreleases/monetary20220126a.htm">Statement</a>
<a href="/monetarypolicy/beigebook202201.htm">Beige Book</a>
<p><a href="/fomc/MINUTES/1994/19940204min.htm">Minutes</a> Released March 25, 1994</p>
"""


def test_parse_finds_every_kind_with_dates():
    cal = fomc_calendar.parse_calendar_html(SNIPPET)
    got = {(r.kind, r.date.strftime("%Y%m%d")) for r in cal.itertuples()}
    assert ("minutes", "20050322") in got and ("minutes", "20220126") in got
    assert ("minutes", "19940204") in got and ("statement", "20050202") in got
    assert ("statement", "20220126") in got and ("presconf", "20220126") in got
    assert ("minutes", "20080625") in got and ("statement", "20081216") in got
    assert ("beige", "20050119") in got and ("beige", "20220101") in got


def test_minutes_release_dates_are_parsed():
    cal = fomc_calendar.parse_calendar_html(SNIPPET).set_index(["kind", "date"])
    assert cal.loc[("minutes", pd.Timestamp("2005-03-22")), "release"] == pd.Timestamp("2005-04-12")
    assert cal.loc[("minutes", pd.Timestamp("2008-06-25")), "release"] == pd.Timestamp("2008-07-16")
    assert cal.loc[("minutes", pd.Timestamp("2022-01-26")), "release"] == pd.Timestamp("2022-02-16")
    assert cal.loc[("minutes", pd.Timestamp("1994-02-04")), "release"] == pd.Timestamp("1994-03-25")


def test_six_digit_beige_has_unknown_release_and_statement_release_is_its_date():
    cal = fomc_calendar.parse_calendar_html(SNIPPET).set_index(["kind", "date"])
    assert pd.isna(cal.loc[("beige", pd.Timestamp("2022-01-01")), "release"])
    assert cal.loc[("statement", pd.Timestamp("2022-01-26")), "release"] == pd.Timestamp("2022-01-26")


def test_implausible_release_is_dropped_to_nat():
    html = '<p><a href="/fomc/minutes/20050202.htm">M</a> Released Feb 23, 2011</p>'
    cal = fomc_calendar.parse_calendar_html(html)
    assert pd.isna(cal["release"].iloc[0])


def test_release_is_not_borrowed_from_neighbouring_block():
    html = ('<p><a href="/fomc/MINUTES/1994/19940204min.htm">M</a> (Released Mar 25, 1994)</p>'
            '<p><a href="/fomc/MINUTES/1994/19940322min.htm">M</a></p>')
    cal = fomc_calendar.parse_calendar_html(html).set_index("date")
    assert cal.loc[pd.Timestamp("1994-02-04"), "release"] == pd.Timestamp("1994-03-25")
    assert pd.isna(cal.loc[pd.Timestamp("1994-03-22"), "release"])


def test_release_not_read_from_closed_previous_block():
    html = ('<p><a href="/fomc/MINUTES/1994/19940204min.htm">M</a> (Released Mar 25, 1994)</p>'
            '<a href="/fomc/MINUTES/1994/19940322min.htm">M</a></p>')
    cal = fomc_calendar.parse_calendar_html(html).set_index("date")
    assert cal.loc[pd.Timestamp("1994-02-04"), "release"] == pd.Timestamp("1994-03-25")
    assert pd.isna(cal.loc[pd.Timestamp("1994-03-22"), "release"])


@pytest.mark.network
def test_live_calendar_2005_minutes_release():
    cal = fomc_calendar.load_calendar(refresh=True).set_index(["kind", "date"])
    assert cal.loc[("minutes", pd.Timestamp("2005-03-22")), "release"] == pd.Timestamp("2005-04-12")
