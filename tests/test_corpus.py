import pandas as pd
import pytest

from fed_news.collect import corpus

T = pd.Timestamp


def test_html_to_text_prefers_article_and_drops_scripts():
    html = ("<html><body><nav>menu</nav><div id='article'><p>The Committee decided.</p>"
            "<script>x=1</script></div><footer>foot</footer></body></html>")
    assert corpus.html_to_text(html) == "The Committee decided."


def test_split_sentences_keeps_reasonable_lengths():
    text = ("Inflation has moved up over the past year and remains elevated. Ok. "
            "The Committee decided to raise the target range for the federal funds rate.")
    assert corpus.split_sentences(text) == [
        "Inflation has moved up over the past year and remains elevated.",
        "The Committee decided to raise the target range for the federal funds rate."]


ARCHIVE_2012 = """<table><thead><tr><th class="alternate" colspan="4" id="year">2012</th></tr></thead><tbody>
<tr class="odd"><td>January 11</td>
<td><a href="https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook201201.htm">HTML</a> | <a href="https://www.federalreserve.gov/monetarypolicy/beigebook/files/fullreport20120111.pdf">PDF</a></td></tr>
<tr class="even"><td>February 29</td>
<td><a href="https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook201202.htm">HTML</a> | <a href="https://www.federalreserve.gov/monetarypolicy/beigebook/files/fullreport20120229.pdf">PDF</a></td></tr>
<tr><td>Sometime soon</td><td><a href="/x.htm">HTML</a></td></tr>
<tr><td>March 7</td><td><a href="/monetarypolicy/files/BeigeBook_20120307.pdf">PDF</a></td></tr>
</tbody></table>"""


def test_parse_beige_archive_uses_row_date_and_pdf_link():
    frame = corpus.parse_beige_archive(ARCHIVE_2012, 2012)
    assert frame["release"].tolist() == [T("2012-01-11"), T("2012-02-29"), T("2012-03-07")]
    assert frame["url"].tolist() == [
        "https://www.federalreserve.gov/monetarypolicy/beigebook/files/fullreport20120111.pdf",
        "https://www.federalreserve.gov/monetarypolicy/beigebook/files/fullreport20120229.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20120307.pdf"]
    assert frame.attrs["skipped"] == 1


def test_parse_current_beige_page_cells():
    html = ('<tr><th id="year">2026</th></tr><tr><td>January 14: <a href="/monetarypolicy/beigebook202601-summary.htm">'
            'HTML</a> | <a href="https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20260114.pdf">PDF</a></td></tr>'
            '<tr><td>March 4: <a href="/monetarypolicy/beigebook202602-summary.htm">HTML</a> | '
            '<a href="/monetarypolicy/files/BeigeBook_20260304.pdf">PDF</a></td></tr>')
    frame = corpus.parse_beige_archive(html, 2026)
    assert frame["release"].tolist() == [T("2026-01-14"), T("2026-03-04")]
    assert frame["url"].tolist() == [
        "https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20260114.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20260304.pdf"]


def test_calendar_beige_takes_url_dates_in_both_patterns(monkeypatch):
    base = "https://www.federalreserve.gov/"
    cal = pd.DataFrame([
        {"kind": "beige", "date": T("1995-01-18"), "release": T("1995-01-18"),
         "url": base + "monetarypolicy/files/fomc19950201beige19950118.pdf"},
        {"kind": "beige", "date": T("2004-01-01"), "release": T("2004-01-14"),
         "url": base + "fomc/beigebook/2004/20040114/default.htm"},
        {"kind": "beige", "date": T("2017-01-01"), "release": T("2017-01-09"),
         "url": base + "monetarypolicy/beigebook201701.htm"}])
    listing = corpus.calendar_beige_listing(cal)
    assert listing["release"].tolist() == [T("1995-01-18"), T("2004-01-14")]
    monkeypatch.setattr(corpus, "_fetch_text", lambda url, d=None: "word " * 200)
    docs, dropped = corpus.docs_from_calendar(cal, "beige")
    assert docs["release"].tolist() == [T("1995-01-18"), T("2004-01-14")] and dropped == 0


def test_merge_beige_keeps_highest_priority_source_per_release_date():
    cal = pd.DataFrame({"url": ["cal-url"], "release": [T("2005-02-02")]})
    archive = pd.DataFrame({"url": ["arch-url", "arch-other"], "release": [T("2005-02-02"), T("2005-03-09")]})
    current = pd.DataFrame({"url": ["cur-url"], "release": [T("2005-03-09")]})
    merged = corpus.merge_beige(cal, archive, current)
    assert merged["url"].tolist() == ["cal-url", "arch-other"]


def test_corrupt_pdf_is_dropped_not_fatal(monkeypatch):
    from pypdf.errors import PdfReadError

    def boom(url, **k):
        raise PdfReadError("bad")
    monkeypatch.setattr(corpus, "get", boom)
    docs, dropped = corpus._make_docs([("http://x/a.pdf", T("2005-01-01"), "", None)], "speech")
    assert dropped == 1 and docs.empty


def test_minutes_without_release_are_dropped(monkeypatch):
    cal = pd.DataFrame([
        {"kind": "minutes", "date": T("2005-02-02"), "release": pd.NaT, "url": "u1"},
        {"kind": "minutes", "date": T("2005-03-22"), "release": T("2005-04-12"), "url": "u2"},
    ])
    monkeypatch.setattr(corpus, "get", lambda url, **k: "<div id='article'>" + "word " * 200 + "</div>")
    docs, dropped = corpus.docs_from_calendar(cal, "minutes")
    assert dropped == 1
    assert docs["release"].tolist() == [T("2005-04-12")]
    assert list(docs.columns) == corpus.DOC_COLS


def test_parse_speech_json():
    raw = ('\ufeff[{"d":"10/8/2026 4:30:00 AM","t":"Title","s":"Governor A &amp; B",'
           '"l":"/newsevents/speech/waller20261008a.htm"}]')
    frame, skipped = corpus.parse_speech_json(raw)
    assert skipped == 0
    assert frame.loc[0, "release"] == T("2026-10-08")
    assert frame.loc[0, "speaker"] == "Governor A & B"
    assert frame.loc[0, "url"].endswith("/newsevents/speech/waller20261008a.htm")


def test_parse_speech_json_date_only_and_missing_fields():
    raw = ('[{"d":"11/28/2006","s":"X","l":"/newsevents/speech/a.htm"},'
           '{"s":"Y","l":"/newsevents/speech/b.htm"},{"d":"1/2/2007"},{"updateDate":"x"}]')
    frame, skipped = corpus.parse_speech_json(raw)
    assert frame["release"].tolist() == [T("2006-11-28")] and skipped == 3


def test_parse_old_speech_list():
    page = ('<a href="/boarddocs/speeches/2002/20021121/default.htm">x</a>'
            '<a href="http://www.federalreserve.gov/boarddocs/speeches/2002/200211212/default.htm">y</a>')
    frame = corpus.parse_old_speech_list(page)
    assert frame["release"].tolist() == [T("2002-11-21"), T("2002-11-21")]
    assert all(u.startswith("https://www.federalreserve.gov/boarddocs/") for u in frame["url"])


def test_check_coverage_raises_on_missing_year():
    docs = pd.DataFrame({"type": ["statement", "minutes", "statement"],
                         "release": [T("2001-03-20"), T("2001-04-12"), T("2002-01-30")]})
    cov = corpus.coverage(docs)
    corpus.check_coverage(cov, [2001], ("statement", "minutes"))
    with pytest.raises(corpus.CoverageError, match="2002"):
        corpus.check_coverage(cov, [2001, 2002], ("statement", "minutes"))


@pytest.mark.network
def test_live_speech_json_is_parseable():
    frame, skipped = corpus.parse_speech_json(corpus.get(corpus.SPEECH_JSON, cache=False))
    assert skipped >= 0 and len(frame) > 500 and frame["release"].min() < T("2008-01-01")


@pytest.mark.network
@pytest.mark.parametrize("year", [2012, 2023])
def test_live_beige_archive_has_eight_rows(year):
    frame = corpus.parse_beige_archive(corpus.get(corpus.BEIGE_ARCHIVE.format(year), cache=False), year)
    assert len(frame) == 8 and (frame["release"].dt.year == year).all()


def test_parse_beige_archive_date_and_links_in_one_cell():
    html = ('<tr><th id="year">2024</th></tr><tr><td>January 17: <a href="/monetarypolicy/'
            'beigebook202401-summary.htm">HTML</a> | <a href="/x/BeigeBook_20240117.pdf">PDF</a></td></tr>')
    frame = corpus.parse_beige_archive(html, 2024)
    assert frame["release"].tolist() == [T("2024-01-17")]
    assert frame["url"].tolist() == ["https://www.federalreserve.gov/x/BeigeBook_20240117.pdf"]


@pytest.mark.network
def test_live_current_beige_page_has_this_years_books():
    year = pd.Timestamp.today().year
    frame = corpus.parse_beige_archive(corpus.get(corpus.BEIGE_CURRENT, cache=False), year)
    assert len(frame) >= 6 and (frame["release"].dt.year == year).all()


def test_parse_old_speech_list_flat_file_links():
    page = ('<a href="/boarddocs/speeches/1999/19991215.htm">x</a>'
            '<a href="http://www.federalreserve.gov/boarddocs/speeches/1999/199910282.htm">y</a>'
            '<a href="/boarddocs/speeches/2002/20021121/default.htm">z</a>')
    frame = corpus.parse_old_speech_list(page)
    assert frame["release"].tolist() == [T("1999-12-15"), T("1999-10-28"), T("2002-11-21")]
    assert frame["url"].tolist()[1] == "https://www.federalreserve.gov/boarddocs/speeches/1999/199910282.htm"


def test_statement_header_date_both_layouts():
    assert corpus.statement_header_date("Release Date: June 28, 2007 The Committee decided") == T("2007-06-28")
    assert corpus.statement_header_date("June 28, 2007 FOMC statement For immediate release") == T("2007-06-28")
    assert pd.isna(corpus.statement_header_date("no date here"))


def _statement_cal(release):
    return pd.DataFrame([{"kind": "statement", "date": release, "release": release, "url": "u"}])


def test_statement_release_moves_to_later_header_date(monkeypatch):
    monkeypatch.setattr(corpus, "get", lambda url, **k: "<div id='article'>June 28, 2007 FOMC statement "
                        + "word " * 200 + "</div>")
    docs, dropped = corpus.docs_from_calendar(_statement_cal(T("2007-06-18")), "statement")
    assert docs["release"].tolist() == [T("2007-06-28")] and dropped == 0
    assert docs["doc_id"].iloc[0].startswith("statement:20070628:")


def test_statement_release_never_moves_earlier(monkeypatch):
    monkeypatch.setattr(corpus, "get", lambda url, **k: "<div id='article'>June 18, 2007 FOMC statement "
                        + "word " * 200 + "</div>")
    docs, _ = corpus.docs_from_calendar(_statement_cal(T("2007-06-28")), "statement")
    assert docs["release"].tolist() == [T("2007-06-28")]
