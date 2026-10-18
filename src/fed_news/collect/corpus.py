"""The Fed's own documents, each with the date it became public (spec section 5, N).

Statements and press conferences are public on the decision day; minutes on their stated
release date; Beige Books on their publication date; speeches on the day delivered. A document
without a usable release date, or with under MIN_DOC_CHARS characters of text, is dropped and
counted. Every text is cached by URL, so a rebuild is offline.
"""
import hashlib
import html as htmllib
import io
import json
import re

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from fed_news import config
from fed_news.collect.fomc_calendar import BASE, load_calendar
from fed_news.http import get

DOC_COLS = ["doc_id", "type", "speaker", "release", "text"]
TYPE_OF_KIND = {"statement": "statement", "minutes": "minutes", "presconf": "presconf", "beige": "beige"}
PRESCONF_PDF = BASE + "/mediacenter/files/FOMCpresconf{:%Y%m%d}.pdf"
SPEECH_JSON = BASE + "/json/ne-speeches.json"
OLD_SPEECH_LIST = BASE + "/newsevents/speech/{}speech.htm"
OLD_SPEECH_LINK = re.compile(
    r'href="((?:https?://www\.federalreserve\.gov)?/boarddocs/speeches/\d{4}/(\d{8})\d*(?:/default)?\.htm)"', re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
MONTH_DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|"
                        r"November|December)" + r"\s+(\d{1,2}),\s+(\d{4})")
RELEASE_DATE = re.compile(r"Release Date:\s*(January|February|March|April|May|June|July|August|September|"
                          r"October|November|December)\s+(\d{1,2}),\s+(\d{4})", re.I)
BEIGE_ARCHIVE = BASE + "/monetarypolicy/beigebook{}.htm"
BEIGE_CURRENT = BASE + "/monetarypolicy/publications/beige-book-default.htm"
CAL_BEIGE_DATE = (r"files/fomc\d{8}beige(\d{8})\.pdf|/fomc/beigebook/\d{4}/(\d{8})/default\.htm")
ARCHIVE_ROW = re.compile(r"<tr\b.*?</tr>", re.I | re.S)
ARCHIVE_DAY = re.compile(r"\s*([A-Z][a-z]+ \d{1,2})(?!\d)")
ARCHIVE_DATE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
ARCHIVE_LINK = re.compile(r'<a\s[^>]*href="([^"]+)"[^>]*>\s*(HTML|PDF)\b', re.I)


class CoverageError(RuntimeError):
    pass


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    node = (soup.find(id="article") or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            or soup.body or soup)
    return re.sub(r"\s+", " ", node.get_text(" ")).strip()


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages)).strip()


def split_sentences(text: str) -> list[str]:
    parts = (s.strip() for s in SENT_SPLIT.split(text))
    return [s for s in parts if 30 <= len(s) <= 1000]


def _doc_id(doc_type: str, release: pd.Timestamp, url: str) -> str:
    return f"{doc_type}:{release:%Y%m%d}:{hashlib.sha1(url.encode()).hexdigest()[:8]}"


def _text(url: str) -> str:
    return pdf_to_text(get(url, binary=True)) if url.lower().endswith(".pdf") else html_to_text(get(url))


def _fetch_text(url: str, presconf_date=None) -> str | None:
    """Text of one document; None when it is missing or its PDF cannot be parsed. Network errors raise."""
    try:
        if presconf_date is not None:
            try:
                return pdf_to_text(get(PRESCONF_PDF.format(presconf_date), binary=True))
            except FileNotFoundError:
                pass
        return _text(url)
    except (FileNotFoundError, PyPdfError):
        return None


def statement_header_date(text: str) -> pd.Timestamp:
    """Date printed on a statement: 'Release Date: Month D, YYYY', else the first date in the
    first 400 characters, else NaT."""
    m = RELEASE_DATE.search(text) or MONTH_DATE.search(text[:400])
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    except ValueError:
        return pd.NaT


def _statement_release(url: str, text: str, release: pd.Timestamp) -> pd.Timestamp:
    """Never earlier than the URL date; moved later when the statement header says it was issued later
    (the 2007-06-28 statement sits at a 20070618 URL)."""
    header = statement_header_date(text)
    if pd.notna(header) and header > release:
        print(f"statement header date {header:%Y-%m-%d} differs from URL date {release:%Y-%m-%d}: {url}",
              flush=True)
        return header
    return release


def _make_docs(rows_in, doc_type: str, adjust=None) -> tuple[pd.DataFrame, int]:
    """rows_in: iterable of (url, release, speaker, presconf_date). Unusable documents are counted.
    adjust(url, text, release) may return a corrected release date."""
    rows, dropped = [], 0
    for url, release, speaker, pdate in rows_in:
        text = _fetch_text(url, pdate)
        if text is None or pd.isna(release) or len(text) < config.MIN_DOC_CHARS:
            dropped += 1
            continue
        if adjust is not None:
            release = adjust(url, text, release)
        rows.append({"doc_id": _doc_id(doc_type, release, url), "type": doc_type,
                     "speaker": speaker, "release": pd.Timestamp(release), "text": text})
    return pd.DataFrame(rows, columns=DOC_COLS), dropped


def calendar_beige_listing(cal: pd.DataFrame) -> pd.DataFrame:
    """Calendar Beige rows whose URL carries the publication date (8 digits): 1994-2010."""
    sub = cal[cal["kind"] == "beige"]
    found = sub["url"].str.extract(CAL_BEIGE_DATE)
    day = found[0].fillna(found[1])
    sub = sub[day.notna()]
    return pd.DataFrame({"url": sub["url"], "release": pd.to_datetime(day.dropna(), format="%Y%m%d")})


def docs_from_calendar(cal: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int]:
    """Calendar rows of one kind as documents. Beige rows without a date in the URL are not used
    here; build_beige fills them from the Beige Book archive pages."""
    if kind == "beige":
        listing = calendar_beige_listing(cal)
        return _make_docs(((r.url, r.release, "", None) for r in listing.itertuples()), "beige")
    sub = cal[cal["kind"] == kind]
    pdate = (lambda r: r.date) if kind == "presconf" else (lambda r: None)
    adjust = _statement_release if kind == "statement" else None
    return _make_docs(((r.url, r.release, "", pdate(r)) for r in sub.itertuples()), TYPE_OF_KIND[kind], adjust)


def parse_beige_archive(html: str, year: int) -> pd.DataFrame:
    """Rows of a yearly archive page: 'Month D' plus the page year, and the PDF (the full report; else HTML) link.
    Rows with an unparseable date or no link are skipped; their number is in frame.attrs['skipped']."""
    rows, skipped = [], 0
    for row in ARCHIVE_ROW.findall(html):
        label = ARCHIVE_DATE.search(row)
        if not label:
            continue  # header row
        links = {kind.upper(): href for href, kind in ARCHIVE_LINK.findall(row)}
        link = links.get("PDF") or links.get("HTML")
        try:
            release = pd.to_datetime(f"{ARCHIVE_DAY.match(label.group(1)).group(1)} {year}", format="%B %d %Y")
        except (ValueError, AttributeError):
            release = None
        if release is None or not link:
            skipped += 1
            continue
        rows.append({"url": BASE + link if link.startswith("/") else link, "release": release})
    frame = pd.DataFrame(rows, columns=["url", "release"])
    frame.attrs["skipped"] = skipped
    return frame


def merge_beige(*listings: pd.DataFrame) -> pd.DataFrame:
    """Union of listings in priority order; one row per release date, the earliest listing wins."""
    return pd.concat(listings, ignore_index=True).drop_duplicates("release").sort_values("release")


def build_beige(cal: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Beige Books from, in priority order, calendar URLs with a date, yearly archive pages
    (1996 to last year) and the current-year page. Release date = the date on the source row."""
    listings, skipped = [calendar_beige_listing(cal)], 0
    this_year = pd.Timestamp.today().year
    for year in range(1996, this_year):
        try:
            page = parse_beige_archive(get(BEIGE_ARCHIVE.format(year)), year)
        except FileNotFoundError:
            continue
        listings.append(page)
        skipped += page.attrs["skipped"]
    current = parse_beige_archive(get(BEIGE_CURRENT, cache=False), this_year)
    listings.append(current)
    listing = merge_beige(*listings)
    docs, dropped = _make_docs(((r.url, r.release, "", None) for r in listing.itertuples()), "beige")
    # current-page rows without a link are future meetings, not dropped documents
    return docs, dropped + skipped


def parse_speech_json(raw: str) -> tuple[pd.DataFrame, int]:
    """(speeches, number of items skipped for a missing link or date)."""
    rows, skipped = [], 0
    for item in json.loads(raw.lstrip("\ufeff")):
        link, stamp = item.get("l", ""), item.get("d", "")
        if not link or not stamp:
            skipped += 1
            continue
        rows.append({"url": BASE + link if link.startswith("/") else link,
                     "release": pd.to_datetime(stamp.split()[0], format="%m/%d/%Y"),
                     "speaker": htmllib.unescape(item.get("s", ""))})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"]), skipped


def parse_old_speech_list(page: str) -> pd.DataFrame:
    rows = []
    for link, day in OLD_SPEECH_LINK.findall(page):
        url = link.replace("http://", "https://")
        rows.append({"url": url if url.startswith("https") else BASE + url,
                     "release": pd.to_datetime(day, format="%Y%m%d"), "speaker": ""})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"]).drop_duplicates("url")


def fetch_speeches() -> tuple[pd.DataFrame, int]:
    first, skipped = parse_speech_json(get(SPEECH_JSON, cache=False))
    listing = [first]
    for year in range(1996, 2006):
        try:
            listing.append(parse_old_speech_list(get(OLD_SPEECH_LIST.format(year))))
        except FileNotFoundError:
            continue
    listing = pd.concat(listing, ignore_index=True).drop_duplicates("url")
    docs, dropped = _make_docs(((r.url, r.release, r.speaker, None) for r in listing.itertuples()), "speech")
    return docs, dropped + skipped


def coverage(docs: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(docs["release"].dt.year, docs["type"])


def check_coverage(cov: pd.DataFrame, years, required) -> None:
    missing = [f"{year}:{t}" for year in years for t in required
               if t not in cov.columns or year not in cov.index or cov.loc[year, t] == 0]
    if missing:
        raise CoverageError("no documents for " + ", ".join(missing))


def build_corpus(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "corpus.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    cal = load_calendar()
    parts, dropped = [], {}
    for kind in ("statement", "minutes", "presconf"):
        docs, n = docs_from_calendar(cal, kind)
        parts.append(docs)
        dropped[kind] = n
        print(f"{kind:9s} kept {len(docs):5d} dropped {n}", flush=True)
    beige, n = build_beige(cal)
    parts.append(beige)
    dropped["beige"] = n
    print(f"beige     kept {len(beige):5d} dropped {n}", flush=True)
    speeches, n = fetch_speeches()
    parts.append(speeches)
    dropped["speech"] = n
    print(f"speech    kept {len(speeches):5d} dropped {n}", flush=True)
    docs = pd.concat(parts, ignore_index=True).drop_duplicates("doc_id")
    docs = docs[docs["release"] >= pd.Timestamp("1993-06-01")].sort_values("release").reset_index(drop=True)
    docs.to_pickle(path)
    config.RESULTS.mkdir(exist_ok=True)
    coverage(docs).to_csv(config.RESULTS / "coverage.csv")
    (config.RESULTS / "dropped_docs.json").write_text(json.dumps(dropped, indent=1))
    return docs


if __name__ == "__main__":
    corpus = build_corpus(refresh=True)
    cov = coverage(corpus)
    print(cov.to_string())
    check_coverage(cov, range(2001, corpus["release"].dt.year.max() + 1), ("statement", "minutes"))
    print("coverage check passed")
