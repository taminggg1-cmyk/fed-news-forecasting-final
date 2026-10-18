"""Parse the Fed's FOMC calendar pages into dated links (statements, minutes, press
conferences, Beige Books). Historical pages cover 1994-2020; the current page covers 2021 on.

Minutes are released weeks after the meeting (after the next meeting before 2005), so their
release date is read from the "Released ..." text inside the same <p>/<div> block as the
minutes link (it sits before, after or beside the links depending on the page era). Nothing
outside that block is read, so a neighbour's date is never borrowed. A release date that
cannot be read, or is not within 0-120 days after the meeting, is left as NaT and the document
is later dropped rather than dated by its meeting.
"""
import re

import pandas as pd

from fed_news import config
from fed_news.http import get

BASE = "https://www.federalreserve.gov"
HISTORICAL = BASE + "/monetarypolicy/fomchistorical{}.htm"
CURRENT = BASE + "/monetarypolicy/fomccalendars.htm"

PATTERNS = {
    "statement": [r"/newsevents/pressreleases/monetary(\d{8})[a-z]\.htm",
                  r"/newsevents/press/monetary/(\d{8})[a-z]\.htm",
                  r"/boarddocs/press/(?:monetary|general)/\d{4}/(\d{8})/"],
    "minutes": [r"/fomc/MINUTES/\d{4}/(\d{8})min\.htm", r"/fomc/minutes/(\d{8})\.htm",
                r"/monetarypolicy/fomcminutes(\d{8})\.htm",
                r"/monetarypolicy/files/fomcminutes(\d{8})\.pdf"],
    "presconf": [r"/monetarypolicy/fomcpresconf(\d{8})\.htm"],
    "beige": [r"/monetarypolicy/files/fomc\d{8}beige(\d{8})\.pdf",
              r"/fomc/beigebook/\d{4}/(\d{8})/default\.htm",
              r"/monetarypolicy/beigebook(\d{6})\.htm"],
}
RELEASED = re.compile(r"Released\s+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})")


BLOCK_OPEN = re.compile(r"<(?:p|div)[\s>]", re.I)
BLOCK_CLOSE = re.compile(r"</(?:p|div)>", re.I)


def _release_after(html: str, start: int, end: int, date: pd.Timestamp) -> pd.Timestamp:
    """Release date of the minutes link at html[start:end], read only from the enclosing
    <p>/<div> block (the date sits before, after or beside the links depending on the page
    era). Never borrows a date from a neighbouring block."""
    opens = [m.start() for m in BLOCK_OPEN.finditer(html, 0, start)]
    close = BLOCK_CLOSE.search(html, end)
    if not opens or not close or BLOCK_CLOSE.search(html, opens[-1], start):
        return pd.NaT  # no block, or the nearest opening tag was already closed before the link
    found = RELEASED.search(html[opens[-1]:close.end()])
    if not found:
        return pd.NaT
    release = pd.Timestamp(pd.to_datetime(found.group(1).replace(".", "")))
    if not (date <= release <= date + pd.Timedelta(days=120)):
        return pd.NaT
    return release


def parse_calendar_html(html: str) -> pd.DataFrame:
    rows = []
    for kind, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, html, re.I):
                path, raw = match.group(0), match.group(1)
                url = BASE + path + ("default.htm" if path.endswith("/") else "")
                date = pd.to_datetime(raw, format="%Y%m%d" if len(raw) == 8 else "%Y%m")
                if kind == "minutes":
                    release = _release_after(html, match.start(), match.end(), date)
                elif kind == "beige" and len(raw) == 6:
                    release = pd.NaT
                else:
                    release = date
                rows.append({"kind": kind, "date": date, "release": release, "url": url})
    frame = pd.DataFrame(rows, columns=["kind", "date", "release", "url"])
    return frame.drop_duplicates(subset=["kind", "date"]).reset_index(drop=True)


def load_calendar(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "calendar.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frames = [parse_calendar_html(get(HISTORICAL.format(year))) for year in range(1994, 2021)]
    frames.append(parse_calendar_html(get(CURRENT, cache=False)))
    cal = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset=["kind", "date"])
           .sort_values(["kind", "date"]).reset_index(drop=True))
    cal.to_pickle(path)
    return cal
