"""DOWNLOAD EVERYTHING, in order. This is where the downloading starts.

Run:     python -m fed_news.download              (downloads only the files that are missing)
         python -m fed_news.download --refresh    (downloads everything again)
Needs:   a free FRED API key in <data>/secrets/fred_api_key.txt, for the ALFRED step only.
Writes:  the files in the STEPS table below, in the data folder (config.DATA).

Each step calls one module in this folder. That module downloads its source with http.get, the only function in the
project that opens a network connection, and saves the result in the data folder. A step whose file is already
there just loads it, without downloading. The order matters: the meeting list needs the calendar and the daily target
rate, the documents need the calendar, and the labels are dated with the documents.

Next: build the tone scores and the training rows (docs/CODE_GUIDE.md, stage 2).
"""
import argparse
import time

from fed_news import config
from fed_news.download import (alfred_vintages, fed_documents, fomc_calendar, fomc_meetings, fred_market, news_indices,
                               stance_labels)

# (what, downloaded from, saved as in the data folder, function that downloads it when missing)
STEPS = [
    ("FOMC calendar", "federalreserve.gov calendar pages", "calendar.pkl", fomc_calendar.load_calendar),
    ("Daily market data", "fred.stlouisfed.org", "market_daily.pkl", fred_market.load_daily),
    ("Monthly fed funds rate", "fred.stlouisfed.org", "fedfunds.pkl", fred_market.load_fedfunds),
    ("Economic statistics, every vintage", "api.stlouisfed.org (ALFRED)", "alfred/", alfred_vintages.load_all),
    ("News indices", "frbsf.org, matteoiacoviello.com, policyuncertainty.com", "indices.pkl",
     news_indices.load_indices),
    ("FOMC decisions", "built from calendar.pkl and market_daily.pkl", "meetings.pkl", fomc_meetings.load_meetings),
    ("Fed documents (full text)", "federalreserve.gov", "corpus.pkl", fed_documents.build_corpus),
    ("Human stance labels", "huggingface.co (gtfintechlab)", "labels.pkl", stance_labels.load_labels),
]


def saved(target: str) -> bool:
    """Whether a step's output is already in the data folder."""
    path = config.DATA / target
    return any(path.glob("*.pkl")) if target.endswith("/") else path.exists()


def run(refresh: bool = False, steps=STEPS) -> None:
    """Run every step in order, printing where each file comes from and where it is saved."""
    print(f"data folder: {config.DATA}", flush=True)
    for k, (what, source, target, load) in enumerate(steps, 1):
        action = "downloading" if refresh or not saved(target) else "already saved, loading"
        print(f"[{k}/{len(steps)}] {what}: {source} -> {config.DATA / target} ({action})", flush=True)
        t0 = time.time()
        out = load(refresh)
        print(f"      {len(out)} {'series' if isinstance(out, dict) else 'rows'} ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None) -> None:
    """Command line: optional --refresh."""
    ap = argparse.ArgumentParser(description="Download every data source, in order, into the data folder.")
    ap.add_argument("--refresh", action="store_true", help="download again even when the file is already saved")
    run(ap.parse_args(argv).refresh)


if __name__ == "__main__":
    main()
