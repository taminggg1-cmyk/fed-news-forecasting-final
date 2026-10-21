"""Every setting in one place.

WHERE THE DATA IS (DATA below): the folder in the environment variable FED_NEWS_DATA if set; else <repo>/data,
which `python scripts/download_data.py` fills from this repository's release; else D:/fed_news (the author's
machine). docs/guides/DATA.md lists every file. Results go to results/.

Details: Paths and every fixed setting of the design (spec sections 4, 6, 7, 8).
"""
import os
from pathlib import Path

import numpy as np

if os.name == "nt":  # the author's Windows machine keeps the Hugging Face cache on D:
    os.environ.setdefault("HF_HOME", "D:/hf_cache")

REPO = Path(__file__).resolve().parents[2]
# The data folder: FED_NEWS_DATA if set; else <repo>/data if it exists (scripts/download_data.py creates it);
# else D:/fed_news (the author's machine).
DATA = Path(os.environ.get("FED_NEWS_DATA") or (REPO / "data" if (REPO / "data").is_dir() else "D:/fed_news"))
CACHE = DATA / "cache"
SECRETS = DATA / "secrets"
RESULTS = REPO / "results"

SAMPLE_START = "1994-01-01"
TEST_START = "2001-01-01"
PAPER_T1 = ("2016-01-01", "2025-04-01")
PAPER_T2 = ("2016-01-01", "2025-04-30")

FIRST_VINTAGE, LAST_VINTAGE = 1999, 2024
FIRST_CLASSIFIER_YEAR = 2000
STANCE_SEEDS = (0, 1, 2)
STANCE_LR, STANCE_EPOCHS, STANCE_BATCH, STANCE_MAXLEN = 2e-5, 3, 16, 256
STANCE_HOLDOUT = 0.2
MIN_LABELS = 50

TEXT_WINDOW_DAYS = 30
INDEX_LAG_DAYS = 3
MIN_DOC_CHARS = 500

ORDLOGIT_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = tuple(float(a) for a in np.logspace(-1, 3, 7))
CV_FOLDS = 5
CLASS_BAND = 0.125
CLASS_EPS = 0.02
ALPHA = 0.05
PREREG_TAG = "prereg"
# The Plan 1 pre-registration that run_spec's news guard checks for is tag `prereg` (2026-10-09) in the original
# repository; this repository was copied from it on 2026-10-18, after every result here was produced.
PREREG_ELSEWHERE = "github.com/taminggg1-cmyk/fed-news-forecasting, tag prereg"



def report_path(name: str) -> Path:
    """Where a generated report (RESULTS*.md) is written: docs/reports/ (the repository root until 2026-10-18)."""
    path = REPO / "docs" / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs() -> None:
    """Create the data, cache, secrets and results folders if they are missing."""
    for path in (DATA, CACHE, SECRETS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
