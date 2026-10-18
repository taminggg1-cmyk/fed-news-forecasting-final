# Data for the final study

The data files the final study reads are attached to this repository's release
[`data-2026-10-18`](https://github.com/taminggg1-cmyk/fed-news-forecasting-final/releases/tag/data-2026-10-18) as
`final_study_data.zip`. The easy way: `uv run python scripts/download_data.py` downloads it, unpacks it into
`data/` (where the code looks by itself) and checks every file. By hand: unzip it and point `FED_NEWS_DATA` at
the `fed_news_data` folder. `MANIFEST.txt` (also [`docs/DATA_MANIFEST.txt`](DATA_MANIFEST.txt)) gives the sha256 and size of
every file. These are the exact files that produced `results/study4/`: rerunning the simple model, the network
evaluations, the pooled test and the diagnostics on them reproduced every saved output byte for byte (2026-10-18).

API keys are not included. The full project's data, including the earlier designs and the raw download cache, is in
the original repository's release
[`data-2026-10-17`](https://github.com/taminggg1-cmyk/fed-news-forecasting/releases/tag/data-2026-10-17).

## Training rows

| File | Rows | What |
| --- | --- | --- |
| `t2_full.pkl` | 270 meetings, Feb 1994 – Sep 2026 | Per-meeting panel: cutoff (the day before the meeting), the 16 statistics (`s_*`, as first published), the 8 market signals, news columns and the targets (new target rate, change, cut / hold / hike). Test meetings start in 2001 |
| `t2.pkl` | 270 | The same meetings' cutoffs and text windows, used to attach a tone series to the panel |
| `t1b_full.pkl` | 393 months, Jan 1994 – Sep 2026 | Monthly panel on the corrected cutoff (start of the 2nd US business day); target = average effective fed funds rate (FRED `FEDFUNDS`). Test months start in 2001 (308) |
| `t1b.pkl` | 393 | The same months' cutoffs and text windows |

## Fed tone

| File | Rows | What |
| --- | --- | --- |
| `doc_stance.pkl` | 2,578 documents | ChronoBERT stance per Fed document (mean over its sentences), each scored by the classifier trained only on labels from before that document's year |
| `stance/<year>.pkl` | one per year, 2000–2026 | ChronoBERT sentence scores for that year, the classifier's held-out accuracy and its training settings |
| `doc_stance_haiku.pkl` | 2,578 documents | Claude Haiku 5.5 stance per document, from its sentence labels |
| `study4/sentence_labels.pkl` | 343,742 sentences | Haiku's label for each sentence (hawkish / dovish / neutral; 343,717 labelled), with the masked text it saw |
| `labels.pkl` | 3,074 sentences | Human stance labels used to train ChronoBERT (Trillion Dollar Words and World Central Banks datasets), dated by their source document |
| `corpus.pkl` | 2,840 documents | Fed documents with release date and full text: 1,981 speeches, 263 minutes, 262 Beige Books, 240 statements, 94 press conferences |

## Economic and market inputs

| File | What |
| --- | --- |
| `alfred/<series>.pkl` | ALFRED real-time vintages for 12 series: CPI, core CPI, PCE and core PCE prices, PPI, GDP, M2, payrolls, average hourly earnings, unemployment, U-6, house prices |
| `market_daily.pkl` | Daily market series from FRED, 1990 – Oct 2026: target rate and range, effective rate, Treasury bill and yields (3 months to 10 years), VIX, oil |
| `fedfunds.pkl` | Monthly average effective fed funds rate |
| `indices.pkl` | Daily news indices: SF Fed news sentiment, geopolitical risk (GPR), economic policy uncertainty (EPU). Current vintages; their revision history is unverified |
| `meetings.pkl` | FOMC meetings 1994–2026: target before and after, change, cut / hold / hike, unscheduled flag |
| `calendar.pkl` | FOMC calendar with document release dates and URLs |
