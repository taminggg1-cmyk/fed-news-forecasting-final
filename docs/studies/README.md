# Study documents

One folder per study, in the order the studies were run. Each design was committed and pushed before its results
were computed; the commit history shows the dates. Every attempt and its primary result are listed in
[`results/ATTEMPTS.md`](../../results/ATTEMPTS.md).

| Folder | Study | Attempts | Documents | Code | Results |
| --- | --- | --- | --- | --- | --- |
| [`1_study4/`](1_study4/) | Study 4 and version 2: ordered logit and tuned LSTM / GRU / TFT, with ChronoBERT (C) or Haiku (H) tone | 6-8 | design (section 12 is the version 2 amendment), results report | `src/fed_news/studies/simple_model/`, `src/fed_news/studies/networks/` | `results/study4/` |
| [`2_coral/`](2_coral/) | CORAL study: ordinal tone readers (ChronoBERT-CORAL "Co", Haiku ordinal "Ho") and CORAL networks; amendment 3 adds the meetings from 2010 | 9-11 | design, implementation plan | `src/fed_news/studies/coral/` | `results/coral/` |
| [`3_sigtests_distill/`](3_sigtests_distill/) | Part A: tests used in the literature, run on the saved forecasts (exploratory). Part B: Haiku-distilled ChronoBERT ("Hd") | 12, 12b | design (Parts A and B) | `src/fed_news/studies/sigtests/`, `src/fed_news/studies/distill/` | `results/sigtests/`, `results/distill/` |
| [`4_text_vs_economy/`](4_text_vs_economy/) | Fed text against a Taylor-rule baseline without market prices | 13 | design | not built | not run |
| [`5_doctype/`](5_doctype/) | The distilled tone split by document type (statements and press conferences, minutes, speeches) | 14, 14b | design | `src/fed_news/studies/doctype/` | `results/doctype/` |

## Paths inside the documents

The documents are kept exactly as they were committed, so they can be checked against their dates. The repository
was reorganised on 2026-10-21, after these documents were written, and some paths inside them refer to the old
layout:

| Old path | Now |
| --- | --- |
| `src/fed_news/coral/`, `distill/`, `sigtests/` (modules `fed_news.coral`, ...) | `src/fed_news/studies/coral/`, `studies/distill/`, `studies/sigtests/` (modules `fed_news.studies.coral`, ...) |
| `tests/coral/`, `tests/distill/`, `tests/sigtests/` | `tests/studies/coral/`, ... |
| other `tests/test_*.py` | `tests/shared/` (shared code) or `tests/studies/simple_model/`, `tests/studies/networks/` |
| `docs/design/<file>` | `docs/studies/<study folder>/<file>` |
| `docs/CODE_GUIDE.md`, `DATA.md`, `DATA_MANIFEST.txt`, `DEVIATIONS.md`, `PIPELINE_TECHNICAL.md` | `docs/guides/` |
| `docs/RESULTS_STUDY4.md` | `docs/studies/1_study4/RESULTS_STUDY4.md` |
| `audit/` | `scripts/audit/` |
