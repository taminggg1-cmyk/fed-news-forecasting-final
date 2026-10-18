# Fed News Forecasting: the final study

**Does what the Federal Reserve says help forecast what it does?** This study tests whether the hawkish or dovish
tone of Fed communication improves out-of-sample forecasts of FOMC decisions and of the federal funds rate, beyond
the economic statistics and market interest rates that were available at the time.

This repository holds **only the final study**: Study 4 (the simple model) and Study 4 version 2 (the tuned neural
networks). The full project record is in
[fed-news-forecasting](https://github.com/taminggg1-cmyk/fed-news-forecasting): the earlier trial-and-error designs
(Plan 1, Plan 2, Plan 5, Study 2, Study 3, Study 5 and the proofs of concept), their pre-registrations, the
2026-10-13 audit and the draft paper. Commit hashes on this page (b4a00fd, 7879a70, 6048e0d, 1a31c73) refer to that
repository, where the date of each design can be checked.

*Draft research, October 2026. The code, analyses and documents were written with an AI coding agent
(see [How this was made](#how-this-was-made)).*

## Quick start (Linux, macOS or Windows)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # install uv (Windows: see docs.astral.sh/uv)
git clone https://github.com/taminggg1-cmyk/fed-news-forecasting-final.git
cd fed-news-forecasting-final
uv sync                                                # Python 3.13 and the exact package versions (uv.lock)
uv run python scripts/download_data.py                 # the training data -> data/ (79 MB, checksums checked)
uv run pytest -q                                       # offline tests, about 1 minute
uv run python -m fed_news.studies.simple_model.run_simple_model          # Study 4, about 5 minutes
uv run python -m fed_news.studies.networks.run_networks forecast --task T2 --device cpu   # Study 4 v2 tables
```

**Where is the data?** In `data/` after the download step (the code finds it there by itself; you can also set
`FED_NEWS_DATA` to any folder). [`docs/DATA.md`](docs/DATA.md) describes every file, and
[`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) shows where each one is downloaded from, which file builds it and which
file reads it next. **Start reading the code with `docs/CODE_GUIDE.md`.** To download the raw data yourself instead
of using the zip: `uv run python -m fed_news.download` (every source, in order; needs a free FRED API key).

## Short answer

News improves most models: Fed tone lowers the forecast error in 12 of 14 main comparisons (10 of 12 for the
networks, and both versions of the simple model). Only one gain is statistically significant, LSTM + Claude Haiku on
meetings, and it is not confirmed on strict real-time data. Not significant does not mean news fails: with about
200 meetings, gains this small cannot yet be told apart from chance.

## The two parts

| Part | Code | Models | Tasks | Status |
| --- | --- | --- | --- | --- |
| Study 4 | `src/fed_news/studies/simple_model/` | Simple model: penalised ordered logit | Per meeting | Exploratory study; design b4a00fd committed before the first run, results 7879a70 |
| Study 4 version 2 | `src/fed_news/studies/networks/` | LSTM, GRU, TFT, tuned with Optuna inside each training window, 8 seeds | Per meeting and monthly | Dated amendment 6048e0d, written after the version 1 network results had been seen; results 1a31c73. Not a pre-registration |

**Shared design.** FOMC meetings and months, 2001–2026. Every forecast uses only information available before it:
16 economic statistics as first published (ALFRED vintages), plus 8 market signals for meetings. Fed tone is read
by ChronoBERT (retrained each year on earlier text only, so it cannot see the future) or by Claude Haiku 5.5, which
labelled 343,717 of 343,742 Fed sentences (its training data run to June 2026). Per meeting: 206 scheduled meetings, predicting
cut / hold / hike and the new target rate. Monthly: 308 months, predicting the average effective fed funds rate.
The design is in [`docs/design/`](docs/design/2026-10-12-study4-haiku-index-tuned-design.md) (section 12 is the
version 2 amendment).

## Results

**Per meeting (206 scheduled meetings, 2001–2026): decisions called correctly, with p for the news versions**

| Model | No news | + ChronoBERT | + Haiku |
| --- | --- | --- | --- |
| Simple model (ordered logit) | 86.4% | 86.9%, p 0.21 | 88.8%, p 0.14 |
| LSTM, normal data | 78.6% | 83.5%, p 0.37 | **84.5%, p 0.003 (Holm p 0.020)** |
| GRU, normal data | 80.6% | 80.1%, p 0.99 | 82.0%, p 0.20 |
| TFT, normal data | 86.4% | 85.9%, p 0.21 | 86.9%, p 0.17 |
| LSTM, strict data | 81.6% | 84.5%, p 0.47 | 85.4%, p 0.46 |
| GRU, strict data | 83.0% | 85.9%, p 0.85 | 86.4%, p 0.93 |
| TFT, strict data | 87.4% | 86.9%, p 0.94 | 87.9%, p 0.69 |

Only one gain is significant: LSTM + Haiku on normal data (RMSE 0.1420 → 0.1333 percentage points). The strict-data
accuracy gains are within chance: a 3-point gain is about 6 meetings, and the LSTM + Haiku RMSE barely moved
(0.1251 → 0.1246). For the simple model, Haiku lowers log loss by 7.4% and ChronoBERT by 4.2%.

**Monthly (308 months, 2001–2026): RMSE in percentage points, with p for the news versions**

| Model | No news | + ChronoBERT | + Haiku |
| --- | --- | --- | --- |
| "Rate stays the same" | 0.181 | | |
| LSTM, normal data | 0.188 | 0.198, p 0.88 | 0.172, p 0.026 (Holm p 0.156) |
| GRU, normal data | 0.180 | 0.178, p 0.36 | 0.178, p 0.45 |
| TFT, normal data | 0.191 | 0.177, p 0.042 (Holm p 0.169) | 0.173, p 0.034 (Holm p 0.168) |
| LSTM, strict data | 0.197 | 0.185, p 0.13 | 0.174, p 0.12 |
| GRU, strict data | 0.184 | 0.189, p 0.67 | 0.165, p 0.014 (secondary check) |
| TFT, strict data | 0.186 | 0.181, p 0.31 | 0.173, p 0.084 |

The final study has no monthly simple model with tone. The monthly simple model of the original project (Plan 1's
ridge) used a different news input and is in the original repository.

**How to read the p-values.** Each is one-sided: does news lower the error? Network p-values are Newey–West tests on
squared error. Simple-model p-values are HLN Diebold–Mariano tests on log loss, over all 212 meetings. Holm p
corrects over the 6 network comparisons per task (LSTM, GRU and TFT × ChronoBERT and Haiku) on normal data. A
four-test value (0.0135) printed in an early run log was provisional and is superseded
([`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)). Strict data = only the 8 fully vintaged statistics; it is a robustness
check outside the Holm family. Sources: `results/study4/neural/metrics_neural_T2.csv`, `comparison_neural_T2.csv`,
`results/study4/neural_t1/metrics_neural_T1.csv`, `comparison_neural_T1.csv`, `results/study4/verdict_T2.json`.

## Checks on the news signal

- **Placebo:** the tone series shifted by half the sample (about 16 years). It never beats the base model
  significantly (smallest p 0.115). LSTM + Haiku beats its own placebo (p 0.0008).
- **But fake tone sometimes helps too:** it lowered meeting RMSE for the GRU (0.1445 → 0.1345) and the TFT
  (0.1415 → 0.1355), so network gains of this size can come from noise.
- **Shuffled labels** (training decisions shuffled): no gain for the simple model (p 1.00).
- **Haiku memory:** Haiku recalls 87.8% of decisions from the date alone, and in an earlier leak test its labels
  tracked the next decision more than ChronoBERT's (p 0.00005). In Study 4, Haiku's gain over ChronoBERT is not
  larger on meetings it recalled (p 0.378). Only 2 test meetings (29 Jul and 16 Sep 2026) fall after its training
  cutoff.

## Why the simple model beats the bigger networks

On normal data the ordered logit (RMSE 0.127 percentage points) has lower RMSE than every network (0.133–0.161), and
calls at least as many decisions right as the best network in each column. Descriptive diagnostics, not registered
tests ([`audit/why_simple_model_wins.txt`](audit/why_simple_model_wins.txt)):

- Each refit trains on 48 to 246 meetings. The tuned LSTM and GRU have a median of about 211,000 and 217,000
  parameters; the logit has 26.
- The networks call moves that never happen. On the 140 holds, the logit called a move 12 times, against 28 (LSTM),
  25 (GRU) and 18 (TFT). Holds give 95% of the LSTM's extra squared error.
- A few unusual meetings decide it. In early 2009, with the rate at 0.125%, the LSTM and GRU forecast −0.39% and
  −0.48%. In March 2023 they forecast cuts while the Fed hiked.
- Tuned settings jump between refits (LSTM hidden size 256 → 16 → 128), and single-seed RMSE ranges 0.140–0.176
  across the networks.
- Market yields already price the expected move, and the ordered logit turns them straight into cut / hold / hike
  probabilities.
- With strict data (16 inputs instead of 24), the GRU and LSTM reach lower RMSE (0.122, 0.125) but call fewer
  decisions right (83.0%, 81.6%). No network beats the simple model on both measures.

**Tuning matters more than text.** An earlier design used the published paper's network settings without tuning.
Tuning inside each training window lowered monthly squared error against those settings by 0.158 squared percentage
points for the LSTM (Newey–West p 0.000007) and 0.224 for the GRU (p 0.012); the TFT gain was 0.0006 (p 0.43)
(`results/study4/neural_t1/comparison_neural_T1.csv`, "settings" rows).

## What this study borrows from the earlier work

The final study reuses a few pieces built earlier. They are copied here unchanged, as helpers or input files, and
each says where it came from.

| Piece | Here | Came from (original repository) | Used for |
| --- | --- | --- | --- |
| Haiku prompt, reply parser, sentence masking | `features/haiku_prompt.py` | the Haiku labelling pilot and leak test (proofs of concept) | building the Haiku index |
| Strictly real-time column list | `features/strict_columns.py` | Study 5 | the "strict data" rows |
| Tone-column swap for a panel | `features/swap_tone.py` | Plan 5 | Haiku and ChronoBERT panels |
| `standardise`, `nw_summary` | `models/neural.py`, `evaluation/significance.py` | Plan 2, Study 5 | network inputs, the pooled test |
| Haiku memory exam | `results/inputs_from_earlier/memory_exam.csv` | proofs of concept | the memory flag |
| Untuned paper-settings forecasts | `results/inputs_from_earlier/paper_settings/` | Plan 2 | the "tuning matters" rows |

## Repository layout

```
src/fed_news/
  config.py     settings, and where the data folder is
  download/     everything fetched from the internet (ALFRED, FRED, Fed documents, FOMC calendar, labels, indices)
  features/     Fed tone (ChronoBERT, Haiku) and the training rows (per meeting, per month)
  models/       ordered logit, ridge, the walk-forward loop, LSTM / GRU / TFT
  evaluation/   accuracy numbers and significance tests (DM-HLN, Newey-West, McNemar, Holm)
  studies/      simple_model/ (Study 4) and networks/ (Study 4 version 2): the runs that write results/
scripts/        download_data.py: fetch and check the training data
tests/          offline unit tests
results/study4/ saved forecasts, metrics and test tables (each folder has a README)
results/inputs_from_earlier/   the two input files listed above
audit/          why the simple model beats the networks (script and output)
docs/           code guide, data description, design (with the version 2 amendment), Study 4 report, deviations
```

## Reproduce

- **Environment.** `uv sync` installs Python 3.13 and the exact package versions in `uv.lock` (the ones that produced
  the results; full list in [`results/pip_freeze.txt`](results/pip_freeze.txt)). On Linux, PyPI's torch includes CUDA.
- **Data.** `uv run python scripts/download_data.py` downloads `final_study_data.zip` (79 MB zipped, 279 MB unzipped,
  53 files) from the release
  [`data-2026-10-18`](https://github.com/taminggg1-cmyk/fed-news-forecasting-final/releases/tag/data-2026-10-18),
  unpacks it into `data/` and checks every file against [`docs/DATA_MANIFEST.txt`](docs/DATA_MANIFEST.txt). It holds
  the feature panels, ChronoBERT and Haiku tone scores, Haiku's sentence labels, the Fed documents, human labels,
  ALFRED vintages and market series ([`docs/DATA.md`](docs/DATA.md)). API keys are not included.
- **Runs** (`uv run python -m ...`; [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) lists every step):
  - `fed_news.studies.simple_model.run_simple_model` reruns the simple model (CPU, about 5 minutes).
  - `fed_news.studies.networks.run_networks forecast --task T2 --device cpu` (and `--task T1`) rebuilds the network
    ensembles, metrics and tests from the saved per-seed forecasts. Deleting those forecasts makes it retrain the
    networks (hours; a GPU helps for the TFT). `run_networks tune --trials 20` reruns the Optuna search.
  - `fed_news.studies.simple_model.sensitivity_checks` and `fed_news.studies.networks.pooled_news_test` rerun the
    exploratory checks; `uv run python audit/why_simple_model_wins.py` reruns the diagnostics.
- **Checked on 2026-10-18** in a fresh copy set up with `uv sync` and `scripts/download_data.py`: rerunning the simple
  model, both network evaluations, the pooled test and the diagnostics reproduced every file in `results/study4/`
  and `audit/` byte for byte.
- Building the tone scores again: `fed_news.features.chronobert_tone` (CUDA GPU) and
  `fed_news.features.haiku_tone project|submit|collect` (Claude API, about 23.4 million tokens).

## How this was made

All code, analyses and documents were written by an AI coding agent (Claude, by Anthropic) working with the
project's author. The author's decisions appear where commit messages and design documents record them. Results
added after earlier results had been seen are labelled exploratory, and the deviations are listed in
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## License

MIT, see [`LICENSE`](LICENSE).
