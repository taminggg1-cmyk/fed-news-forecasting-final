# Results

[`ATTEMPTS.md`](ATTEMPTS.md) lists every attempt so far, the test each one committed to in advance, and what came
out. Read it before reading any single p-value.

| Folder or file | Study | What |
| --- | --- | --- |
| `ATTEMPTS.md` | all | Every attempt, its primary test and its verdict |
| `study4/` | Study 4 and version 2 (attempts 6-8) | The simple model (ordered logit, this folder), the tuned networks (`neural/` per meeting, `neural_t1/` monthly), the sensitivity checks and the pooled news test. Its README lists every file and its status |
| `coral/` | CORAL study (attempts 9-11) | `logit/` and `networks/` (2001-2026), `logit_from2010/` and `networks_from2010/` (amendment 3), the ChronoBERT-CORAL accuracy, the tuned network settings (`networks_hyperparameters.csv`), and `EXPLAINER_TH.md`, which explains the study in Thai |
| `sigtests/` | Literature tests (exploratory) | `SUMMARY.md` first; pairwise tests, in-sample tests and joint tests on the saved forecasts |
| `distill/` | Haiku-distilled ChronoBERT (attempts 12, 12b) | `settings.json` (tuning), `fidelity.json` (agreement with Haiku), `logit/` (primary test), `networks/` (declared secondary test) |
| `inputs_from_earlier/` | Study 4 | Two input files copied from earlier designs (Haiku memory exam, untuned paper-settings forecasts) |
| `stance_accuracy.csv` | Study 4 | Held-out accuracy of the per-year ChronoBERT stance classifiers |
| `coverage.csv`, `dropped_docs.json` | all | Fed document corpus coverage by year and type, and documents dropped while building it |
| `pip_freeze.txt` | all | Exact Python package versions |
