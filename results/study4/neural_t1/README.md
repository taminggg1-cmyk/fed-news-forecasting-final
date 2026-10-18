# AMENDED neural study: Study 4 **version 2**, T1, 308 months on the corrected cutoff (panel t1b), base S

**Status: dated amendment, not a pre-registration.** Protocol: design section 12, commit 6048e0d
(2026-10-13 00:51), written after the version 1 results (74ab18f) had been seen. Results: commit 1a31c73.
Version 1 must not be cited (it is in fed-news-forecasting, `results/study4/superseded_v1/`). The original
Plan 2 networks are in fed-news-forecasting `results/h2a/`; their untuned forecasts used in the "settings"
rows are copied to `results/inputs_from_earlier/paper_settings/`.

Code: `src/fed_news/studies/networks/run_networks.py` (`python -m fed_news.studies.networks.run_networks forecast`;
called `tune.py` in the original repository).

**Which file is which.** The data-vintage treatment is in the feature-set name, in every file name and in the
`features` column of the metrics and comparison tables:

| Feature set in the file name | Vintage | Role |
| --- | --- | --- |
| `S` | mixed (revised fallback in 4 statistics before 2011-12/2012-05) | base |
| `S+H`, `S+C` | mixed | **primary** (Holm over 6: 3 networks x 2 readers) |
| `strict`, `strict+H`, `strict+C` | **strict real-time** (only the 8 fully vintaged statistics) | robustness, outside the Holm family |
| `placebo-H`, `placebo-C` | mixed; text circularly shifted by half the panel | control |

H = Claude Haiku 5.5 stance, C = ChronoBERT stance. Files: `preds_<arch>_<features>_seed<k>.csv` (one seed),
`preds_<arch>_<features>.csv` (mean of 8 seeds, the reported forecast), `settings_<arch>_<features>.json`
(tuned hyperparameters at each refit), `metrics_neural_<task>.csv`, `comparison_neural_<task>.csv` (Newey-West
p, `nw_p_holm` only for the primary rows).
