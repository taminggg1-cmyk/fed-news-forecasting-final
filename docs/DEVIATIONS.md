# Deviations and notes: the final study

Copied unchanged from `DEVIATIONS.md` in fed-news-forecasting (ad2183c); only the Study 4 entries. Paths such as
`results/study4/` are the same in this repository.

### Study 4 notes (2026-10-12)

- Amendment 7.1: the Haiku token projection (23,980,280) exceeded the 20M limit; nothing was sent until the user
  allowed the full corpus; the limit became 25M. Actual use 23,395,731 tokens.
- Design section 8 (sensitivity checks) was added after the primary result, at an external reviewer's request,
  and recorded before those runs: real-time-data reruns, a news-column ablation, class counts, HAC and
  block-bootstrap versions of the primary test, two-sided McNemar. All exploratory; RESULTS_STUDY4.md.
- The July and September 2026 meetings are a retrospective post-cutoff evaluation, not live forecasts
  (Haiku 5.5 was released on 2026-10-07).
- Archived copies of the daily news-index files could not be retrieved (archive.org offline on 2026-10-12);
  their revision behaviour is unverified.
- Found 2026-10-12 while writing the methods note: the Plan 2 "paper LSTM" (240 units, 1 layer, 300 epochs;
  PREREG_H2A.md) does not match the LSTM settings in the AMRO WP/25-10 version read on 2026-10-12 (109 units,
  two layers, no dropout, learning rate 1e-4, 644 epochs). The GRU (188 units, 1 layer, 2e-4, 449 epochs)
  matches. The source of the 240/300 setting is not recorded; the Plan 2 LSTM rows should not be described as
  the paper's LSTM settings until this is resolved.
- Resolved 2026-10-17: the 240-unit, one-layer, 300-epoch LSTM and the published RMSE 0.2376 / 0.2705 come from
  the July 2025 preliminary version of the paper (SSRN 5401236, `ssrn-5401236.pdf`, section 3.2.2 and Table 2).
  The September 2025 AMRO WP/25-10 version changes the LSTM to 109 units, two layers, 644 epochs, and its Table 2
  reports RMSE 0.1868 (LSTM) and 0.2672 (GRU), MAE 0.1164 and 0.1476. The Plan 2 LSTM therefore follows the
  preliminary version, not the working paper cited in the references. The README and the draft paper now quote
  both versions. With the working-paper numbers, the no-change forecast (0.192) no longer beats the paper's LSTM.
  No result was rerun.
- Study 4 v2 compute rule (design 12.8), applied 2026-10-13 before any v2 tuning: timed upper bounds were
  about 14.3 h (CPU: LSTM and GRU, both tasks) and 25.6 h (GPU: TFT, both tasks), above 16 h, so every v2 arm
  uses 20 Optuna trials instead of 30 (results/study4/v2_timing/).

### Exploratory pooled news test (recorded 2026-10-13, about 14:00, before running it)

Added after seeing the v2 LSTM and GRU results (the user noted that the LSTM's news versions mostly point the
right way), so it is exploratory, chosen post hoc, and outside the Holm family of the primary comparisons.
- For each network (LSTM, GRU) and task, the per-row loss differential is the squared error of the no-news
  model minus that of the news model, in rate units, for the four news-vs-no-news pairs: normal + ChronoBERT,
  normal + Haiku (against the normal no-news model), strict + ChronoBERT, strict + Haiku (against the strict
  no-news model). The pooled differential is the row-wise mean of the four; positive favours news.
- Test: one-sided Newey-West test of its mean (lags = the refit interval: 8 meetings, 12 months) and a
  moving-block bootstrap 90% interval (same block, 10,000 draws, seed 0). Also by reader (the two ChronoBERT
  pairs, the two Haiku pairs) and the count of pairs whose mean differential is positive.
- Subsets: scheduled meetings (main) and all meetings for the meeting task; all test months (main) and the
  2016-2025 paper window for the monthly task; plus test rows before 2016 vs from 2016 for both tasks.
- TFT is excluded until its forecasts finish. Results: results/study4/exploratory_pooled_news.json.

### Post-audit correction (2026-10-13), Study 4 version 2

- tune.forecast_all now takes the Holm family from the registered specification (tune.registered_holm). It
  reports no adjustment while the family is incomplete. The four-test adjustment printed in
  logs/v2_fc_t2_rnn.log (LSTM + Haiku 0.0135) is provisional and superseded; the committed six-test values are
  unchanged (0.020).

### Changes made when this repository was copied (2026-10-18)

This repository was created on 2026-10-18 from fed-news-forecasting `ad2183c`, after every result in it had been
produced. Code was copied unchanged except for the following.

- Helpers the final study borrows from earlier designs were copied, text unchanged, into shared modules, and the
  imports that use them were rewritten: `features/haiku_prompt.py` (proofs of concept), `features/realtime.py`
  (Study 5), `features/stance_panel.py` (Plan 5), `models/neural.standardise` (Plan 2) and
  `evaluation/stats_tests.nw_summary` (Study 5). Their tests are in `tests/test_borrowed_helpers.py`.
- Two input files from earlier designs moved to `results/inputs_from_earlier/` (the Haiku memory exam and Plan 2's
  untuned paper-settings forecasts); `haiku_compare.py` and `tune.py` read them from there.
- The news guard in `models/walkforward.run_spec` refuses news columns until the git tag `prereg` exists. That tag
  is Plan 1's pre-registration in the original repository (2026-10-09). Rather than creating a tag of that name
  here, `config.PREREG_ELSEWHERE` records where it is, and `prereg_ok` accepts it.
- The version 1 networks (`results/study4/superseded_v1/`, superseded and not to be cited) and every earlier design
  were left out; they remain in the original repository.
