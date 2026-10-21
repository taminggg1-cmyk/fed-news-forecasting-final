# Study 4: a Haiku news index and tuned neural forecasters (exploratory)

Date: 2026-10-12. Requested by the user after the proof-of-concept series (RESULTS_POC.md). Fixed and committed
before any API call, any tuning run and any Study 4 forecast. The 2001-2026 meetings are not untouched data:
every earlier study used them, so all Study 4 results are labelled exploratory.

## 1. Question

Does a news index built by Claude Haiku 5.5 (sentence stance labels, replacing ChronoBERT) improve forecasts
of FOMC decisions over the same model without news? And do LSTM, GRU and TFT forecasters, tuned on our own
data instead of using settings taken from a reference paper, change that answer?

## 2. What the literature says about leakage, and what this study does

Pre-cutoff skill of a commercial LLM is not identified: a model that saw the outcomes in training cannot be
shown to forecast them without memory (Lopez-Lira, Tang and Zhu 2025). Masking names and dates is partial
(Glasserman and Lin 2023; Lopez-Lira et al. 2025 find models rebuild entities and dates); prompting the model to
ignore the future does not work (Lopez-Lira et al. 2025); clean evidence needs a leak-free model (He, Lv, Manela
and Wu 2025; Sarkar 2024) or data after the training cutoff (Ludwig, Mullainathan and Rambachan 2025; Crane,
Karra and Soto 2025). Gao, Jiang and Yan (2025) test whether forecast power is larger where the model recalls
the outcome. Engelberg, Manela, Mullins and Vulicevic (2025) bound the bias by neutering the text until a
separate model cannot identify it.

This study therefore uses every available layer and claims no more than they support:

1. **Masking** (`poc_haiku_leak.mask`: years, months, numbers, official names, named events). Its known limit
   is reported: Haiku dates 38.7% of masked sentences within 2 years (Test A) and 89.5% of masked statements
   within 1 year (memory exam).
2. **Random order across documents and years**, 25 sentences per request, so no request carries a whole
   document's context.
3. **No "ignore the future" instruction** (shown not to work).
4. **Leak-free reference**: the ChronoBERT stance index in the identical model on identical rows.
5. **Recall-interaction test** (adapted from Gao, Jiang and Yan 2025; the API gives no token probabilities, so
   recall is the memory exam's per-meeting result: Haiku named the decision from the date alone).
6. **Post-cutoff meetings**: Haiku 5.5's training data ends June 2026, so 2026-07-29 and 2026-09-16 are unseen.
   Two meetings carry no statistical weight; they are reported one by one. Live forecasts of future meetings
   get their own committed rules before the next meeting.
7. **Prior red flag reported**: Test B already found Haiku's sentence labels track the next decision beyond the
   human label and ChronoBERT (p = 0.00005). The paper states this next to any Study 4 result.

## 3. The Haiku news index

- Documents: every corpus document of types statement, minutes, press conference and speech (2,578 documents),
  split into sentences by `stance._sentence_table` (343,742 sentences), exactly the set ChronoBERT scored.
- Each sentence is masked (layer 1). Sentences go in a random order (seed 0) in requests of 25.
- Prompt and settings are the attempt-1 labeller, unchanged: `poc_llm_labels.SYSTEM`, `batch_prompt`, model
  `claude-haiku-5-5`, thinking disabled, effort low, max_tokens 1024. No prompt is tuned on any outcome.
- Sent through the Message Batches API. A request whose reply cannot be parsed, is refused or is cut off is
  sent once more in a second batch; sentences still unlabelled are left out of their document's mean, and
  their count is reported.
- **Token rule:** before submitting, the input tokens of 200 random requests are counted with the free
  token-counting endpoint and scaled to all requests; output is projected at 8 tokens per sentence. If the
  projection exceeds 20,000,000 tokens, nothing is submitted and the user is asked.
- Sentence score: hawkish +1, dovish -1, neutral 0. Document stance: mean over its labelled sentences (share
  hawkish minus share dovish, the same scale as ChronoBERT's probability difference). Saved as
  `doc_stance_haiku.pkl` in the `doc_stance.pkl` format.
- Meeting features: `naive_stance.naive_panel("t2", ...)`, the aggregation every earlier study used
  (`n_stance`, `n_stance_chg`; text window from the previous meeting's cutoff to the day before the decision).

## 4. Forecast comparison (primary: penalised ordered logit, T2)

Same engine as Plan 1 (`walkforward.run_spec`, penalty by expanding-window CV inside the training rows,
features standardised on the training rows), with a new option that saves the three class probabilities.
Rows: the meetings where every column below is present in both panels; test window 2001-01-01 to 2026-09-16.

| Name | Features |
|---|---|
| S+M | economic data and market prices, no news (baseline) |
| S+M+H | S+M plus the Haiku stance (`n_stance`, `n_stance_chg`) |
| S+M+C | S+M plus the ChronoBERT stance (leak-free reference) |
| placebo-H | S+M+H with the Haiku columns permuted across training rows (Plan 1 placebo) |
| shuffled-H | S+M+H with the training targets permuted (Plan 1 control) |

- Primary loss: per-meeting log loss of the true class. Primary test: one-sided Harvey-Leybourne-Newbold
  Diebold-Mariano, S+M+H against S+M. Secondary: Clark-West on the rate forecast, McNemar on accuracy, Brier.
- **Verdict** (`evaluate.decide` rules): placebo-H must not beat S+M at p < 0.05 and shuffled-H not at p < 0.01,
  else "void". Otherwise "supported" if the primary p < 0.05, else "not supported".
- **Memory flag**: for test meetings with a memory-exam row, regress the Haiku gain minus the ChronoBERT gain
  (gain = S+M log loss minus the news model's log loss) on a recalled indicator, HC3 standard errors. Flag if
  the coefficient is positive with one-sided p < 0.05: Haiku's advantage over the leak-free reader is larger
  where Haiku remembers the decision. The same regression on a dated indicator (year guessed within 1) is
  reported without a rule. Fewer than 30 meetings are not recalled, so power is low; a quiet flag is weak evidence.
- **Wording fixed now.** "Supported" without the flag: "the Haiku index improves forecasts; memory cannot be
  ruled out (pre-cutoff skill is not identified), but no memory pattern was detected". "Supported" with the
  flag: "the improvement is consistent with memory". The comparison of S+M+H with S+M+C (same tests) is
  reported without a rule.

## 5. Tuned neural forecasters (secondary)

The earlier neural settings were not tuned on our data (GRU 188 hidden units, learning rate 2e-4, 449 epochs
and 12 steps are copied from Chan-Lau, Quach and Shi 2025; the LSTM's 240 / 1e-4 / 300 were set by hand; the
TFT used default sizes). They stay in the results only as labelled reference rows.

- Task T2 (one row per meeting), target the rate change, standardised on the fit rows. Architectures LSTM, GRU
  and TFT; feature sets S+M, S+M+H, S+M+C. Every feature set gets the same search space and trial budget.
- Tuning points: the first test meeting of 2001, 2005, 2009, 2013, 2017, 2021 and 2025. At each, only meetings
  before that point are used: 4 expanding time-series folds (`models.expanding_folds(n, 4)`), features
  standardised inside each fold's training rows, early stopping on the fold's validation rows (at most 300
  epochs, patience 20). Objective: mean validation MSE over the folds.
- Optuna TPE sampler, seed 0, 30 trials per architecture, feature set and tuning point. Search space:
  hidden {8, 16, 32, 64, 128, 256} (TFT: {8, 16, 32, 64}); recurrent layers {1, 2} (LSTM/GRU); attention heads
  {1, 2, 4} (TFT); dropout 0-0.5; learning rate 1e-4 to 1e-2 (log); weight decay 1e-6 to 1e-2 (log);
  lookback {2, 4, 8, 12} meetings; batch size {16, 32, 64}. Epochs for the final fit: the median best epoch (at least 1) of
  the chosen trial's folds; the final fit uses all training rows with no early stopping.
- The chosen settings are frozen until the next tuning point. Walk-forward refits every 8 meetings, 8 seeds,
  seed-mean forecasts, as in Plan 2.
- Tests: Diebold-Mariano on squared rate errors and Clark-West, S+M+H against S+M per architecture; McNemar on
  accuracy. Reported with the same wording rules, as secondary evidence.
- **Compute rule:** one trial is timed first. If the projected total exceeds 12 hours, the budget drops to 20
  trials for every arm alike; this is decided before any tuned forecast exists.

## 6. Order of work and outputs

1. This document is committed. 2. Code with tests. 3. Token projection, then the Haiku batch. 4. Primary
comparison and leak diagnostics. 5. Tuning, then tuned neural walk-forward. 6. RESULTS_STUDY4.md with every
number, the leak section and the deviations, if any.

## 7. Amendments (each recorded before the step it affects)

7.1 (before submitting the batch). The token projection (results/study4/token_projection.json) is 23,980,280
tokens: 1,544 input tokens per request on average over 200 counted requests, 13,750 requests, plus 8 output
tokens per sentence. That exceeds the 20,000,000 limit, so nothing was submitted and the user was asked. The
user allowed the full corpus; the limit is now 25,000,000. Nothing else changes.

## 8. Sensitivity checks after the external review (exploratory, added after the section 4 results)

Recorded before any of these runs. None of them changes the section 4 verdict; all are labelled exploratory.

8.1 **Real-time data.** 96 of 212 test rows (to 2012-04-25) use at least one revised value from the ALFRED
fallback (average hourly earnings, U6, house prices). (a) Drop the four S columns built from those series
(`s_earn_yoy`, `s_u6_mom`, `s_u6_yoy`, `s_hpi_yoy`) and rerun S+M, S+M+H, S+M+C and S+M+N on the full test
window. (b) Report the original and the (a) forecasts on the test meetings after the last fallback row
(2012-06-20 onward, 116 meetings); a walk-forward forecast depends only on earlier rows, so this subset equals
a rerun that starts there. Training rows before 2001 still hold fallback values for PCE, PPI and core CPI
(their vintages start 1996-2000); this is stated as a remaining limitation.

8.2 **Which news columns carry the Plan 1 gain, in one pipeline.** The Plan 1 panel equals the Study 4
ChronoBERT panel row for row and column for column (checked). Cumulative: S+M, + ChronoBERT stance (2 columns),
+ Beige Book tone, + SF Fed news sentiment, + GPR, + EPU (the last is S+M+N). Single additions: S+M plus each
of Beige Book, news sentiment, GPR and EPU alone. Reported: log loss, accuracy, one-sided DM-HLN p on log loss
against S+M. No decision rule. Wording fixed now: "the higher accuracy appears in specifications that include
indices with vintage concerns"; no column is called the cause.

8.3 **Reporting.** Correct counts by class and net change against S+M (cuts, holds, hikes), and turning points
(decision class differs from the previous meeting's; these overlap the classes and are not added to them).
Primary comparison with Newey-West HAC standard errors (lags 4 and 8) and a moving-block bootstrap (blocks of
8 meetings, 10,000 draws, seed 0). McNemar reported one-sided (the registered direction) and two-sided.

8.4 **Wording.** The July and September 2026 meetings are a *retrospective post-cutoff evaluation*: Claude
Haiku 5.5 was released on 2026-10-07, after both meetings. Live forecasts are a separate, future exercise.

8.5 **Index provenance.** The daily indices were downloaded in 2026, so they are current vintages, not the
values published at each date. Known construction facts are reported (SF Fed: statistical adjustment for the
changing newspaper sample, 16 then 24 newspapers; GPR: methodology revised in 2021 and rebased; EPU:
normalised to its 1985-2009 mean). Archived copies of the files are compared with the current ones if they
can be retrieved; otherwise revision behaviour is reported as unverified.

## 10. Placebo news for the tuned networks (exploratory, added after the LSTM and GRU results)

Requested by the user after seeing the tuned LSTM and GRU results (LSTM gained with Haiku, GRU with
ChronoBERT). Recorded before any placebo run.

- Placebo arms, for each of LSTM, GRU and TFT: placebo-C and placebo-H. The two stance columns are shifted
  circularly by half the panel (130 of 260 rows), so each meeting gets the stance of a meeting about 13 years
  away. This keeps the series' distribution and persistence (which sequence models use) and breaks its link to
  the meeting being forecast. Every other column is unchanged.
- Each placebo arm gets the same Optuna search, budget, tuning points and 8 seeds as the real arms (section 5).
  LSTM and GRU placebo arms are tuned on the CPU while the TFT's real arms finish on the GPU; numerical results
  can differ slightly by device, with no expected direction.
- Reported for every architecture and reader: placebo against no news (DM-HLN, one-sided), and real news
  against its placebo (DM-HLN, one-sided: does the real stance beat the shifted one?), with accuracy and
  McNemar counts.
- Reading fixed now: a real-news gain counts as coming from the news content only if real news beats its own
  placebo at p < 0.05 and the placebo does not beat no news at p < 0.05. Otherwise the gain is attributed to
  tuning and training noise. All of this is exploratory and does not change the section 4 verdict.

## 11. Tuned monthly networks (T1) (exploratory, added after the T2 neural results)

Requested by the user, to test whether the poor monthly LSTM/GRU results of Plan 2 came from the reference
paper's settings. Recorded before any T1 tuning run.

- Task T1 as in Plans 1-2: one row per month t, cutoff the first day of t, target the monthly federal funds
  rate FEDFUNDS(t), forecast as the change from FEDFUNDS(t-1); class at +/-0.125. 378 rows (1995-03 to
  2026-09); test months from 2001-01 (308); the paper's window 2016-01 to 2025-04 (112 months) reported
  separately. A forecast at month t trains on rows up to t-2 (Plan 1-2 protocol); tuning at a point uses only
  rows before the point minus one.
- Inputs identical to Plan 2's T1 networks, so the comparison isolates the settings: S (the paper's Table 1
  statistics, real-time with the documented ALFRED fallback). News arms add only the two stance columns:
  S+C (ChronoBERT), S+H (Haiku); placebo-C and placebo-H shift the stance circularly by half the panel (189
  rows). No daily news indices.
- LSTM, GRU and TFT; the section 5 search space except lookback {3, 6, 12, 24} months; Optuna TPE seed 0,
  30 trials, 4 expanding folds; tuning points the first test month of 2001, 2005, ..., 2025; refit every 12
  months and at each tuning point; 8 seeds, seed-mean forecasts. (Plan 2 refit its LSTM/GRU every 3 months.)
- Comparisons, one-sided DM-HLN on squared rate errors, with accuracy and McNemar counts:
  (a) **settings question**: tuned S against Plan 2's paper-settings S forecasts of the same architecture on
  the same months (for the TFT, Plan 2 used default sizes); (b) news against no news; (c) news against its
  placebo, read as in section 10. Plan 1's random walk and linear ridge on S are reported for context, and the
  paper's own numbers (revised data) alongside.
- Reading fixed now for (a): "tuning explains the gap" for an architecture if tuned S beats paper-settings S
  at p < 0.05 on the 308 test months; the 112-month paper window is reported but not used for the reading.

## 12. Corrected network protocol, version 2 (amendment dated 2026-10-13, about 00:45)

**Disclosure.** This amendment follows inspection of earlier results: the version 1 network results
(sections 5, 10 and 11) had been seen, and an external review of the methods note found problems that affect
both tasks. The original sections stay as written; version 1 outputs are kept in
`results/study4/superseded_v1/` as superseded exploratory results. Version 2 replaces them for every network
result. The ordered-logit results (section 4, section 8) are not affected.

12.1 **Monthly cutoff.** For month t the forecast is made at 00:00 America/New_York at the start of the second
US business day of month t (business days: weekdays that are not US federal holidays, pandas
USFederalHolidayCalendar). Every input must be dated before that day: market prices (last value dated before
it), ALFRED vintages (published before it), Fed documents and the Beige Book (released before it), daily
indices (as before, three days earlier). The text window for month t runs from month t-1's cutoff to month
t's. FEDFUNDS(t-1), used as an input and as the base of the level forecast, is the monthly average of daily
effective rates; the last one (for the last business day of t-1) is published by the New York Fed at about
9:00 ET on the first business day of t, before the cutoff. Before March 2016 the daily effective rate appeared
in the Board's H.15 release; its exact posting time is not verified, so the cutoff leaves one full business
day of margin. The monthly panel is rebuilt with these cutoffs as a new file (`t1b`); the version 1 panel
(`t1`, first calendar day) is left unchanged.

12.2 **Identical validation folds.** At each tuning point, the validation rows are the same for every
candidate: the rows from (longest lookback in the search space - 1) up to the point (monthly: up to the point
minus one), split into 5 equal blocks; fold k trains on rows before block k+1 (monthly: before it minus one,
the t-2 rule) and validates on block k+1, for k = 1..4. A candidate's training rows are those rows with enough
history for its lookback. A candidate is infeasible (scored as infinite) if any fold has fewer than 10
training sequences.

12.3 **Score in rate units.** Each fold's best validation MSE is converted back to rate units (multiplied by
the fold's target variance) before averaging over folds. Early stopping inside a fold is unchanged (the same
epoch minimises both).

12.4 **Strict real-time arms.** In addition to the version 1 inputs (mixed vintage, disclosed fallback), each
task gets arms that use only series whose ALFRED vintages cover every row (policy rate, M2 monthly and yearly,
payroll change, nominal GDP, CPI, the two yield slopes; plus the market block in the meeting task), with no
fallback value anywhere in tuning, training or prediction: strict base, strict + ChronoBERT, strict + Haiku.

12.5 **Arms.** Per task and architecture: base, base + ChronoBERT, base + Haiku, placebo-ChronoBERT,
placebo-Haiku (mixed vintage) and the three strict arms: 8 arms x 3 architectures x 2 tasks. The placebo
(circular shift, section 10) is a retrospective negative control only, not proof of the absence of leakage.
The transformer is described as a TFT-inspired single-horizon model.

12.6 **Evaluation.** Primary comparisons, fixed now: in each task and architecture, base + ChronoBERT against
base, and base + Haiku against base (mixed vintage); Holm across the six within a task. Test: one-sided test
that the mean squared-error difference is positive, Newey-West standard errors with lags equal to the refit
interval (8 meetings, 12 months); paired moving-block bootstrap (same block length, 10,000 draws, seed 0) for
the squared-error and accuracy differences as robustness; HLN-DM and McNemar counts reported as descriptive.
Clark-West is dropped for the networks (separately tuned networks are not nested). The strict arms repeat the
two primary comparisons as robustness; placebo arms are reported as diagnostics. Meeting task: the primary
evaluation uses scheduled meetings only; all meetings are also reported, and forecasts for the unscheduled
decisions are described as conditional on the event date. Decision claims (cut, hold, hike) use the meeting
task only, with accuracy, balanced accuracy, recall by class and confusion matrices; the monthly task is a
rate-forecasting task (RMSE, MAE, MBE), and its direction classes are labelled as monthly average-rate
changes. Settings question (section 11): tuned strict-or-mixed base against Plan 2's paper-settings S forecasts
is reported descriptively, noting that Plan 2's LSTM settings do not match the paper version read on
2026-10-12 and that Plan 2 used the version 1 monthly cutoff.

12.7 **Verification before the batch.** (a) The cutoff rule against known calendars (holidays on the first
business days); (b) the monthly fold gap and identical validation rows across candidate lookbacks; (c)
rate-unit scoring; (d) end-to-end truncation: rebuilding sampled rows' features from raw inputs truncated at
their cutoff gives the stored features, and forecasting a row with later rows removed gives the same forecast.

12.8 **Compute.** LSTM and GRU arms run on the CPU and TFT arms on the GPU in parallel. One trial per
architecture and task is timed first; if the projected wall time exceeds 16 hours, every arm uses 20 trials
instead of 30.
