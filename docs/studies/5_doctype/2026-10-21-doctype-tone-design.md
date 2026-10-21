# Attempt 14: the distilled tone split by document type

Date: 2026-10-21. Branch `coral`. Written, committed and pushed before any result in this document is computed.

## 0. Status and disclosure

- This attempt was chosen after attempts 1-12b were known (results/ATTEMPTS.md). Most relevant:
  - Attempt 12b: the TFT with the Haiku-distilled ChronoBERT tone ("Hd") beat the TFT without news from 2010 (Holm
    p = 0.0050).
  - Its gain over a placebo tone was not significant (p = 0.082).
- It is exploratory and is listed as attempt 14. It does not replace any earlier verdict.
- Attempt 13 (Taylor-rule baseline) stays designed and not run.

**Motivation, from the inputs only (no decision was used).**
- Today the news input is one tone: the mean of three document groups.
- From 2010, that combined tone mostly repeats the statements and the minutes. Its correlation with each group's
  tone is: statements and press conferences 0.96, minutes 0.91, speeches 0.84.
- Speech tone moves on its own. Its meeting-to-meeting change has correlation 0.15 with the statement tone's change
  and 0.02 with the minutes' change.
- Speeches are the most numerous group (7.8 per meeting window on average).
- One average gives the speech signal one third of the weight. Separate columns let the model weigh each group.

Code: `src/fed_news/studies/doctype/`. Tests: `tests/studies/doctype/`. Results: `results/doctype/`.

## 1. The news columns ("Hs", 6 columns)

**Source.** The same document tones as attempt 12 (`<data>/distill/doc_stance_distill.pkl`). They come from
ChronoBERT trained on Haiku's probabilities and frozen at the end of 2009:
- documents from 2010 are scored by the final model;
- earlier documents are scored by the cross-fitted fold models.

**Columns.** For each meeting window [win_start, cutoff), the same windows as `n_stance`:
- `n_stance_policy`: mean tone of the statements and press conferences released in the window.
- `n_stance_minutes`: mean tone of the minutes released in the window.
- `n_stance_speech`: mean tone of the speeches released in the window.
- `n_stance_<group>_chg`: the change from the previous meeting, one for each of the three groups.

**Gaps.**
- When a window has no document of a group, the group's last value is carried forward.
- Before a group's first document, the meeting's combined tone `n_stance` (the attempt 12 column) is used. This
  affects 17 panel meetings for statements (1995-03-28 to 1997-03-25) and 10 for speeches (1995-03-28 to
  1996-05-21). All of them are training rows only.

**Meetings.** The panel has the same 260 meetings as the CORAL panels (checked), so the S+M forecasts of the CORAL
study are reused.

## 2. Versions

**Ordered logit.** Same engine as attempt 12.
- S+M (reused);
- S+M+Hs;
- placebo-Hs: the 6 columns shuffled across the training rows, the engine's placebo;
- shuffled-Hs: the training labels shuffled.

**Networks.** CORAL LSTM, GRU and TFT; same engine as attempt 12b.
- Settings: tuned per arm, 20 Optuna trials at each of 7 tuning points, 8 seeds.
- S+M+Hs;
- placebo-Hs: the 6 columns shifted by half the panel length, as for every earlier placebo;
- S+M is reused.

**Engine change.** `Spec` gets an optional `news_cols` field naming the columns the placebo shuffles. This is needed
because the 6 new columns are not in the original news column list. When it is not given, the behaviour is
unchanged.

## 3. Tests with a rule

The structure is the same as attempt 12.

**Primary.**
- **Test:** one-sided Clark-West, ordered logit S+M+Hs against S+M, on the 135 test meetings from 2010-01-01. One
  test, no correction.
- **Supported** if p < 0.05 and the result is not void.
- **Void** if the placebo beats S+M at Clark-West p < 0.05, or the shuffled control does so at p < 0.01.
- **Wording:**
  - If supported: "in this exploratory attempt, the distilled tone split by document type improved ordered-logit
    forecasts of FOMC decisions from 2010 (p = …)".
  - Otherwise: "not significant at 5% (p = …)".

**Declared secondary.**
- **Test:** for each network, one-sided Newey-West (8 lags) on the log-loss difference, S+M+Hs against S+M, on the
  134 scheduled meetings from 2010. Holm over the 3 networks.
- **Placebo reading.** For each network, the gain is said to come from the tone's content only if both hold:
  - S+M+Hs beats placebo-Hs at p < 0.05;
  - placebo-Hs does not beat S+M at p < 0.05.

## 4. Reported with no rule attached

- The full test period from 2001.
- **Split against the single tone:** S+M+Hs against S+M+Hd (Clark-West and DM-HLN for the logit; Newey-West for the
  networks). This answers "does the extra dimension help?".
- Accuracy, log loss, macro F1, McNemar fixes and breaks.
- For the networks, S+M+Hs against S+M: fixed-b, fixed-m, and the ranked probability score with Newey-West
  (results/sigtests methods).

## 5. Order of work

1. Push this document.
2. Write the code with tests, then push it.
3. Run the ordered logit on the CPU (minutes).
4. Run the networks on CUDA. At attempt 12b's timings this takes about 4-5 hours: tuning 2 h 45 min, forecasting
   1 h 26 min. The run is detached, with its scripts in `D:/fed_news/runs/14_doctype/`.
5. Write the result into ATTEMPTS.md, whatever it is.
