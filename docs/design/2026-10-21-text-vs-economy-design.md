# Attempt 13: does Fed text help forecast FOMC decisions beyond economic data (no market prices)?

Date: 2026-10-21. Branch `coral`. Written, committed and pushed before any result in this document is computed.

## 0. Status and disclosure

The project's question is "does Fed text help forecast FOMC decisions?". Every registered test so far (Study 4, the
CORAL study, the distilled ChronoBERT) put market prices in the baseline (M: Treasury bill and yield spreads, the
target level, the zero bound, VIX, oil). In effect they asked the harder question "does text help beyond what
markets already know?". None reached p < 0.05 after its correction (results/ATTEMPTS.md, attempts 6-12).

Published studies of this question mostly use a Taylor-rule baseline (for example Hayo and Neuenkirch, ordered
probit). This attempt asks that question:
- **Baseline:** the economic data, with no market prices.
- **Engine:** the same out-of-sample engine and placebo checks as before.

It was chosen after attempts 1-12 were known, and that is stated wherever its result is reported. It is exploratory
and is listed as attempt 13. It does not replace any earlier verdict.

Code: `src/fed_news/textonly/`. Results: `results/textonly/`.

## 1. Versions (ordered logit, same engine as the CORAL study section 6)

Walk-forward from 2001 with the penalty chosen by cross-validation inside the training rows. All versions use the
CORAL meetings (the common panel of the H, C, Co, Ho and Hd tones).

**Primary baseline T, the Taylor rule:**
- columns: s_rate_lag, s_pce_core_yoy, t_u3 (the previous target rate, core PCE inflation, unemployment);
- T+Hd: T plus the Haiku-distilled ChronoBERT tone (n_stance, n_stance_chg);
- T+Ho: T plus the Haiku ordinal tone;
- placebo and shuffled-label controls for both tones.

**Reported, no rule attached:**
- T+Co, T+H and T+C (the ChronoBERT-CORAL tone and the Study 4 Haiku and ChronoBERT tones);
- baseline E, the 16 economic statistics (S), with E+Hd and E+Ho. S contains two Treasury yield-curve slopes, so E is
  not fully free of market prices.

## 2. Primary test

- **Test:** one-sided Clark-West, T+Hd and T+Ho against T.
- **Meetings:** test meetings from 2010-01-01 (135 meetings).
  - From 2010 the distilled tone comes only from the frozen 2009 model.
  - Before 2010 it comes from cross-fitted models with 2009 knowledge.
- **Correction:** Holm over the 2 tones.
- **Supported** for a tone if its Holm p < 0.05 and it is not void.
- **Void** if its placebo beats T at Clark-West p < 0.05, or its shuffled control does so at p < 0.01.
- **Wording:**
  - If supported: "in this exploratory attempt, with a Taylor-rule baseline without market prices, the <tone> tone
    improved ordered-logit forecasts of FOMC decisions from 2010 (p = …, Holm-adjusted)".
  - Otherwise: "not significant at 5% (p = …, Holm-adjusted)".
- **Haiku memory caveat:** every Haiku ordinal result carries the memory caveat. The distilled tone sees no Haiku
  output after 2009.

## 3. Reported with no rule attached

- The full test period from 2001.
- Every other version.
- Accuracy, log loss, DM-HLN and Newey-West on log loss, McNemar.
- In-sample (all panel meetings, and from 2010): unpenalised ordered logit T against T plus tone. Reported for each
  tone:
  - the likelihood-ratio test;
  - its rotation-placebo p-value (99 circular shifts, results/sigtests method).

## 4. Order of work

1. Push this document.
2. Write the code with tests.
3. Run it on the CPU (the GPU stays with the running network job).
4. Write the result into ATTEMPTS.md, whatever it is.
