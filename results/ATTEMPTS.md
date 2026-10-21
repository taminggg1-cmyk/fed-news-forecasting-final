# Every attempt so far, with its primary test

This is the record a reader needs to judge any single p-value: how many attempts were made, which test each one
committed to in advance, and what came out. A p-value below 0.05 found on the tenth attempt is weaker evidence than
the same p-value found on the first.

**How to read it.**
- A p-value is **one-sided** ("news is better") unless marked two-sided.
- **Holm** means the p-value was corrected for the number of tests in that attempt's family.
- **Registered** means the test and its family were written down and pushed before the result was computed.
- **Exploratory** means the analysis was chosen after earlier results were known.

## A. Attempts in the original repository (github.com/taminggg1-cmyk/fed-news-forecasting)

The full write-ups are in that repository. Only the headline primary result is repeated here.

| # | Attempt | Primary result | Verdict |
|---|---|---|---|
| 1 | Plan 1 (pre-registered, tag `prereg`), monthly, hypothesis H1 | p = 0.103 | not supported |
| 2 | Plan 2, hypothesis H2a | 0 of 10 tests significant after Holm | not supported |
| 3 | Plan 5 / Study 2 | Clark-West p = 0.129 and 0.369 | not supported |
| 4 | Proofs of concept | did not pass their checks | not supported |
| 5 | Study 5 | p = 0.898 and 0.300 | not supported |

## B. Attempts in this repository

| # | Attempt | Primary test (status) | Result | Verdict |
|---|---|---|---|---|
| 6 | **Study 4**: ordered logit, 212 meetings, + ChronoBERT (C) / + Haiku (H) tone | DM-HLN on log loss (registered) | H: p = 0.144; C: p = 0.207. Clark-West: H 0.110, C 0.189 | not supported |
| 7 | **Study 4 version 2**: tuned LSTM/GRU/TFT per meeting, 206 scheduled meetings | Newey-West on squared error, Holm over 6 (dated amendment written after version 1 results; not a pre-registration) | **LSTM + Haiku: p = 0.0034, Holm p = 0.020**; the other 5 had Holm p ≥ 0.85 | 1 of 6 significant |
| 8 | Study 4 version 2, monthly (308 months) | Newey-West on squared error, Holm over 6 | best: LSTM + Haiku p = 0.026 (Holm 0.156); TFT + C 0.042 (Holm 0.169); TFT + H 0.034 (Holm 0.168) | not supported |
| 9 | **CORAL study** (design 2026-10-20, registered): ordered logit with ChronoBERT-CORAL (Co) and Haiku ordinal (Ho) tones, 212 meetings | Clark-West, Holm over 2 | Ho: p = 0.066 (Holm 0.133); Co: p = 0.424 | not supported |
| 10 | CORAL study, CORAL LSTM/GRU/TFT, 206 scheduled meetings | Newey-West (8 lags) on log loss, Holm over 6 | best: LSTM + Ho p = 0.024 (Holm 0.145); TFT + Ho p = 0.033 (Holm 0.163); GRU + Ho 0.076 (Holm 0.303); all Co Holm ≥ 0.44 | not supported |
| 11 | CORAL amendment 3 (declared after the full-sample p-values): same tests on meetings from 2010 | as attempts 9 and 10 | logit Ho p = 0.0498 (Holm over 2: 0.0997); networks: TFT + Ho p = 0.012 (Holm over 6: 0.073) | not supported |
| 12 | **Haiku-distilled ChronoBERT** (design 2026-10-21, Part B; declared before the run, chosen after attempts 9-11): ChronoBERT 2009 trained on Haiku's probabilities for 144,211 pre-2010 sentences, frozen, reads 2010 onward alone | Clark-West, one test, ordered logit, 135 meetings from 2010 | p = 0.220 (placebo 0.105, shuffled 0.930, not void). Log loss 0.356 to 0.320, accuracy 83.7% to 85.9% (Haiku ordinal itself: 0.317, 85.9%). Full period: p = 0.157 | not supported |
| 12b | Same tone in the CORAL LSTM/GRU/TFT (declared secondary part of attempt 12) | Newey-West (8 lags) on log loss, 134 scheduled meetings from 2010, Holm over 3 | **TFT: p = 0.0017, Holm p = 0.0050**; LSTM 0.110 (Holm 0.220); GRU 0.167 (Holm 0.220). TFT log loss 0.403 to 0.318, accuracy 88.8% to 89.6%. Small-sample checks for TFT: fixed-b 0.0061, fixed-m 0.0117, RPS 0.0056. Placebo reading: TFT real tone vs its placebo p = 0.082 (placebo vs S+M 0.237), so the gain is not shown to come from the tone's content; LSTM real vs placebo p = 0.032 | **TFT significant against S+M; not significant against its placebo** |
| 14 | **Distilled tone split by document type** (design 2026-10-21, docs/studies/5_doctype/; declared before the run, chosen after attempts 1-12b): the same distilled tone as attempt 12 as 6 columns (statements and press conferences, minutes, speeches; level and change) | Clark-West, one test, ordered logit, 135 meetings from 2010 | **p = 0.0100** (placebo 0.456, shuffled 0.933, not void). Log loss 0.356 to 0.322, accuracy 83.7% to 85.9%; DM-HLN on log loss p = 0.032, Newey-West (8 lags) 0.061. Against the single distilled tone (S+M+Hd: log loss 0.320, accuracy 85.9%): Clark-West p = 0.071, DM-HLN on log loss p = 0.544, so the split is not shown to beat the single tone. Full period (no rule): p = 0.0102, placebo 0.914 | **supported** (exploratory; 14th attempt) |
| 14b | Same split tone in the CORAL LSTM/GRU/TFT (declared secondary part of attempt 14) | Newey-West (8 lags) on log loss, 134 scheduled meetings from 2010, Holm over 3 | LSTM p = 0.099 (Holm 0.297); GRU 0.186 (Holm 0.372); TFT 0.408 (Holm 0.408). Log loss: LSTM 0.665 to 0.496, GRU 0.560 to 0.441, TFT 0.403 to 0.391; accuracy: LSTM 76.1% to 83.6%, GRU 76.9% to 88.8%, TFT 88.8% to 92.5%. Placebo reading: only the GRU's gain is shown to come from the tone's content (real vs placebo p = 0.010; placebo vs S+M 0.594); the TFT's placebo beat S+M (p = 0.047) and its real tone did not beat the placebo (0.899). Against the single distilled tone of 12b: p = 0.42 (LSTM), 0.43 (GRU), 0.95 (TFT; log loss 0.391 against 0.318). Small-sample checks from 2010 (LSTM / GRU / TFT): fixed-b 0.113 / 0.210 / 0.414, fixed-m 0.140 / 0.234 / 0.421, RPS with Newey-West 0.095 / 0.055 / 0.073. Full period (no rule): 0.070 / 0.303 / 0.239 | not supported |

Supporting facts for reading the Haiku results (from Study 4):
- Haiku recalls 87.8% of FOMC decisions from the date alone.
- In Study 4, its gain over ChronoBERT was not larger on the meetings it recalled (p = 0.378).
- In the CORAL study the same memory check gave p = 0.205 (full sample) and 0.132 (from 2010).

Notes on attempt 14:
- Its network part (declared secondary, Holm over 3) is row 14b. Both network steps (tuning and forecasting) ended
  with a Windows fail-fast exit code (0xC0000409) after all their outputs had been written. The automatic retry
  reused the saved settings and seed forecasts; the retried forecast step wrote the same summary as the first run.
- In `results/doctype/logit/verdict.json` the "full" block repeats the decision and wording template of the primary
  test, which says "from 2010". The full period has no rule attached; only the from-2010 decision counts.

## C. Exploratory checks, done after the results (they test no new claim)

- **Why the split tone's Clark-West p is much lower than the single tone's** (attempt 14 against attempt 12, from
  2010; chosen after both results).
  - The two tones lower the mean log loss by about the same amount: 0.034 (split) and 0.036 (single) per meeting.
  - The split tone's gain is steadier: the standard deviation of the per-meeting log-loss gain is 0.21, against 0.35
    for the single tone.
  - The single tone's largest losses in the Clark-West statistic were the two emergency meetings of March 2020
    (2020-03-03 and 2020-03-15).
  - Without those two meetings, Clark-West p is 0.0098 for the split tone and 0.058 for the single tone. The split
    tone's result does not depend on them.

- **Romano-Wolf correction instead of Holm** (2026-10-21, scratch script): the 6 network tests are only weakly correlated (mostly −0.5 to +0.5). The Romano-Wolf adjusted p-values were larger than Holm's for the best tests: LSTM + Ho 0.180 vs 0.145; TFT + Ho 0.210 vs 0.163. A correlation-aware correction does not rescue any test.
- **Literature tests on the saved forecasts** (design 2026-10-21, Part A; full tables in results/sigtests/).
  576 pairwise tests, 8 in-sample tests, joint tests per family. What they show:
  - **Calibration on the placebo (fake-tone) versions.**
    - Clark-West on the networks' rate forecasts calls a fake tone significant in 11 of 12 cases (8 of 12 with Newey-West 8 lags). It is invalid for separately tuned networks, which are not nested models, and its small network p-values are not evidence.
    - The log-score tests are calibrated: Newey-West and fixed-b 0 of 12, fixed-m 1 of 12.
    - Ranked probability score (RPS) 1 of 12, and Giacomini-White 1 of 12; both of those false positives are TFT placebos.
    - On the ordered logit every test is clean (0 of 4).
  - **Small-sample corrections (Coroneo and Iacone 2020) raise the network p-values.**

    | Test | NW 8 lags | fixed-b | fixed-m |
    |---|---|---|---|
    | LSTM + Ho | 0.024 | 0.034 | 0.055 |
    | TFT + Ho | 0.033 | 0.051 | 0.067 |
    | TFT + Ho from 2010 | 0.012 | 0.025 | 0.032 |

  - **Ranked probability score** (the proper score for ordered classes) gives the smallest p-values for the Haiku ordinal tone: LSTM 0.0086, GRU 0.011, TFT 0.021, and TFT 0.0084 from 2010. Holm over the registered 6 on this score is **0.0515** (full) and **0.0502** (from 2010), just above 0.05.
  - **Giacomini-White** (conditional, two-sided): TFT + Ho p = 0.0007, Holm over 6 0.004. But GW also rejects for the TFT placebo (0.030) and for both Study 4 tones on TFT (0.0037). It picks up the TFT's persistent performance swings within refit blocks, not an average gain, so it is not evidence that the tone helps.
  - **Joint tests (log score).**
    - SPA over the registered 6: p = 0.067 (full), 0.196 (from 2010).
    - SPA over the 3 Haiku ordinal networks only (a family not registered): p = 0.037, with StepM marking LSTM + Ho as superior at 5%.
    - Ordered logit (Co, Ho): 0.114.
    - The 90% model confidence set drops S+M only for the LSTM (full period).
  - **In-sample ordered logit (Inoue and Kilian).**
    - The chi-square likelihood-ratio p-values look strong: Ho 0.0047, H 0.023, Co 0.0006 (with the wrong sign).
    - But the market columns nearly separate the classes. Circularly shifted fake tones reach likelihood-ratio gains of about 17-22 at their 95th percentile, against 6.0 for chi-square(2).
    - Against that rotation placebo, the in-sample p-values are Ho 0.16, H 0.22, Co 0.10, C 0.59.
    - From 2010 the in-sample fit is not identified (separation).
  - **Summary:** no test that passes its own placebo check gives p < 0.05 after Holm over the registered family. The closest is the ranked probability score with Newey-West at Holm 0.050-0.052.
