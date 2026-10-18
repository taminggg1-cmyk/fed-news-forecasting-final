# Results: Study 4 (Haiku news index, tuned neural forecasters) - exploratory

Design: `docs/design/2026-10-12-study4-haiku-index-tuned-design.md` (committed before any call or
run; amendment 7.1 and section 8 recorded before the steps they affect). The 2001-2026 meetings were used by
every earlier study, so nothing here is a confirmatory test. Neural results (section 5 of the design) are
still running and will be added.

*Note added 2026-10-18 for this repository: the neural results (Study 4 version 2) are in `results/study4/neural/`
and `results/study4/neural_t1/` and are summarised in the README. "S+M+N (Plan 1 news)" rows below use the
six-column news block of an earlier design, run here only as a comparison.*

## Haiku news index

- Model `claude-haiku-5-5`: a pinned snapshot (Anthropic models overview: dateless IDs are pinned from the 4.6
  generation on), released 2026-10-07, reliable knowledge and training data cutoff June 2026. Every batch
  reply reports this model ID.
- 343,742 sentences from 2,578 documents (statements, minutes, press conferences, speeches), masked, in random
  order, 25 per request, through the Message Batches API; 343,717 labelled (one request failed twice).
  23,395,731 tokens (21,392,903 input, 2,002,828 output). Shares: hawkish 8.5%, dovish 12.4%.

## Primary comparison (T2, ordered logit, 212 meetings, 2001-01-03 to 2026-09-16)

| model | log loss | accuracy | one-sided DM p (log loss) vs S+M |
|---|---:|---:|---:|
| S+M (no news) | 0.2938 | 0.868 | |
| S+M+H (Haiku stance) | 0.2721 | 0.892 | 0.144 |
| S+M+C (ChronoBERT stance) | 0.2814 | 0.873 | 0.207 |
| placebo-H | 0.2913 | 0.868 | 0.441 |
| shuffled-H | 0.8796 | 0.660 | 1.000 |

**Verdict: not supported** (primary p 0.144; controls do not void). Haiku against ChronoBERT: p 0.190.
McNemar, S+M+H against S+M: 6 meetings fixed, 1 broken; exact p 0.0625 one-sided (the registered direction),
0.125 two-sided.

Memory flag (Haiku's gain minus ChronoBERT's, regressed on whether Haiku recalled the decision from its date):
coefficient 0.010, one-sided p 0.378, 190 of 211 meetings recalled. Not raised. With only 21 meetings not
recalled the test has little power: it did not detect the pattern; it does not show that memory is absent.
Dated indicator (descriptive): coefficient 0.012, p 0.321.

Retrospective post-cutoff evaluation (Haiku 5.5 was released after both meetings, so these are not live
forecasts): 2026-07-29 (hold) log loss S+M 1.204, S+M+H 1.456, S+M+C 1.052; 2026-09-16 (hike) 0.124, 0.099,
0.117. Two meetings carry no statistical weight.

## Sensitivity checks after the external review (design section 8; exploratory, added after results)

### Dependence between meetings

The primary log-loss gain (mean 0.0217 per meeting): Newey-West one-sided p 0.156 (4 lags) and 0.154 (8 lags);
moving-block bootstrap (blocks of 8 meetings, 10,000 draws) 90% interval [-0.008, 0.061]. Same conclusion as
the registered test.

### Correct forecasts by class (net change against S+M)

| group | n | S+M | S+M+H | S+M+C | S+M+N (Plan 1 news) |
|---|---:|---:|---:|---:|---:|
| cut | 34 | 24 | 24 (0) | 25 (+1) | 25 (+1) |
| hold | 140 | 128 | 132 (+4) | 128 (0) | 134 (+6) |
| hike | 38 | 32 | 33 (+1) | 32 (0) | 34 (+2) |
| turning point* | 42 | 28 | 32 (+4) | 28 (0) | 32 (+4) |
| all | 212 | 184 | 189 (+5) | 185 (+1) | 193 (+9) |

\* Decision class differs from the previous meeting's; these overlap the three classes and are not added.
Four of Haiku's five net gains are holds. Every model is weakest on cuts and turning points.

### Which news columns carry the Plan 1 gain (one pipeline, same 212 meetings)

| model | accuracy | log loss | DM p vs S+M |
|---|---:|---:|---:|
| S+M | 0.868 | 0.2938 | |
| + ChronoBERT stance | 0.873 | 0.2814 | 0.207 |
| + Beige Book tone | 0.877 | 0.2793 | 0.228 |
| + SF Fed news sentiment | 0.892 | 0.2638 | 0.122 |
| + GPR | 0.887 | 0.2606 | 0.097 |
| + EPU (= S+M+N, Plan 1) | 0.910 | 0.2537 | 0.095 |
| S+M + Beige Book only | 0.882 | 0.2812 | 0.227 |
| S+M + news sentiment only | 0.892 | 0.2554 | 0.068 |
| S+M + GPR only | 0.887 | 0.2854 | 0.096 |
| S+M + EPU only | 0.901 | 0.2709 | 0.121 |

The higher accuracy appears in specifications that include indices with vintage concerns (below). This does
not show that those indices cause it, or that they leak.

### Real-time economic data

96 of 212 test rows (to 2012-04-25) use at least one revised value from the ALFRED fallback.

| window | columns | S+M | +Haiku (p) | +ChronoBERT (p) | +all news (p) |
|---|---|---:|---:|---:|---:|
| all 212 | original | 0.868 | 0.892 (0.144) | 0.873 (0.207) | 0.910 (0.095) |
| all 212 | fallback columns dropped | 0.887 | 0.901 (0.038) | 0.882 (0.190) | 0.896 (0.071) |
| 116 after 2012-06-20 | original | 0.819 | 0.836 (0.178) | 0.819 (0.140) | 0.879 (0.057) |
| 116 after 2012-06-20 | fallback columns dropped | 0.836 | 0.845 (0.165) | 0.828 (0.112) | 0.871 (0.056) |

Accuracy, with one-sided DM p on log loss against the S+M row of the same block. One of these twelve
exploratory comparisons is below 0.05 (Haiku with the fallback columns dropped, p 0.038); with that many
comparisons one such value is expected by chance (Holm across the twelve news rows: not significant).
In the window where every test-row input is real-time, no news reader reaches p < 0.05.

## Limitations

- **Daily news indices are current vintages.** The SF Fed news sentiment, GPR and EPU files were downloaded
  in 2026, not as published at each date. Known construction facts: the SF Fed index adjusts statistically for
  the changing newspaper sample (16 newspapers in 2020, 24 now); GPR's methodology was revised in 2021 and
  rebased; EPU is normalised to its 1985-2009 mean. Archived copies could not be retrieved (archive.org was
  offline), so revision behaviour is unverified. These indices are in Plan 1's S+M+N, not in the Study 4
  primary comparison.
- **Revised economic data** in 45% of test rows (ALFRED fallback, to April 2012), shared by every model; it
  does not guarantee the bias cancels between models with and without news.
- **Haiku memory is not identified**: it recalls 87.8% of decisions from dates alone and dates 89.5% of masked
  statements within a year; a quiet memory flag is weak evidence.
- **Human labels** (Trillion Dollar Words, World Central Banks) were made after the events; annotators may have
  known what followed.
- **Effective sample**: 212 meetings, not 343,717 sentences; errors within a policy cycle are correlated.
