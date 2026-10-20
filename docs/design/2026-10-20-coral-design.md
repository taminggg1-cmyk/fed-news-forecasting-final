# CORAL study: ordinal learning for the tone readers and the neural forecasters (exploratory)

Date: 2026-10-20. Requested by the user after a meeting with their professor. Fixed and committed before any
code for this study is written, any model is trained and any API call is made.

## 0. Status and disclosure

This study was designed after every earlier result was seen. In particular these numbers were known when the
rules below were written (212 test meetings, 2001-01-03 to 2026-09-16, ordered logit, results/study4/):

| Comparison | Log loss (no news / news) | DM-HLN p (log loss, one-sided) | Clark-West p (rate forecast, one-sided) |
|---|---|---|---|
| S+M+H against S+M | 0.294 / 0.272 | 0.144 | 0.110 |
| S+M+C against S+M | 0.294 / 0.281 | 0.207 | 0.189 |

and, for the version 2 networks, LSTM + Haiku against no news: Newey-West p 0.0034 on squared rate errors,
Holm over 6 p 0.020. The 2001-2026 meetings have been used by every study, so every result here is
**exploratory**. The aim stated by the user is to find out whether ordinal learning improves the forecasts
and lowers the p-values; every attempt and every number is reported whatever the outcome (section 11).

## 1. Question

Does ordinal learning, as in CORAL (Cao, Mirjalili and Raschka, "Rank consistent ordinal regression for neural
networks with application to age estimation", arXiv:1901.07884; Pattern Recognition Letters 140, 2020), improve
(a) the Fed tone measured by ChronoBERT and by Claude Haiku 5.5, and (b) the LSTM, GRU and TFT forecasts of FOMC
decisions, compared with the no-news model?

## 2. What CORAL is, in the form used here

Ranks: the outcome has K = 3 ordered classes with rank r in {0, 1, 2}. For tone: dovish 0 < neutral 1 <
hawkish 2. For decisions: cut 0 < hold 1 < hike 2.

- **Targets.** Each rank becomes K - 1 = 2 binary levels, y_k = 1{r >= k} for k = 1, 2:
  rank 0 = (0, 0), rank 1 = (1, 0), rank 2 = (1, 1).
- **Output layer.** One score g(x) = w . h(x) from the network's last hidden vector h(x), with no bias, and two
  separate biases: logit_k = g(x) + b_k. The weight vector w is shared by both binary tasks; only the biases
  differ. Biases are initialised as in the authors' reference implementation (coral-pytorch):
  b = (1.0, 0.5).
- **Loss.** The CORAL loss with equal task weights (lambda_k = 1):
  L = -(1/N) sum_i sum_k [ y_ik log s(logit_ik) + (1 - y_ik) log(1 - s(logit_ik)) ], s the logistic function.
  This replaces softmax cross-entropy (ChronoBERT) and mean squared error (LSTM, GRU, TFT).
- **Probabilities.** P(r >= 1) = s(logit_1), P(r >= 2) = s(logit_2); class probabilities
  P(0) = 1 - s(logit_1), P(1) = s(logit_1) - s(logit_2), P(2) = s(logit_2).
- **Predicted class.** The paper's rule: r_hat = 1{s(logit_1) > 0.5} + 1{s(logit_2) > 0.5}.
- **Rank consistency.** Because w is shared, s(logit_1) >= s(logit_2) for every input if and only if
  b_1 >= b_2. The paper proves the loss optimum has ordered biases; we check b_1 >= b_2 on every trained model and
  report any violation (section 8). A violation is not repaired silently: the model is listed, and its
  class probabilities are clipped at zero and renormalised.
- **Relation to the ordered logit.** With h(x) = x (no hidden layers) this is the cumulative logit model already
  used as the simple model (models/linear.py, OrderedLogit). The ordered logit is therefore not changed; it only
  receives the new tone inputs.

## 3. Where the files go (branch `coral`)

Nothing outside these paths is changed, so every earlier result stays reproducible.

```
docs/design/2026-10-20-coral-design.md    this document
src/fed_news/coral/                       all new code
tests/coral/                              all new tests
results/coral/                            all new results
<data>/coral/                             Haiku batch state, sentence-level outputs, tone per document
D:/fed_news/chronobert_coral_models/      every fine-tuned ChronoBERT-CORAL model (not in git)
```

## 4. Reader 1: ChronoBERT-CORAL

Identical to features/chronobert_tone.py except the output layer, the loss and the label order:

- Same per-year protocol: classifier year y (2000 onward; documents 1994-1999 use y = 2000) fine-tunes ChronoBERT
  vintage y - 1 (clipped to 1999-2024) on human labels dated before y; same 20% holdout (rng seed 0); same
  settings (learning rate 2e-5, 3 epochs, batch 16, max length 256, AdamW weight decay 0.01, fp16 autocast,
  gradient clipping 1.0); same mean-pooled token vectors; seeds 0, 1, 2.
- Labels re-coded to rank order: the stored codes are dovish 0, hawkish 1, neutral 2 (download/stance_labels.py);
  the CORAL ranks are dovish 0, neutral 1, hawkish 2.
- Output layer and loss as in section 2.
- Sentence tone = P(hawkish) - P(dovish), the same scale as before; the three seeds' level probabilities
  P(r >= 1) and P(r >= 2) are averaged before conversion (an average of ordered pairs stays ordered).
  Document tone = mean over its sentences. Saved as <data>/coral/doc_stance_chronobert_coral.pkl in the
  doc_stance.pkl format, with sentence-level P(r >= 1) and P(r >= 2) kept per year.
- **Every model is saved**: <D:/fed_news/chronobert_coral_models>/<year>/seed<k>/ with the encoder in Hugging Face
  format (save_pretrained, float32), the tokenizer, the output layer (w, b) and a JSON file with the vintage,
  the training rows' date range, the label order and the settings. Classifier years whose training labels and
  vintage are identical (2025 and 2026) still each get their own folder.
- Holdout report per year: accuracy and macro F1 (comparable with results/stance_accuracy.csv), plus the paper's
  ordinal error, mean absolute rank error.

## 5. Reader 2: Claude Haiku 5.5 ordinal probabilities

Haiku cannot be trained by us (no fine-tuning of Claude Haiku 5.5 is documented for the Anthropic API, and its
output layer is not accessible). CORAL's output form is used instead: Haiku is asked for the two level probabilities of
each sentence. No training is involved; the paper's rank-consistency guarantee does not apply, so it is
enforced as stated below.

- Same sentences (343,742), same masking (features/haiku_prompt.mask), same random order (seed 0), 25 sentences per
  request, Message Batches API, model `claude-haiku-5-5`, thinking disabled, effort low (the attempt-1 settings of
  Study 4). max_tokens 2048.
- System prompt: the Study 4 definitions of H, D and N, unchanged, followed by:
  "For each sentence give two whole numbers from 0 to 100: A, the probability that the sentence is not dovish
  (it is neutral or hawkish); B, the probability that it is hawkish. B can never be larger than A. Judge only the
  text of each sentence. Reply with one JSON object mapping each sentence number to [A, B], and nothing else."
  The prompt is not tuned on any outcome.
- Parsing: A and B must be numbers in [0, 100]. If B > A, B is set to A and the sentence is counted as
  inconsistent (the count is reported). Failed, refused or cut-off requests are resent once; sentences still
  missing are left out of their document's mean and counted.
- Sentence tone = B/100 - (1 - A/100) = P(hawkish) - P(dovish). Document tone = mean over its sentences. Saved as
  <data>/coral/doc_stance_haiku_ordinal.pkl.
- **Token and cost rule:** the input tokens of 200 random requests are counted with the token-counting endpoint
  and scaled to all requests; output is projected from 15 tokens per sentence. If the projection exceeds
  35,000,000 tokens, nothing is submitted and the user is asked. (Study 4 used 21.4 million input and 2.0
  million output tokens.)
- Descriptive checks, no rule: correlation of the new document tone with the Study 4 Haiku tone; share of
  sentences where the class with the largest probability equals the Study 4 label.
- The Study 4 leak layers still apply (masking, random order, no "ignore the future" instruction) and the Study 4
  red flag (Test B) is reported next to every Haiku result.

## 6. Forecaster 1: the ordered logit (unchanged model, new inputs)

Same engine and rows as Study 4 section 4 (walk_forward.run_spec with class probabilities; penalty by
expanding-window CV inside the training rows; standardisation on the training rows).

| Name | Features |
|---|---|
| S+M | no news (the Study 4 forecasts, reused unchanged) |
| S+M+Co | S+M plus the ChronoBERT-CORAL tone (n_stance, n_stance_chg) |
| S+M+Ho | S+M plus the Haiku ordinal tone |
| placebo-Co, placebo-Ho | the news columns permuted across training rows (Study 4 placebo) |
| shuffled-Co, shuffled-Ho | the training targets permuted (Study 4 control) |

Rows: the meetings where every column is present in every panel. If that differs from the Study 4 212 rows,
S+M is rerun on the new rows and the difference is reported.

- **Primary test: Clark-West** (one-sided, horizon 1, evaluation/clark_west.py) on the rate forecast
  (last rate plus the class probabilities times the training rows' mean move per class), S+M+Co against S+M and
  S+M+Ho against S+M; Holm over these 2.
- Reported without a rule: DM-HLN on log loss (the Study 4 primary), accuracy, McNemar, Brier, Newey-West
  versions (lags 4 and 8).
- Validity: a CORAL tone result is "void" if its placebo beats S+M at p < 0.05 or its shuffled control at
  p < 0.01 (Clark-West), as in Study 4.
- Memory flag for S+M+Ho exactly as Study 4 section 4 (Haiku gain minus ChronoBERT-CORAL gain regressed on the
  recalled indicator, HC3).

## 7. Forecaster 2: CORAL LSTM, GRU and TFT

The version 2 protocol (Study 4 section 12, task T2, meetings) with one change: the output layer, the loss and
the scoring are CORAL.

- Architectures: LSTM, GRU, TFT (models/neural.py, models/tft.py); their final linear layer is replaced by the
  CORAL layer of section 2 (one shared weight vector without bias, two biases).
- Target: the decision rank (cut 0, hold 1, hike 2) as levels; the training loss is the CORAL loss. No target
  standardisation is needed.
- Arms (7 per architecture): S+M; S+M+C and S+M+H (the Study 4 tones, so the CORAL forecaster can be judged
  with the old tone); S+M+Co and S+M+Ho (the new tones); placebo-Co and placebo-Ho (tone shifted circularly by
  half the panel, Study 4 section 10).
- **Tuning separately for every arm** (as in version 2): Optuna TPE sampler seed 0, 20 trials (the number used by
  the committed version 2 runs), the 7 tuning points (first test meeting of 2001, 2005, ..., 2025), the same 4
  shared expanding folds, the same search space (hidden, dropout, learning rate, weight decay, lookback 2/4/8/12,
  batch 16/32/64, layers 1-2 or heads 1/2/4), at most 300 epochs with patience 20. Objective: mean validation
  log loss of the 3-class probabilities over the folds (replaces validation MSE); early stopping on the same
  quantity. Final-fit epochs: median best epoch.
- Walk-forward: refit every 8 meetings and at each tuning point, 8 seeds; the seeds' level probabilities are
  averaged, then converted to class probabilities and the paper's predicted class.
- Rate forecast (for squared errors): last rate plus class probabilities times the training rows' mean move per
  class, as for the ordered logit.
- **Primary tests:** per architecture, S+M+Co against S+M and S+M+Ho against S+M, one-sided Newey-West test
  (lags 8) that the mean log-loss difference is positive; Holm over the 6 (3 architectures x 2). Clark-West is
  not used for the networks: separately tuned networks are not nested.
- Reported without a rule: the same tests for S+M+C and S+M+H; squared-error Newey-West, DM-HLN and the
  moving-block bootstrap (blocks of 8, 10,000 draws, seed 0); accuracy, balanced accuracy, recall by class,
  confusion matrices, McNemar; scheduled meetings as the main subset, all meetings also reported.
- Placebo reading (Study 4 section 10): a news gain counts as coming from the tone only if the real tone beats its
  own placebo at p < 0.05 and the placebo does not beat S+M at p < 0.05.
- **Comparison with version 2 (descriptive):** CORAL networks against the version 2 MSE networks of the same arm,
  on squared rate errors and accuracy.

## 8. Checks that the ordinal learning is implemented correctly (before any real run)

All are automated tests in tests/coral/ and must pass before section 10 step 4.

1. **Against the authors' code.** The loss, the level encoding and the probability conversion give the same
   numbers as the coral-pytorch package (installed as a test-only dependency) on random inputs.
2. **Equivalence with the ordered logit.** A CORAL layer with no hidden layers, fitted on simulated data from a
   cumulative logit model, recovers the true slopes and thresholds, and its probabilities match
   models.linear.OrderedLogit (zero penalty) on the same data to within a small tolerance.
3. **Learns order on simulated data.** A small LSTM with the CORAL layer, trained on simulated sequences with a
   known ordinal target, beats the majority class and has ordered biases.
4. **Probabilities.** Every class probability is >= 0 and the three sum to 1; P(r >= 1) >= P(r >= 2) on all
   outputs of every trained model; b_1 >= b_2 is logged for every saved model and every network fit.
5. **No look-ahead.** The Study 4 truncation checks (studies/networks/check_no_lookahead.py) are repeated for the
   CORAL networks: forecasting a row with later rows removed gives the same forecast.
6. **Label order.** A test that dovish, neutral and hawkish map to ranks 0, 1, 2 and that the levels are
   (0, 0), (1, 0), (1, 1).

## 9. Wording fixed now

- A primary p < 0.05 after Holm: "in this exploratory study, the <reader> ordinal tone improved <model> forecasts
  of FOMC decisions (p = ..., Holm-adjusted)". Never "confirmed": the data were used before.
- Otherwise: "not significant at 5% in this exploratory study"; a non-significant result is not described as
  showing that news does not help.
- Any Haiku result carries the memory caveat of Study 4.

## 10. Order of work, compute and cost

1. This document is approved by the user, committed and pushed to GitHub (the push date is the timestamp).
2. Code and the section 8 tests.
3. Haiku token projection (section 5 rule), then the batch. ChronoBERT-CORAL fine-tuning runs on the GPU at the
   same time (about 8 hours on the RTX 2060, judging by the Study 4 run).
4. Ordered logit (section 6).
5. Networks (section 7): LSTM and GRU on the CPU, TFT on the GPU after ChronoBERT. One trial per architecture
   is timed first; if the projected wall time exceeds 36 hours, the user is asked before anything is cut.
6. Results and write-up (section 11).

## 11. Outputs

- results/coral/: tone files' summaries, holdout accuracy per year, prediction files, metrics, comparisons with
  all p-values, Holm-adjusted p-values, the b_1 >= b_2 log, the Haiku token and inconsistency counts.
- results/coral/ATTEMPTS.md: every method tried in the project so far (Plans 1-2, Studies 2-5, the version 2
  networks, the proof-of-concept series, this study), each with its primary test and p-value, so the number of
  attempts is visible.
- results/coral/RESULTS_CORAL.md: every number of this study next to the earlier ones.
- A Thai explanation for the user, written after the runs: for every model, what one input row looks like (the
  vector, its length and columns, and the sequence shape for the networks), how it is trained (loss, optimiser,
  epochs, early stopping, refits, seeds), and the hyperparameters chosen at each tuning point.

## 12. Left to the builder

Module and function names inside src/fed_news/coral/; how ChronoBERT-CORAL and the CORAL networks reuse the
existing code (import or copy, as long as the existing files are not changed); job scheduling between CPU and
GPU; file formats of intermediate outputs.

## 13. Amendment 1 (2026-10-20, before any code, training run or API call)

The user asked whether the section 5 prompt is unnatural. It is: "the probability that the sentence is not
dovish" is an odd question for a language model and invites B > A. The user agreed to replace it. Section 5's
prompt and parsing are replaced by the following; everything else in section 5 stays.

- System prompt: the Study 4 definitions of H, D and N, unchanged, followed by:
  "For each sentence give three whole numbers from 0 to 100 that add up to 100: the probabilities that the
  sentence is dovish (D), neutral (N) and hawkish (H). Judge only the text of each sentence. Reply with one JSON
  object mapping each sentence number to [D, N, H], and nothing else."
- Parsing: three numbers in [0, 100] per sentence. If they do not add up to 100 but their sum is positive, they
  are divided by their sum (the count is reported). A reply with a missing sentence, a negative number or a zero
  sum fails the request (resent once, as before).
- Levels in CORAL form: P(r >= 1) = (N + H) / 100, P(r >= 2) = H / 100. These are always ordered, so the
  section 5 inconsistency count no longer applies. Sentence tone = H/100 - D/100 = P(hawkish) - P(dovish).
- Descriptive check: the share of sentences where the most probable class equals the Study 4 label (ties count
  as a mismatch).
- The output projection stays at 15 tokens per sentence and the limit at 35,000,000 tokens.

## 14. Amendment 2 (2026-10-20, before any training run or API call; found by the code review)

**Problem.** With CORAL the two logits of a row differ only by b_1 - b_2, so P(middle class) can never exceed
s((b_1 - b_2)/2) - s(-(b_1 - b_2)/2). With the section 2 start values (1.0, 0.5) that cap is 0.124. ChronoBERT
fine-tuning (learning rate 2e-5, at most about 456 AdamW steps per classifier year) can move a bias by at most about
0.009, so the gap would stay at 0.5 and P(neutral) would stay below 0.124 for every sentence, although neutral is
the largest label class (44.8% of the human labels). Networks tuned at small learning rates would meet the same cap
for "hold" (about two thirds of the meetings).

**Change (replaces the bias start values of section 2; everything else stays):**

- Biases start at the logits of the training rows' cumulative class shares, b_k = logit(share of rows with
  rank >= k), each share clipped to [0.001, 0.999]: the intercept-only ordered fit, the same start values that
  models.linear.OrderedLogit uses. This applies to ChronoBERT-CORAL (the year's training labels) and to the CORAL
  networks (the rows of each fit: every tuning fold and every walk-forward refit).
- The CORAL biases are not weight-decayed (the thresholds are not penalised, as in OrderedLogit).
- ChronoBERT-CORAL only: the two biases get their own learning rate, 1e-2, so the thresholds can be learned within
  the fine-tuning budget; every other parameter keeps 2e-5. The networks keep their tuned learning rate for all
  parameters.
- The section 8 check 1 (agreement with coral-pytorch) still covers the loss, levels and prediction rule; the head's
  start values now differ from the package's by design.

## 15. Amendment 3 (2026-10-21): test from 2010. Declared after the full-sample results, before any result by period

**Why.** The user's hypothesis: ChronoBERT classifiers of the early classifier years are too weak to carry the
signal, so the early meetings dilute any news gain. The evidence for weak early classifiers comes from the labels,
not from forecast results: the out-of-time hawkish-vs-dovish AUC of the Study 4 classifiers was 0.648 for
2000-2004 (2001: 0.526, 2003: 0.497) and 0.75-0.84 later; the ChronoBERT-CORAL holdout accuracy was 0.50-0.59 for
classifier years 2000-2013. The user chose the start 2010-01-01.

**Known cost.** From 2010-01-01 the test has 135 meetings (of 212) and 32 rate moves (11 cuts, 21 hikes; of 72),
because 2009-2015 was almost all holds at the zero lower bound. Fewer moves means less power.

**What is tested.** This is a further attempt. It does not replace sections 6 and 7: their full-sample results
stand and are reported first.

- Nothing is refit. The saved walk-forward forecasts (each made only from data before its meeting) are scored on
  the test meetings dated 2010-01-01 or later. This equals training up to the end of 2009 and forecasting from 2010
  with the same walk-forward. Every version must have the same meetings in the subset.
- Ordered logit: the 7 versions of section 6. Primary: one-sided Clark-West, S+M+Co and S+M+Ho against S+M, Holm
  over the 2. The void rule, the wording of section 9, the other statistics and the Haiku memory flag are
  unchanged, all computed on the subset.
- Networks: the section 7 primary tests (Newey-West, lags 8, on the log-loss difference; Holm over the 6) and the
  placebo reading, on the scheduled meetings dated 2010-01-01 or later; all meetings also reported.
- Outputs: results/coral/logit_from2010/ and results/coral/networks_from2010/.
- Disclosure: the start year was chosen after the full-sample p-values were known, but before any result by
  period was computed or seen. The attempt is listed in ATTEMPTS.md with its result, whatever it is. Any other
  start year would be another attempt, declared in this document before it is computed.
