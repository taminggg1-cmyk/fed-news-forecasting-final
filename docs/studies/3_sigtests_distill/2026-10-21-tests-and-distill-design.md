# Literature significance tests (Part A) and Haiku-distilled ChronoBERT (Part B)

Date: 2026-10-21. Branch `coral`. Written, committed and pushed before any result in this document is computed.

## 0. Status and disclosure

Both parts were designed after every CORAL result was known (results/coral/, commit e520297). In those results:

- The ordered logit with the Haiku ordinal tone, on test meetings from 2010, had Clark-West p = 0.0498 (Holm over 2: 0.0997).
- The best network test was LSTM + Haiku ordinal: p = 0.024, Holm over 6: 0.145.
- The ChronoBERT-CORAL tone did no better than its placebo.

- **Part A** is exploratory. It applies tests that other researchers use to the forecasts already saved. It changes no registered verdict.
- **Part B** is a new declared attempt (ATTEMPTS.md). Its single primary test was chosen with the knowledge above, and that is said wherever its result is reported.

Code: `src/fed_news/sigtests/` (Part A) and `src/fed_news/distill/` (Part B). Results: `results/sigtests/` and `results/distill/`.

## Part A: tests other researchers use (exploratory)

### A0. Inputs and conventions

Saved forecasts:
- **Ordered logit:** results/coral/logit/preds_T2_*.csv, plus the Study 4 S+M+H and S+M+C forecasts (results/study4/), which are on the same meetings.
- **Networks:** results/coral/networks/preds_<arch>_<arm>.csv, scheduled meetings.

Each test is run twice: on the full test period, and on test meetings from 2010-01-01.

Every news version is compared with S+M from the same forecaster. Where a test is one-sided, the alternative is "the news version is better". Loss differences are base minus news, so a positive difference favours news.

### A1. Fixed-smoothing Diebold-Mariano (Coroneo and Iacone 2020, Journal of Applied Econometrics)

These are small-sample-correct versions of the DM test. Each is applied to three losses: log score, squared rate error, and ranked probability score (A3).

- **Fixed-b:**
  - Bartlett kernel, bandwidth M = floor(T^(1/2)).
  - The p-value comes from the Kiefer-Vogelsang fixed-b null distribution of the t statistic. It is simulated as the same statistic computed on 100,000 standard normal i.i.d. series of length T (seed 0).
- **Fixed-m:**
  - Daniell weighted periodogram estimate of the long-run variance, with m = floor(T^(1/3)) Fourier frequencies.
  - The statistic is compared with a t distribution with 2m degrees of freedom.

### A2. Newey-West DM with 8 lags on every loss

This is the same statistic as the registered network test. Here it is applied to all three losses and to the logit forecasts as well.

### A3. Ranked probability score (Epstein 1969)

RPS_t = Σ_{k=1}^{2} (F_k − O_k)², where F_k is the forecast cumulative probability of classes ≤ k and O_k is the observed indicator.

RPS is the proper score for ordered classes: a forecast that puts mass on "hike" when the outcome is "cut" is penalised more than one that puts mass on "hold". It is tested with A1 and A2.

### A4. Clark-West for the networks

The networks' rate forecasts (lag plus expected move) are nested-model comparisons in spirit, like the logit's. This applies Clark-West to them in two forms: the plain variance used for the logit, and Newey-West with 8 lags.

### A5. Giacomini-White conditional predictive ability (Econometrica 2006)

- Applied to the log-score difference d_t, with instruments h_t = (1, d_{t−1}).
- Wald statistic: T · z̄' Ω⁻¹ z̄, where z_t = h_t d_t and Ω is Newey-West with 8 lags. Compared with χ²(2).
- The test is two-sided by construction. It is reported with the sign of the mean of d.

### A6. In-sample tests (Inoue and Kilian 2004, Econometric Reviews; ordered-probit studies such as Hayo and Neuenkirch)

- Model: unpenalised ordered logit (statsmodels OrderedModel) on all meetings of the common panel.
- Comparisons: S+M against S+M plus the two tone columns (n_stance, n_stance_chg), for each tone (Co, Ho; H and C for reference).
- Tests reported:
  - likelihood ratio, χ²(2);
  - Wald test of the two tone coefficients with a Newey-West (8 lags) sandwich covariance;
  - one-sided z on the n_stance coefficient (hawkish tone, higher class).
- This tests in-sample association, not out-of-sample forecasting gain. In-sample, Haiku's possible memory of outcomes matters even more than out of sample.

### A7. Joint tests over the family (Hansen 2005, SPA; Romano and Wolf 2005, StepM; Hansen, Lunde and Nason 2011, MCS)

- Implementation: the `arch` package (8.0.0).
- Bootstrap: stationary bootstrap, mean block length 8, 10,000 replications, seed 0.
- Loss: log score. Benchmark: S+M.
- Families:
  - the 6 registered network comparisons, as a whole and per tone;
  - the 2 logit tones.
- Output: the SPA p-value (consistent version); the StepM rejections at 5%; the 90% MCS members, with each forecaster's S+M and news versions as candidates.

### A8. Listed, not run

The Pesaran-Timmermann test of direction (1992; multi-category 2009) tests whether one forecast has skill. It does not compare two forecasts, so it does not answer our question.

### A9. Reporting

- results/sigtests/ gets every p-value for every comparison, both periods.
- The summary lists which p-values are below 0.05. It states that choosing a test after seeing its result is not confirmatory evidence, and that the registered verdicts are unchanged.

## Part B: Haiku-distilled ChronoBERT ("Hd"), a new declared attempt

### B0. Idea

This is the user's idea:
1. Use Haiku's sentence probabilities as soft labels to train ChronoBERT on 1994-2009 text.
2. Freeze the model at the end of 2009.
3. Let it alone read the text of 2010 onward.

Why this limits the memory leak:
- Haiku may remember what the Fed did. That memory can affect only the training labels, which are all dated before the test period.
- The tone for test meetings comes from a ChronoBERT whose pretraining stops at 2009 and which never sees a Haiku label from 2010 onward.

Why it gives more data: there are 144,247 sentences before 2010 with Haiku probabilities, against 1,422 human-labelled sentences before 2010.

### B1. Data

- **Sentences:** <data>/coral/haiku_sentence_probs.pkl (343,742 sentences).
  - The 100 sentences without probabilities are left out of training but still scored.
  - A sentence's date is its document's release date (from doc_id).
- **Input text:** the unmasked sentence (lower case), as in the CORAL study. **Targets:** Haiku's (P_dovish, P_neutral, P_hawkish).

### B2. Model and loss

- Base model: `manelalab/chrono-bert-v1-20091231` with the CORAL head of the CORAL study (mean token vector, one shared weight, two biases).
- Soft level targets: (P_N + P_H, P_H).
- Loss: the CORAL loss with these soft targets, i.e. the sum of two binary cross-entropies.
- Biases start at the logits of the mean soft levels of the training rows. They train at learning rate 1e-2, without weight decay.
- Other settings: AdamW (weight decay 0.01), batch 16, max length 256, fp16 autocast, gradient clipping 1.0, CUDA only.

### B3. Tuning (pre-2010 data only)

- **Tuning-train:** a random 50% (seed 0) of the sentences from documents released 1994-2006.
- **Validation:** 6,000 random sentences (seed 0) from documents released 2007-2009.
- **Grid:** learning rate ∈ {1e-5, 2e-5, 4e-5} × epochs ∈ {1, 2}. Epoch 2 continues the epoch-1 run, and is evaluated after each epoch.
- **Criterion:** mean soft cross-entropy −Σ_c q_c log p_c between Haiku's class probabilities q and the model's class probabilities p on the validation sentences.
- The lowest wins. Ties within 1e-4 go to fewer epochs, then the lower learning rate.
- Seed 0.

### B4. Final models (seed 0, tuned settings, all saved under D:/fed_news/distill_models/)

- **`final`:** trained on every sentence before 2010-01-01. Scores every sentence of documents released 2010-01-01 onward.
- **`foldA` and `foldB`:** cross-fitting for the training-period tone.
  - Fold A = documents 1994-2001, fold B = documents 2002-2009.
  - The model trained on B scores A, and the model trained on A scores B.
  - So no tone value fed to a forecaster comes from a model that trained on that sentence.
- **Single seed:** chosen for the compute budget (RTX 2060, about 90 training sentences per second). With about 144,000 training sentences, seed averaging matters less than with the 1-3 thousand of the CORAL study.

### B5. Tone and panel

- Sentence tone = P(hawkish) − P(dovish), from the CORAL class probabilities.
- Document tone = the mean over the document's sentences, in the doc_stance format of the CORAL study.
- Panel: naive_panel("t2", ·), restricted to the common meetings of the H, Co and Ho panels, i.e. the CORAL meetings.

### B6. Forecasts

Ordered logit, with the same engine and settings as the CORAL study: walk-forward from 2001, penalty by cross-validation inside the training rows. Four versions:
- S+M, which must equal the CORAL S+M forecasts;
- S+M+Hd;
- placebo-Hd;
- shuffled-Hd.

### B7. Primary test: one test, no multiple-testing correction

- **Test:** one-sided Clark-West, S+M+Hd against S+M, on test meetings from 2010-01-01.
- **Supported** if p < 0.05 and the result is not void.
- **Void** if placebo-Hd beats S+M at Clark-West p < 0.05, or shuffled-Hd does so at p < 0.01.
- **Wording:**
  - If supported: "in this exploratory attempt, the Haiku-distilled ChronoBERT tone improved ordered-logit forecasts of FOMC decisions from 2010 (p = …)".
  - Otherwise: "not significant at 5% (p = …); this does not show that the tone does not help".
- **Reported with no rule attached:**
  - every Part A test;
  - accuracy and log loss;
  - the 2001+ period. For 2001-2009, the tone comes from cross-fitted 2009-vintage models, so it has look-ahead in pretraining and in labels.

### B8. Secondary (if time allows, after the logit; the logit verdict does not wait for it)

- Models: CORAL LSTM, GRU and TFT with the arms S+M+Hd and placebo-Hd, under the CORAL tuning protocol.
- Test: Newey-West with 8 lags on log loss, scheduled meetings from 2010, Holm over 3. Plus the placebo reading.

### B9. Fidelity, descriptive

On sentences from 2010 onward, the report gives:
- the correlation of the model tone with the Haiku tone;
- the soft cross-entropy;
- the agreement of the most likely class.

Haiku labels from 2010 onward are used only for this check, never for training or forecasting.

### B10. Order of work

1. Push this document.
2. Write the Part B code with tests, then run it detached (tuning, final and folds, tone, logit), with a watcher.
3. Write the Part A code with tests and run it on the saved forecasts.
4. Run Part A on the Part B forecasts too.
5. Write up everything in ATTEMPTS.md.
