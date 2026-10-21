# Technical note: from Fed text to FOMC-decision forecasts

Exact mechanics of the pipeline in the `coral` branch, with tensor shapes at every hand-off. Each point names the
code that implements it. Notation: B = batch size, T = tokens per sentence, L = lookback (meetings),
H = hidden size, N = number of meetings.

**The most important fact first.** ChronoBERT's 768-dimensional embeddings are **not** passed to the forecasting
models. Each sentence is reduced to **one scalar tone** inside the text stage, and each meeting gets **two scalars**:

```
[N_sent, 768] -> CORAL head -> [N_sent, 2] -> tone -> [N_sent] -> averaging -> [N_meetings, 2]
```

The downstream models never see an embedding.

---

## 1. Input representation

There are two separate input spaces: text, measured in tokens, and meetings, measured in time steps.

### 1a. Text stage (per token)

**Corpus.** Fed documents of 4 types: FOMC statements, press-conference transcripts, minutes and speeches.
- A document needs a release date and at least 500 characters.
- Each document is split into sentences of 30 to 1,000 characters and lower-cased.
- Total: 343,742 sentences, 1994-2026 (`download/fed_documents.py`, `features/chronobert_tone._sentence_table`).

**Tokenisation.** The ModernBERT tokenizer of the ChronoBERT checkpoint (`AutoTokenizer`, `coral/chronobert_coral._batches`):
- Truncation at 256 tokens (`config.STANCE_MAXLEN`).
- Padding to the longest sentence in the mini-batch.

```
B sentences (strings) -> tokenizer -> input_ids [B, T] (int64), attention_mask [B, T] (0/1),  T <= 256
```

There is no normalisation beyond lower-casing. Numbers and dates stay in the ChronoBERT input. They are masked
only in the copy sent to Haiku: `[MONTH] [NUM] [YEAR]`.

### 1b. Forecasting stage (per time step = one FOMC meeting)

One row per meeting j. The meeting panel has N = 260 meetings, 1995-03-28 to 2026-09-16; the test meetings are the
212 from 2001-01-03 on.

The row x_j is built from information dated **before the cutoff c_j = the day before the decision**
(`features/panels.py`). It is a 26-vector (24 without news):

| Block | Dim | Columns | Timing |
|---|---|---|---|
| S, economic | 16 | s_rate_lag, s_m2_mom, s_m2_yoy, s_pce_yoy, s_pce_core_yoy, s_payems_chg, s_earn_yoy, s_gdp_yoy, s_u6_mom, s_u6_yoy, s_ppi_yoy, s_cpi_yoy, s_cpi_core_yoy, s_hpi_yoy, s_slope_10y2y, s_slope_1y6m | as published before c_j (ALFRED real-time vintages) |
| M, market | 8 | m_sp_tbill_3m, m_sp_y6m, m_sp_y1, m_sp_y2 (each a yield minus the target rate), m_target_level, m_zero_bound (1 if target ≤ 0.30), m_vix, m_wti | last value before c_j |
| D, text | 2 | n_stance, n_stance_chg | documents released in [c_{j-1}, c_j) |

**Target.** y_j ∈ {0 cut, 1 hold, 2 hike}, the class of the change in the target rate at decision j
(±0.125-point band).

**Normalisation.** z-scores computed on the training rows only, refreshed at every refit:

$$z_{j,k} = \frac{x_{j,k} - \mu_k^{\text{train}}}{\sigma_k^{\text{train}}}$$

(`models/neural.standardise`, `models/walk_forward._standardise`.)

---

## 2. ChronoBERT feature extraction and pooling

### 2a. Encoder

- **Model:** `manelalab/chrono-bert-v1-<year>1231`, ModernBERT architecture: 22 layers, hidden size 768, about 149 M
  parameters. Its pre-training text stops at the end of `<year>`.
- **Layer used:** only the **final layer's output** (`last_hidden_state` of `AutoModel`). No intermediate layers, no
  layer mixing.
- **Fine-tuning:** the encoder is fine-tuned end to end, not frozen.

```
input_ids [B, T] -> ModernBERT (22 layers) -> E = last_hidden_state [B, T, 768]
```

### 2b. Pooling: masked mean over tokens (no [CLS])

$$h_b = \frac{\sum_{t=1}^{T} m_{b,t}\, E_{b,t,:}}{\max\!\left(1, \sum_{t} m_{b,t}\right)} \in \mathbb{R}^{768}$$

- m is the attention mask, so padding is excluded. The special tokens [CLS]/[SEP] have m = 1 and are included.
- Code: `coral/chronobert_coral.StanceNetCoral.forward`.

```
E [B, T, 768], mask [B, T] -> masked mean over T -> h [B, 768]
```

### 2c. CORAL head: from vector to probabilities

**Head** (`coral/core.CoralHead`):
- one shared weight vector w ∈ ℝ^768 with no bias;
- K − 1 = 2 threshold biases b = (b₁, b₂);
- ranks: dovish 0 < neutral 1 < hawkish 2.

$$s_b = w^\top h_b, \qquad z_{b,k} = s_b + b_k, \qquad P_{b,k} = \sigma(z_{b,k}) = P(\text{rank}_b \ge k), \quad k = 1, 2$$

```
h [B, 768] -> w^T h -> s [B, 1] -> + b (broadcast) -> z [B, 2] -> sigmoid -> P [B, 2]
```

**Class probabilities and tone** (`core.class_probs`, `core.tone_from_levels`):

$$p_D = 1 - P_1,\quad p_N = P_1 - P_2,\quad p_H = P_2, \qquad \text{tone}_b = p_H - p_D \in [-1, 1]$$

```
P [B, 2] -> [B, 3] -> tone [B]
```

Since b₁ ≥ b₂ (checked for every model), P₁ ≥ P₂ for every input. This is CORAL's rank consistency.

**Worked example** (final distilled model, b = (1.54, −1.99)).
- Sentence: "job gains have been strong in recent months, and the unemployment rate has declined substantially."
  (statement, 16 Mar 2022).
- s = 2.40, so z = (3.94, 0.41) and P = (0.98, 0.60).
- p = (0.02, 0.38, 0.60), tone = +0.58.

### 2d. The three text readers that use this pipeline

| Reader | Checkpoint(s) | Trained on | Seeds |
|---|---|---|---|
| ChronoBERT (Study 4) | vintage y−1 for documents of year y (1994-1999 use y = 2000; vintages clipped to 1999-2024) | 3,074 human labels dated before y; **softmax** head on the same pooled h, giving [B, 3] | 3 |
| ChronoBERT-CORAL | same vintages | same labels; **CORAL** head (above) | 3, mean of P |
| Distilled ChronoBERT (Hd) | vintage 2009 only | Haiku's probabilities for 144,211 sentences before 2010 as soft CORAL targets | 1 |

Haiku readers (H, Ho) produce p_D, p_N, p_H through the API with no embeddings. The tone formula and every step
below are the same for all readers.

**Which model scores which sentence (no model scores its own training data):**
- ChronoBERT and ChronoBERT-CORAL: year y's model scores year y's sentences and trains only on labels dated before y.
- Distilled ChronoBERT:
  - sentences from 2010 on: the final model (trained on 1994-2009);
  - 2002-2009: the model trained on 1994-2001;
  - 1994-2001: the model trained on 2002-2009.

### 2e. From sentences to meetings: averaging over time intervals

**1. Document tone** (`coral/chronobert_coral.run`, `distill/chronobert_distill.doc_tone`):

$$\tau_d = \frac{1}{n_d}\sum_{i \in d} \text{tone}_i$$

```
[n_d] -> mean -> scalar
```

**2. Meeting tone over the window** W_j = [c_{j−1}, c_j) (`features/news_columns.fed_stance`, `features/swap_tone.naive_panel`):
- Documents are grouped g ∈ {policy = statements + press conferences, minutes, speeches}.
- Average per group, then average the groups present. Every group counts equally, whatever its number of documents.

$$\bar\tau_{j,g} = \operatorname{mean}_{d \in W_j \cap g}\, \tau_d, \qquad n\_stance_j = \operatorname{mean}_{g\ \text{present}}\ \bar\tau_{j,g}$$

**3. Empty window:** carried forward, n_stance_j = n_stance_{j−1}.

**4. Change:** n_stance_chg_j = n_stance_j − n_stance_{j−1}.

```
all sentences in W_j -> tones -> document means -> group means -> mean -> n_stance_j (scalar)
D_j = [n_stance_j, n_stance_chg_j] in R^2
```

Typical values: document tones are about −0.05 to +0.05, because most sentences are neutral and hawkish and dovish
sentences partly cancel. Example: 4 May 2022, D = [+0.028, +0.029].

---

## 3. Lookback window and temporal alignment

### 3a. Sequence construction (`models/neural.make_sequences`)

For meeting j and lookback L, the input is the L consecutive meeting rows **ending at j itself**:

$$\mathbf{X}_j = [\, z_{j-L+1},\ \ldots,\ z_{j-1},\ z_j \,] \in \mathbb{R}^{L \times 26}$$

```
Z [N, 26] -> stack windows -> X [n_rows, L, 26]
```

- **Steps are meetings, not days:** about 8 per year, so L = 4 is about 6 months.
- **Search space:** L ∈ {2, 4, 8, 12}, tuned at each tuning point.
- **Values chosen for the Haiku-ordinal arm:**

  | Tuning year | 2001 | 2005 | 2009 | 2013 | 2017 | 2021 | 2025 |
  |---|---|---|---|---|---|---|---|
  | LSTM | 4 | 4 | 4 | 2 | 2 | 2 | 8 |
  | GRU | 4 | 4 | 4 | 4 | 4 | 4 | 2 |
  | TFT | 4 | 8 | 4 | 4 | 2 | 4 | 2 |

  The full table is in results/coral/networks_hyperparameters.csv.
- **Training rows:** only rows with j ≥ L − 1, so every window is full.

The ordered logit uses L = 1 (the row z_j alone).

### 3b. No look-ahead: every guarantee

1. **Feature timing:** every column of z_j is dated before c_j, the day before decision j.
   - economic data as published (vintage) before c_j;
   - market data: the last value before c_j;
   - text: documents released before c_j.
2. **Training rows:** to forecast meeting i, a model trains only on rows j < i (`walk_forward.train_mask`: `idx < i`).
   Those decisions are already known.
3. **Normalisation:** μ and σ come from the training rows only.
4. **Refits** (`coral/run_networks.run_tuned_coral`):
   - The network is refit every 8 test meetings, and at each new tuning point, on rows < i.
   - The next 7 meetings are forecast without refitting.
   - The ordered logit is refit at every meeting.
5. **Hyperparameters:** chosen at tuning points (the first test meeting of 2001, 2005, …, 2025) using only rows before
   the point.
6. **Text models:**
   - pre-training vintage before the document year;
   - human labels dated by the first document that contains the sentence, and used only if dated before the
     classifier year;
   - distilled model frozen at the end of 2009, with cross-fitted scores before 2010.
7. **The one admitted leak:** Haiku itself (H, Ho) may remember outcomes. The distilled model exists to remove it
   for the period from 2010.

---

## 4. Downstream integration (feature-level fusion)

**Fusion type: early fusion by plain concatenation.**
- The 2 text scalars are appended to the 24 numeric columns.
- No projection layer, no separate text branch, no late fusion.

$$x_j = [\, \underbrace{s_j}_{16} \,\|\, \underbrace{m_j}_{8} \,\|\, \underbrace{D_j}_{2} \,] \in \mathbb{R}^{26}$$

**Full chain** (B = mini-batch of meetings):

```
text:   [N_sent, T] -> ChronoBERT -> [N_sent, T, 768] -> masked mean -> [N_sent, 768] -> CORAL head -> [N_sent, 2]
        -> tone -> [N_sent] -> document/group/window means -> [N, 2]
panel:  concat([N, 16], [N, 8], [N, 2]) -> [N, 26] -> z-score (training rows) -> Z [N, 26]
seq:    Z -> windows -> X [B, L, 26]
```

### 4a. LSTM / GRU (`models/neural.RNNRegressor` with `coral/networks.build_coral_model`)

```
X [B, L, 26] -> LSTM/GRU(input 26, hidden H, layers l in {1,2}, dropout between layers) -> O [B, L, H]
-> last step O[:, -1, :] -> [B, H] -> CORAL head (w in R^H, b in R^2) -> z [B, 2] -> sigmoid -> P [B, 2] -> p [B, 3]
```

H ∈ {8, 16, 32, 64, 128, 256}.

### 4b. TFT (`models/tft.CompactTFT`, CORAL head)

```
X [B, L, 26]
-> variable selection:
     per-variable GRN       -> [B, L, 26, H]
     softmax weights (GRN)  -> [B, L, 26, 1]
     weighted sum over vars -> [B, L, H]
-> LSTM(H)                                              -> [B, L, H]
-> GLU gate + residual + LayerNorm                      -> [B, L, H]
-> interpretable multi-head attention:
     n_heads in {1,2,4}, shared value projection, causal mask (triu)
                                                        -> [B, L, H]
-> GLU + residual + LayerNorm                           -> [B, L, H]
-> position-wise GRN, GLU + residual + LayerNorm        -> [B, L, H]
-> last step [B, H] -> CORAL head (decision ranks cut 0 < hold 1 < hike 2) -> [B, 2] -> [B, 3]
```

H ∈ {8, 16, 32, 64}. The variable-selection weights are the only place where the 2 text columns get their own learned
weighting.

### 4c. Ordered logit (`models/linear.OrderedLogit`)

$$\eta_j = \beta^\top z_j,\quad t_1 = a,\ t_2 = a + e^{b}\ (t_2 > t_1),\quad P(\text{cut}) = \sigma(t_1 - \eta_j),\ P(\le\text{hold}) = \sigma(t_2 - \eta_j)$$

```
z_j [26] -> beta^T z_j [1] -> 2 thresholds -> [3]
```

### 4d. Rate forecast (used by Clark-West)

$$\hat r_j = r_{j}^{\text{before}} + \hat p_{\text{cut}} \cdot \bar\Delta_{\text{cut}} + \hat p_{\text{hike}} \cdot \bar\Delta_{\text{hike}}$$

Δ̄ is the mean size of past cuts and hikes in the training rows.

**Networks: seed ensemble.** 8 seeds; their level probabilities P are averaged, then converted to p.

---

## 5. Hyperparameters, search and losses

### 5a. Text stage

| Model | Hyperparameters | Search | Loss | Selection metric |
|---|---|---|---|---|
| ChronoBERT-CORAL | lr 2e-5 (encoder, w), biases lr 1e-2 with no weight decay, biases start at logit of cumulative label shares; 3 epochs; batch 16; max length 256; AdamW weight decay 0.01; fp16; grad clip 1.0 | **none**: fixed at the Study 4 values, as the design required | CORAL loss (below) | — (holdout reported: accuracy 58.9%, macro F1 0.514) |
| Distilled ChronoBERT | lr ∈ {1e-5, 2e-5, 4e-5} × epochs ∈ {1, 2}; the rest as above | **grid** of 6, time-split: train on 50% of 1994-2006 sentences, validate on 6,000 sentences from 2007-2009 | CORAL loss with soft targets (P_N + P_H, P_H) from Haiku | soft cross-entropy −Σ_c q_c log p_c, lowest wins. Chosen: lr 2e-5, 2 epochs (0.8331); lr 4e-5 diverged at epoch 2 |

**CORAL loss** (per sentence or meeting, averaged over the batch; `coral/core.coral_loss`), with level targets
y_k = 1{rank ≥ k}, or soft targets ∈ [0, 1] for the distilled model:

$$\mathcal{L} = -\sum_{k=1}^{2}\Big[y_k \log\sigma(z_k) + (1-y_k)\log\big(1-\sigma(z_k)\big)\Big]$$

It is tested equal to the reference package coral-pytorch (`tests/coral/test_core.py`).

### 5b. Forecasting stage

**Ordered logit**
- **Tuned:** the L2 penalty λ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}.
- **When:** at **every** forecast.
- **How:** 5 expanding-window folds inside the training rows. The rows are cut into 6 time blocks; fold k trains on
  blocks 1..k and validates on block k+1.
- **Selection metric:** mean validation log loss.
- **Final fit:** L-BFGS on

$$\frac{1}{n}\sum_j -\log \hat p_{j,y_j} + \frac{\lambda}{2}\lVert\beta\rVert^2 \quad \text{(thresholds not penalised)}$$

**LSTM / GRU / TFT**
- **Search:** Optuna TPE (seed 0), **20 trials** at each of 7 tuning points, separately for every network × input set
  (147 tuning runs in the CORAL study).
- **Search space** (`studies/networks/run_networks.suggest`):

  | Hyperparameter | Range |
  |---|---|
  | hidden | {8, 16, 32, 64, 128, 256}; TFT {8, 16, 32, 64} |
  | layers / heads | LSTM/GRU layers {1, 2}; TFT heads {1, 2, 4} |
  | dropout | U(0, 0.5) |
  | lr | log-U(1e-4, 1e-2) |
  | weight decay | log-U(1e-6, 1e-2) |
  | lookback L | {2, 4, 8, 12} |
  | batch | {16, 32, 64} |

- **Cross-validation:**
  - The eligible rows before the tuning point are cut into 5 time blocks; 4 expanding folds (train on all earlier
    blocks, validate on the next).
  - The folds are the same for every candidate.
  - A candidate with fewer than 10 training sequences in any fold scores ∞.
- **Early stopping inside each fold:** patience 20, at most 300 epochs, on the validation log loss of the 3 classes.
- **Trial objective:**

$$\text{objective} = \frac{1}{4}\sum_{f=1}^{4} \text{val log loss}_f, \qquad \text{epochs} = \operatorname{median}_f(\text{best epoch}_f)$$

- **Final fit at each refit:** Adam on the CORAL loss for the tuned number of epochs, with the tuned lr and batch.
  Weight decay applies to every weight except the CORAL biases, whose start values come from the cumulative class
  shares. Shuffled mini-batches, 8 seeds.

### 5c. Evaluation metrics (not used for tuning)

Every model is compared with the same model without text, on the same meetings.

| Forecaster | Primary test | Correction |
|---|---|---|
| Ordered logit | one-sided Clark-West on the rate forecast | Holm over the tones of the study |
| Networks | one-sided Newey-West (8 lags, the refit block) test on the per-meeting log-loss difference | Holm over 6 (3 networks × 2 tones) |

Also reported:
- accuracy, macro F1, log loss, Brier score, RMSE;
- DM-HLN and McNemar;
- placebo (tone shifted or shuffled) and shuffled-label controls;
- the literature battery in results/sigtests/.
