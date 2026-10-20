# CORAL study implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the CORAL study: ChronoBERT-CORAL and Haiku ordinal tones, the ordered logit on the new
tones, and CORAL LSTM/GRU/TFT, with every result written to results/coral/.

**Architecture:** One new package `src/fed_news/coral/` that imports (never edits) the existing modules. A core
module holds the CORAL layer, loss and probability conversions; the readers, the logit runner and the network
runner build on it. Long jobs (GPU fine-tuning, Haiku batch, Optuna) are resumable command-line steps.

**Tech Stack:** Python 3.13, torch 2.11, transformers 5.17, optuna 4.5, anthropic 1.13, pytest; coral-pytorch 1.4.0
as a test-only oracle. Interpreter: `D:/gold_lm/venv/Scripts/python.exe` (run from `D:/fed_news_final`).

**Spec:** `docs/design/2026-10-20-coral-design.md` (sections 0-13; section 13 replaces the section 5 prompt).

## Global Constraints

- No existing file under `src/`, `tests/` or `results/` is modified; only `pyproject.toml` gains `coral-pytorch==1.4.0` in the dev group.
- New code in `src/fed_news/coral/`, tests in `tests/coral/`, results in `results/coral/`, data in `config.DATA / "coral"`, models in `config.DATA / "chronobert_coral_models"`.
- K = 3, levels y_k = 1{r >= k}: rank 0 = (0, 0), 1 = (1, 0), 2 = (1, 1). Bias init (1.0, 0.5). Equal task weights.
- Tone ranks: dovish 0, neutral 1, hawkish 2. Stored label codes are dovish 0, hawkish 1, neutral 2 (must be remapped).
- Decision ranks: cut 0, hold 1, hike 2 (`y_class` already uses this order).
- Predicted class = number of level probabilities > 0.5 (paper rule).
- Haiku: `claude-haiku-5-5`, thinking disabled, effort low, max_tokens 2048, 25 sentences per request, seed 0, Batches API; projection 15 output tokens per sentence; stop if projected total > 35,000,000.
- Networks: 20 Optuna trials (TPE seed 0), tuning years 2001, 2005, ..., 2025, 4 shared folds, max 300 epochs, patience 20, refit every 8 meetings, seeds 0-7.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A stored label code mapped to the wrong rank (neutral treated as hawkish) silently inverts the tone: a test pins the remap.
- A trained model with b_1 < b_2 gives a negative P(neutral)/P(hold): probabilities are clipped at 0, renormalised, and the model is logged.
- Haiku replies with numbers as strings, floats, or sums of 99/101: accepted and normalised; a missing sentence, a negative number or a zero sum fails the request.
- The 8-hour ChronoBERT run is interrupted: finished years are reused only if their fingerprint matches.
- The new tone panels have different rows from the Study 4 panels: S+M is rerun on the common rows and the row difference is reported.

---

### Task 1: CORAL core

**Files:**
- Create: `src/fed_news/coral/__init__.py`, `src/fed_news/coral/core.py`
- Modify: `pyproject.toml` (dev group: `coral-pytorch==1.4.0`)
- Test: `tests/coral/test_core.py`

**Interfaces:**
- Produces:
  - `K = 3`
  - `class CoralHead(nn.Module)`: `__init__(self, size_in: int, n_classes: int = K)`; `forward(h) -> Tensor (n, K-1)` logits; attributes `weight: nn.Linear(size_in, 1, bias=False)`, `bias: nn.Parameter` init `(1.0, 0.5)`.
  - `levels(ranks) -> Tensor (n, K-1)` float.
  - `coral_loss(logits, levels, reduction="mean") -> Tensor`.
  - `level_probs(logits) -> np.ndarray (n, K-1)` (sigmoid).
  - `class_probs(lp: np.ndarray) -> np.ndarray (n, K)`: `[1-lp1, lp1-lp2, lp2]`, clipped at 0 and renormalised.
  - `predict_rank(lp) -> np.ndarray[int]`: count of `lp > 0.5`.
  - `biases_ordered(head) -> bool`.
  - `TONE_RANK = {0: 0, 2: 1, 1: 2}` (stored code -> rank).
  - `tone_from_levels(lp) -> np.ndarray`: P(hawkish) - P(dovish) = `lp2 - (1 - lp1)`.

- [ ] Step 1: Write failing tests:
  - `test_loss_levels_and_labels_match_coral_pytorch`: random logits (seed 0, 50x2) and ranks; `coral_loss` equals `coral_pytorch.losses.coral_loss` (atol 1e-6); `levels` equals `levels_from_labelbatch(ranks, 3)`; `predict_rank` equals `proba_to_label`.
  - `test_head_matches_coral_layer_and_initial_biases`: `CoralHead(4).bias == [1.0, 0.5]`; with weights copied into `coral_pytorch.layers.CoralLayer(4, 3)` the outputs are equal.
  - `test_levels_encoding`: ranks `[0,1,2]` -> `[[0,0],[1,0],[1,1]]`.
  - `test_class_probs_nonnegative_sum_to_one_even_when_levels_cross`: `lp=[[0.3,0.6]]` gives no negative entry and sums to 1.
  - `test_tone_rank_map`: dovish 0 -> 0, neutral 2 -> 1, hawkish 1 -> 2; `tone_from_levels([[0.9,0.2]]) == 0.2 - 0.1`.
  - `test_linear_coral_equals_ordered_logit`: simulate 2,000 rows from a cumulative logit (beta (1.0, -0.5), thresholds -0.5, 1.0); fit a linear `CoralHead(2)` with Adam (lr 0.05, 2,000 full-batch steps); class probabilities within 0.02 (max abs) of `models.linear.OrderedLogit(0.0).fit(X, y).predict_proba(X)`; `biases_ordered` is True.
- [ ] Step 2: Run `python -m pytest tests/coral/test_core.py -q`; expect import errors.
- [ ] Step 3: Implement `core.py`; add the dev dependency.
- [ ] Step 4: Run the tests; expect all pass.
- [ ] Step 5: Commit `CORAL core: layer, loss, levels and probability conversion, checked against coral-pytorch`.

### Task 2: CORAL networks (LSTM, GRU, TFT)

**Files:**
- Create: `src/fed_news/coral/networks.py`
- Test: `tests/coral/test_networks.py`

**Interfaces:**
- Consumes: Task 1 (`CoralHead`, `coral_loss`, `levels`, `level_probs`); `models.neural.build_model`, `make_sequences`, `standardise`.
- Produces:
  - `build_coral_model(arch, n_features, hidden, dropout, seed, layers=1, heads=4) -> nn.Module`: `neural.build_model(...)` then `model.head = CoralHead(hidden)` built under `torch.manual_seed(seed + 1)`; forward returns `(n, 2)` logits (the existing forwards end with `self.head(...).squeeze(-1)`, a no-op on width 2).
  - `fit_fixed_coral(model, X, ranks, lr, epochs, device, batch_size=32, seed=0, weight_decay=0.0)`.
  - `fit_validate_coral(model, X, ranks, X_val, ranks_val, lr, weight_decay, batch_size, max_epochs, patience, device, seed=0) -> tuple[float, int]`: best validation log loss of the class probabilities and its epoch.
  - `predict_levels(model, X, device) -> np.ndarray (n, 2)`.
  - `val_log_loss(lp, ranks) -> float`.

- [ ] Step 1: Failing tests:
  - `test_coral_models_output_two_logits_and_are_seed_deterministic` (all three archs, input (5, 12, 3)).
  - `test_coral_lstm_learns_an_ordinal_target`: sequences (400, 6, 2), rank = digitize of `X[:, -1, 0]` at (-0.5, 0.5); after `fit_fixed_coral` (lr 1e-2, 150 epochs) accuracy of `predict_rank` > majority share + 0.2 and biases ordered.
  - `test_fit_validate_coral_reports_best_epoch_and_log_loss`: returned loss equals `val_log_loss` of a model refit for that many epochs (same seed) within 1e-6.
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement.
- [ ] Step 4: Run; expect pass.
- [ ] Step 5: Commit `CORAL LSTM, GRU and TFT: CORAL head, loss and log-loss early stopping`.

### Task 3: ChronoBERT-CORAL reader

**Files:**
- Create: `src/fed_news/coral/chronobert_coral.py`
- Test: `tests/coral/test_chronobert_coral.py`

**Interfaces:**
- Consumes: Task 1; `features.chronobert_tone` (`STANCE_TYPES`, `vintage_year`, `classifier_year`, `model_name`, `eligible`, `_batches`, `_sentence_table`, `_fingerprint`, `_write_part`); `download.stance_labels.load_labels`; `download.fed_documents.build_corpus`.
- Produces:
  - `MODELS = config.DATA / "chronobert_coral_models"`, `WORK = config.DATA / "coral"`, `OUT_PATH = WORK / "doc_stance_chronobert_coral.pkl"`, `RESULTS = config.RESULTS / "coral"`.
  - `class StanceNetCoral(nn.Module)`: `__init__(self, lm: nn.Module)` (mean-pooled tokens -> `CoralHead`).
  - `train_and_score(train, texts, vintage, seed, save_dir: Path | None) -> np.ndarray (n, 2)` level probabilities; same optimiser and settings as `chronobert_tone.train_and_score`; when `save_dir` is given writes `encoder/` (save_pretrained, float32), `tokenizer/`, `coral_head.pt`, `info.json` (vintage, label order, settings, training date range, biases, `biases_ordered`).
  - `ordinal_mae(lp, ranks) -> float`.
  - `run() -> pd.DataFrame` (per-year loop, resumable by fingerprint plus a `"coral": 1` key; writes `WORK/stance/<year>.pkl` with doc_id, sentence, lp1, lp2, stance; `OUT_PATH`; `RESULTS/chronobert_coral_accuracy.csv` with accuracy, macro_f1, ordinal_mae, n_train, n_holdout, biases_ordered for all seeds).
  - CLI: `python -m fed_news.coral.chronobert_coral`.

- [ ] Step 1: Failing tests (CPU, no download; tiny `BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)` via `AutoModel.from_config`):
  - `test_labels_are_remapped_to_tone_ranks`.
  - `test_seed_average_of_levels_stays_ordered_and_tone_formula`.
  - `test_saved_model_reloads_and_gives_the_same_levels` (save a tiny model with the save helper, reload, equal outputs).
  - `test_fingerprint_marks_coral_runs` (a Study 4 fingerprint is not accepted as a CORAL one).
  - GPU-marked: `test_one_year_runs_end_to_end` (`@pytest.mark.gpu`).
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement (the model is passed to `StanceNetCoral` so tests can use a tiny encoder; `train_and_score` loads `model_name(vintage)`).
- [ ] Step 4: Run; expect pass (gpu test deselected by default).
- [ ] Step 5: Commit `ChronoBERT-CORAL reader: per-year CORAL fine-tuning, saved models, ordinal holdout error`.

### Task 4: Haiku ordinal reader

**Files:**
- Create: `src/fed_news/coral/haiku_ordinal.py`
- Test: `tests/coral/test_haiku_ordinal.py`

**Interfaces:**
- Consumes: `features.haiku_prompt` (`SYSTEM`, `MODEL`, `batch_prompt`, `_client`); `features.haiku_tone` (`chunks`, `sentences`, `PER_BATCH`).
- Produces:
  - `SYSTEM = haiku_prompt.SYSTEM[:haiku_prompt.SYSTEM.index("Judge only")] + <section 13 text>`.
  - `parse_probs(text: str, n: int) -> tuple[np.ndarray (n, 3) in order D, N, H summing to 1, int n_normalised]`; raises `ValueError` on a missing sentence, non-number, negative or zero sum.
  - `request_params(sentences) -> dict` (model, SYSTEM, thinking disabled, effort low, max_tokens 2048).
  - `TOKEN_LIMIT = 35_000_000`, `OUT_PER_SENTENCE = 15`.
  - `tone(p) -> np.ndarray` = H - D; `levels_from_probs(p) -> (n, 2)` = (N + H, H).
  - Steps `project`, `submit`, `collect` (state in `config.DATA / "coral" / "haiku_state.json"`); outputs `doc_stance_haiku_ordinal.pkl`, `sentence_probs.pkl`, `results/coral/haiku_token_projection.json`, `results/coral/haiku_index.json` (counts: unlabelled, normalised, failed first, tokens, correlation with the Study 4 document tone, argmax agreement with Study 4 labels, ties as mismatch).
  - CLI: `python -m fed_news.coral.haiku_ordinal {project,submit,collect}`.

- [ ] Step 1: Failing tests:
  - `test_system_prompt_keeps_the_study4_definitions_and_asks_for_three_numbers`.
  - `test_parse_accepts_strings_and_normalises_sums`: `'{"1": [10, 60, 30], "2": ["20","30","49"]}'` -> rows sum to 1, `n_normalised == 1`.
  - `test_parse_rejects_missing_negative_and_zero_sum` (three cases, `pytest.raises(ValueError)`).
  - `test_tone_and_levels`: `[0.1, 0.6, 0.3]` -> tone 0.2, levels (0.9, 0.3).
  - `test_projection_rule`: mean input 1,600 x 13,750 requests + 15 x 343,742 compared with 35,000,000.
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement.
- [ ] Step 4: Run; expect pass.
- [ ] Step 5: Commit `Haiku ordinal reader: three class probabilities per sentence, CORAL levels, batch steps`.

### Task 5: Ordered logit on the new tones

**Files:**
- Create: `src/fed_news/coral/run_logit.py`
- Test: `tests/coral/test_run_logit.py`

**Interfaces:**
- Consumes: Tasks 3-4 outputs (`OUT_PATH`s); `walk_forward.run_spec`, `Spec`, `common_rows`; `swap_tone.naive_panel`; `run_simple_model` (`log_loss`, `brier`, `same_rows`, `memory_flag`, `_memory_indicators`); `clark_west.compare`; `significance` (`dm_hln`, `holm`, `mcnemar_greater`, `nw_summary`).
- Produces:
  - `panels() -> dict` keys `"Co"`, `"Ho"` (and `"H"` for S+M) on common rows.
  - `specs() -> dict` names `S+M, S+M+Co, S+M+Ho, placebo-Co, placebo-Ho, shuffled-Co, shuffled-Ho`.
  - `void(cw_placebo_p, cw_shuffled_p) -> bool` (placebo < 0.05 or shuffled < 0.01).
  - `run() -> dict`; writes `results/coral/logit/preds_*.csv`, `comparison.csv` (CW t and p, DM-HLN log loss, accuracy, Brier, McNemar, Newey-West lags 4 and 8 on log loss), `verdict.json` (Holm over the 2 primary CW p-values, void flags, memory flag for Ho, rows used vs Study 4's 212, S+M reused or rerun).
  - CLI: `python -m fed_news.coral.run_logit`.

- [ ] Step 1: Failing tests: `test_holm_over_the_two_primary_clark_west_tests`; `test_void_rule_uses_clark_west_thresholds`; `test_specs_pair_each_tone_with_its_own_controls`.
- [ ] Step 2-4: Run (fail), implement, run (pass).
- [ ] Step 5: Commit `Ordered logit on the CORAL tones: Clark-West primary, Holm over 2`.

### Task 6: CORAL network runner

**Files:**
- Create: `src/fed_news/coral/run_networks.py`
- Test: `tests/coral/test_run_networks.py`

**Interfaces:**
- Consumes: Task 2; `studies.networks.run_networks` (`ARCHS`, `YEARS`, `tuning_points`, `point_for`, `suggest`, `folds_for`, `placebo_panel`, `class_report`, `_subsets`, `MIN_TRAIN_SEQ`, `PATIENCE`, `MAX_EPOCHS`, `SEEDS`); `run_simple_model.panels`; Task 3-4 tone files; `significance.nw_summary`, `holm`, `dm_hln`, `mcnemar_greater`; `sensitivity_checks.block_bootstrap`; `models.linear.move_sizes`.
- Produces:
  - `ARMS` (7): `S+M, S+M+C, S+M+H, S+M+Co, S+M+Ho, placebo-Co, placebo-Ho` mapped to panel keys and columns.
  - `PRIMARY = [("S+M+Co","S+M"), ("S+M+Ho","S+M")]`; `N_TRIALS = 20`.
  - `cv_score_coral(X, ranks, point, arch, p, device, max_lookback) -> tuple[float, int]` (mean fold validation log loss; inf if any fold has < 10 sequences).
  - `run_tuned_coral(df, cols, arch, settings, seed, device, test_start) -> DataFrame` with `date, y_true, y_pred, lag, class_true, class_pred, lp1, lp2, p_cut, p_hold, p_hike` (y_pred = lag + class probs @ `move_sizes` of the training rows).
  - `pair_stats_coral(base, new) -> dict` (NW lag 8 on log-loss difference; squared-error NW, DM-HLN, bootstrap, accuracy, McNemar).
  - `registered_holm_coral(comp) -> list` (family = 3 archs x `PRIMARY`, scheduled subset; NaN while incomplete).
  - Steps `time`, `tune`, `forecast`; outputs in `results/coral/networks/` (settings, per-seed and ensemble predictions, metrics, comparisons incl. version 2 comparison rows, bias-order log).
  - CLI: `python -m fed_news.coral.run_networks {time,tune,forecast} [--device] [--archs] [--arms] [--trials]`.

- [ ] Step 1: Failing tests:
  - `test_cv_score_coral_never_reads_rows_at_or_after_the_point` (perturb rows >= point, score unchanged).
  - `test_too_few_sequences_is_infeasible`.
  - `test_forecast_unchanged_when_later_rows_are_deleted` (synthetic panel, one arch, one seed).
  - `test_seed_average_is_over_levels_and_classes_follow_the_paper_rule`.
  - `test_holm_family_is_six_and_waits_for_missing_arms`.
- [ ] Step 2-4: Run (fail), implement, run (pass).
- [ ] Step 5: Commit `CORAL network runner: per-arm Optuna on validation log loss, walk-forward, tests and Holm over 6`.

### Task 7: Runs

- [ ] Step 1: Full test suite: `python -m pytest -q` -> 175 old + new tests pass. GPU test: `python -m pytest -q -m gpu tests/coral` passes.
- [ ] Step 2: Haiku `project`; if within 35M tokens, `submit` (cost about $2-4 at batch prices).
- [ ] Step 3: ChronoBERT-CORAL `run` on the GPU in the background (about 8 hours); check biases ordered and holdout accuracy per year as years finish.
- [ ] Step 4: Networks `time` (one trial per arch); if projected wall time > 36 hours, stop and ask the user. Start LSTM/GRU tuning of arms S+M, S+M+C, S+M+H on the CPU while the readers run.
- [ ] Step 5: Haiku `collect` when batches end (resend failures once).
- [ ] Step 6: `run_logit` once both tone files exist.
- [ ] Step 7: Tune and forecast the remaining arms (TFT on the GPU after ChronoBERT), then `forecast` writes all tables.
- [ ] Step 8: Commit results (`results/coral/`), not the models.

### Task 8: Write-up

**Files:** `results/coral/RESULTS_CORAL.md`, `results/coral/ATTEMPTS.md`, `results/coral/EXPLAINER_TH.md`

- [ ] Step 1: RESULTS_CORAL.md: every number of sections 4-7 next to the Study 4 / version 2 numbers, with the metric named on every number; the section 9 wording.
- [ ] Step 2: ATTEMPTS.md: each earlier method (from the README, docs/ and the old repository's reports) with its primary test and p-value, plus this study's.
- [ ] Step 3: EXPLAINER_TH.md (Thai): per model, one input row (vector length, column names, sequence shape), training (loss, optimiser, epochs, early stopping, refits, seeds), the chosen hyperparameters per tuning point.
- [ ] Step 4: Commit and push branch `coral`.
