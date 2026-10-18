# Inputs from earlier designs

Two files the final study reads, copied unchanged from fed-news-forecasting (`ad2183c`). They are inputs here, not
results of this study.

| File | Came from | Read by | Used for |
| --- | --- | --- | --- |
| `memory_exam.csv` | `results/poc_haiku/memory_exam.csv` (Haiku memory exam, proofs of concept, 2026-10-12) | `studies/study4_haiku_logit/haiku_compare.py` | the memory flag: is Haiku's gain larger on meetings it recalled? |
| `paper_settings/preds_T1_<arch>-S.csv` | `results/h2a/` (Plan 2: LSTM / GRU / TFT with the paper's settings, untuned) | `studies/study4_networks_v2/tune.py` | the "settings" rows of `results/study4/neural_t1/comparison_neural_T1.csv`: tuned against untuned |
