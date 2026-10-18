# Study 4 (exploratory study; design b4a00fd, committed before the first run)

Two separate parts. Do not mix them.

| Location | What | Status |
| --- | --- | --- |
| this folder: `preds_T2_*.csv`, `comparison_T2.csv`, `verdict_T2.json`, `post_cutoff_T2.csv` | Ordered logit, meetings: S+M vs S+M+H (Haiku) and S+M+C (ChronoBERT), placebo-H, shuffled-H | Primary of an exploratory study (7879a70). Mixed vintage |
| `sensitivity/` | Section 8 checks after the external review (partial real-time columns, ablation, HAC, bootstrap) | Exploratory, added after results (c97f905) |
| `neural/` | **Version 2** tuned LSTM / GRU / TFT, meetings (T2) | Dated amendment 6048e0d, written after v1 results; not a pre-registration. Results 1a31c73 |
| `neural_t1/` | **Version 2** tuned LSTM / GRU / TFT, months (T1, corrected cutoff t1b) | Same as above |
| `exploratory_pooled_news.json` | Pooled news test across v2 arms | Post hoc exploratory; not recomputed by the audit |
| `haiku_index.json`, `token_projection.json` | Haiku labelling summary (343,717 of 343,742 sentences) and token budget | Descriptive |
| `v2_verification.json`, `v2_timing/` | Cutoff check of panel t1b (393 months, 0 violations); compute timing | Verification |

The superseded version 1 networks are not in this repository; they remain in fed-news-forecasting
(`results/study4/superseded_v1/`, do not cite). Commit hashes refer to that repository.

Haiku 5.5's training data end in June 2026: only the meetings of 2026-07-29 and 2026-09-16 are after it
(`post_cutoff_T2.csv`).
