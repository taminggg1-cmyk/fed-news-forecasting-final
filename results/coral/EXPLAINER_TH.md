# อธิบายการศึกษา CORAL (ภาษาไทย): input, การ train, hyperparameter และการตรวจ loss

เอกสารนี้อธิบายว่าแต่ละโมเดลในการศึกษา CORAL รับอะไรเข้าไป ถูก train อย่างไร ใช้ hyperparameter อะไร
และเราตรวจได้อย่างไรว่า loss ถูกต้อง ตัวเลขทั้งหมดมาจากไฟล์ใน results/coral/ และโค้ดใน src/fed_news/studies/coral/

## 1. ภาพรวม

มี "ตัวอ่านข่าว" (reader) 2 ตัว ที่เปลี่ยนข้อความของ Fed ให้เป็นตัวเลข tone และมี "ตัวพยากรณ์" (forecaster) 2 แบบ

- Reader 1: **ChronoBERT-CORAL**: ChronoBERT ที่ fine-tune ด้วย output layer แบบ CORAL (ordinal)
- Reader 2: **Haiku ordinal**: Claude Haiku ให้ความน่าจะเป็น 3 ค่า (dovish, neutral, hawkish) ต่อประโยค ไม่มีการ train
- Forecaster 1: **ordered logit** (โมเดลเดิมจาก Study 4) ใส่ tone ใหม่เป็น input
- Forecaster 2: **LSTM, GRU, TFT แบบ CORAL**: เปลี่ยน layer สุดท้ายเป็น CORAL head แล้ว train ด้วย CORAL loss

## 2. Input vector ของแต่ละโมเดล

### 2.1 ChronoBERT-CORAL (ระดับประโยค)

1. ประโยค (ตัวพิมพ์เล็ก) ถูกตัดเป็น token ยาวไม่เกิน 256 token
2. ผ่าน ChronoBERT รุ่นปี y-1 (เช่น ประโยคปี 2015 ใช้ chrono-bert-v1-20141231 ซึ่งรู้ข้อความถึงสิ้นปี 2014 เท่านั้น) เป็น ModernBERT ขนาด hidden 768 จำนวน 22 layer
3. เฉลี่ยเวกเตอร์ของทุก token (mean pooling) ได้เวกเตอร์ h ขนาด **768**
4. CORAL head: คะแนนเดียว s = w·h (w ขนาด 768, ไม่มี bias) แล้วบวก bias 2 ตัว ได้ logit 2 ค่า z_k = s + b_k (k = 1, 2)
5. ความน่าจะเป็นแบบระดับ: P(rank ≥ 1) = σ(z_1), P(rank ≥ 2) = σ(z_2) โดย rank: dovish 0 < neutral 1 < hawkish 2
6. ความน่าจะเป็นราย class: P(dovish) = 1 - P1, P(neutral) = P1 - P2, P(hawkish) = P2
7. **tone ของประโยค = P(hawkish) - P(dovish)** (อยู่ระหว่าง -1 ถึง 1)

### 2.2 Haiku ordinal (ระดับประโยค)

ประโยคเดียวกัน แต่ปิดวันที่และตัวเลข (เช่น [MONTH] [NUM] [YEAR]) ส่งครั้งละ 25 ประโยค Haiku ตอบ [D, N, H]
ที่รวมกันได้ 100 แล้วแปลงเป็นความน่าจะเป็น tone ใช้สูตรเดียวกัน P(H) - P(D)
(343,742 ประโยค, ไม่มีคำตอบ 100 ประโยค = 0.03%)

### 2.3 จาก tone ของประโยค ไปเป็น input ของการพยากรณ์ (ระดับการประชุม FOMC)

- tone ของเอกสาร = ค่าเฉลี่ย tone ของทุกประโยคในเอกสาร
- สำหรับการประชุมแต่ละครั้ง: ใช้เอกสาร Fed ที่ออกระหว่าง cutoff ของการประชุมครั้งก่อน ถึง cutoff ครั้งนี้
  (cutoff = วันก่อนวันตัดสิน) เฉลี่ยแยกกลุ่ม (policy statement/press conference, minutes, speech) แล้วเฉลี่ยของกลุ่มอีกที
  ได้ **n_stance**
- **n_stance_chg** = n_stance ครั้งนี้ ลบ ครั้งก่อน

### 2.4 Input ของ ordered logit (1 แถวต่อการประชุม)

เวกเตอร์ **26 ค่า** (standardise ด้วย mean/sd ของแถว training):
- S (เศรษฐกิจ 16 ค่า, ตามที่ประกาศก่อน cutoff): s_rate_lag, s_m2_mom, s_m2_yoy, s_pce_yoy, s_pce_core_yoy,
  s_payems_chg, s_earn_yoy, s_gdp_yoy, s_u6_mom, s_u6_yoy, s_ppi_yoy, s_cpi_yoy, s_cpi_core_yoy, s_hpi_yoy,
  s_slope_10y2y, s_slope_1y6m
- M (ตลาด 8 ค่า): m_sp_tbill_3m, m_sp_y6m, m_sp_y1, m_sp_y2, m_target_level, m_zero_bound, m_vix, m_wti
- D (tone 2 ค่า): n_stance, n_stance_chg (ไม่มีในเวอร์ชัน S+M)

Output: ความน่าจะเป็น cut / hold / hike

### 2.5 Input ของ LSTM / GRU / TFT

tensor ขนาด **(lookback, 26)**: 26 ค่าแบบเดียวกับข้อ 2.4 ของการประชุม lookback ครั้งล่าสุด
(lookback เลือกจาก 2, 4, 8, 12 โดยการ tune) Output: logit 2 ค่าจาก CORAL head ได้ P(cut), P(hold), P(hike)
และพยากรณ์อัตราดอกเบี้ย = อัตราเดิม + Σ P(class) × ขนาดการขยับเฉลี่ยของ class นั้นในแถว training

## 3. การ train และ hyperparameter

### 3.1 ChronoBERT-CORAL (ไม่ได้ tune: ใช้ค่าจาก Study 4 ตามที่ตกลงไว้)

| รายการ | ค่า |
|---|---|
| ข้อมูล | ประโยคที่คนติด label 3,074 ประโยค (Trillion Dollar Words + World Central Banks) เฉพาะที่ลงวันที่ก่อนปี y; แบ่ง 20% เป็น holdout (seed 0) |
| จำนวนโมเดล | 27 ปี (2000-2026) × 3 seed (0, 1, 2) = 81 โมเดล เก็บไว้ที่ D:/fed_news/chronobert_coral_models/ |
| learning rate | 2e-5 (encoder และ weight w); bias ของ CORAL 1e-2 |
| epochs / batch / max length | 3 / 16 / 256 |
| optimizer | AdamW, weight decay 0.01 (bias ของ CORAL ไม่มี weight decay) |
| อื่น ๆ | fp16 autocast, gradient clipping 1.0 |
| ค่าเริ่มของ bias | logit ของสัดส่วนสะสมของ label ใน training (amendment 2) |
| การรวม seed | เฉลี่ย P1, P2 ของ 3 seed (ยังเรียงลำดับถูกเสมอ) |

ผล holdout (เฉลี่ย 27 ปี): accuracy 58.9% เทียบกับ softmax ของ Study 4 64.4%; macro F1 0.514 เทียบกับ 0.606

### 3.2 Ordered logit

- walk-forward: พยากรณ์ทีละการประชุมตั้งแต่ 2001 ใช้เฉพาะแถวก่อนหน้า
- penalty λ เลือกจาก (0.01, 0.03, 0.1, 0.3, 1.0, 3.0) ด้วย cross-validation 5 fold แบบ expanding window ภายในแถว training
- loss: negative log-likelihood ของ ordered logit + penalty

### 3.3 LSTM / GRU / TFT แบบ CORAL

- tune ด้วย Optuna (TPE) **20 trial** ที่ 7 จุด (การประชุมแรกของปี 2001, 2005, 2009, 2013, 2017, 2021, 2025)
- แต่ละ trial ให้คะแนนด้วย validation log loss เฉลี่ยของ 4 fold แบบ expanding ก่อนจุดนั้น จำนวน epoch = median ของ best epoch (early stopping)
- search space: hidden {8, 16, 32, 64, 128, 256} (TFT {8, 16, 32, 64}), layers {1, 2} (TFT: heads {1, 2, 4}),
  dropout 0-0.5, lr 1e-4 ถึง 1e-2 (log), weight decay 1e-6 ถึง 1e-2 (log), lookback {2, 4, 8, 12}, batch {16, 32, 64}
- tune **แยกกันทุก arm** (S+M, +Co, +Ho, +C, +H, placebo-Co, placebo-Ho) และทุก network: 3 × 7 × 7 = 147 จุด
- forecast: refit ทุก 8 การประชุม, 8 seed แล้วเฉลี่ย P(rank ≥ k)
- loss ตอน train: CORAL loss; bias เริ่มที่ logit ของสัดส่วนสะสม และไม่มี weight decay

hyperparameter ที่เลือกได้ ทุกจุด สำหรับ arm หลัก (S+M, S+M+Co, S+M+Ho); ทั้ง 147 แถวอยู่ใน
results/coral/networks_hyperparameters.csv (cv_logloss = validation log loss ของค่าที่เลือก)

| arch   | arm    | tuning_date   |   hidden | layers/heads   |   dropout |      lr |   weight_decay |   lookback |   batch |   epochs |   cv_logloss |
|:-------|:-------|:--------------|---------:|:---------------|----------:|--------:|---------------:|-----------:|--------:|---------:|-------------:|
| lstm   | S+M    | 2001-01-03    |      256 | layers 1       |     0.003 | 0.00397 |       6.24e-06 |          2 |      64 |       18 |       0.4219 |
| lstm   | S+M    | 2005-02-02    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       60 |       0.4641 |
| lstm   | S+M    | 2009-01-28    |      128 | layers 2       |     0.286 | 0.00929 |       0.00792  |          4 |      64 |       23 |       0.4799 |
| lstm   | S+M    | 2013-01-30    |       16 | layers 2       |     0.487 | 0.00193 |       0.00656  |          2 |      16 |      132 |       0.476  |
| lstm   | S+M    | 2017-02-01    |       32 | layers 2       |     0.059 | 0.0019  |       3.74e-06 |          2 |      16 |      101 |       0.4083 |
| lstm   | S+M    | 2021-01-27    |        8 | layers 2       |     0.28  | 0.00142 |       5.27e-05 |          2 |      16 |      201 |       0.4206 |
| lstm   | S+M    | 2025-01-29    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       58 |       0.4154 |
| lstm   | S+M+Co | 2001-01-03    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       32 |       0.4757 |
| lstm   | S+M+Co | 2005-02-02    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       52 |       0.4437 |
| lstm   | S+M+Co | 2009-01-28    |      128 | layers 2       |     0.143 | 0.00479 |       0.00809  |          4 |      16 |       14 |       0.4458 |
| lstm   | S+M+Co | 2013-01-30    |       16 | layers 2       |     0.225 | 0.0041  |       0.0076   |          2 |      16 |      105 |       0.4014 |
| lstm   | S+M+Co | 2017-02-01    |       32 | layers 2       |     0.059 | 0.0019  |       3.74e-06 |          2 |      16 |      113 |       0.3616 |
| lstm   | S+M+Co | 2021-01-27    |       32 | layers 2       |     0.059 | 0.0019  |       3.74e-06 |          2 |      16 |       90 |       0.459  |
| lstm   | S+M+Co | 2025-01-29    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       52 |       0.4537 |
| lstm   | S+M+Ho | 2001-01-03    |      128 | layers 2       |     0.148 | 0.00457 |       0.00887  |          4 |      16 |       31 |       0.3891 |
| lstm   | S+M+Ho | 2005-02-02    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       53 |       0.4264 |
| lstm   | S+M+Ho | 2009-01-28    |      128 | layers 1       |     0.016 | 0.00376 |       0.00673  |          4 |      16 |       18 |       0.4261 |
| lstm   | S+M+Ho | 2013-01-30    |       16 | layers 2       |     0.225 | 0.0041  |       0.0076   |          2 |      16 |       91 |       0.3687 |
| lstm   | S+M+Ho | 2017-02-01    |       32 | layers 2       |     0.059 | 0.0019  |       3.74e-06 |          2 |      16 |      107 |       0.3925 |
| lstm   | S+M+Ho | 2021-01-27    |       64 | layers 2       |     0.143 | 0.00457 |       6.26e-05 |          2 |      16 |       83 |       0.4415 |
| lstm   | S+M+Ho | 2025-01-29    |      128 | layers 2       |     0.272 | 0.00491 |       0.00723  |          8 |      16 |       42 |       0.3943 |
| gru    | S+M    | 2001-01-03    |       16 | layers 2       |     0.202 | 0.00972 |       0.00782  |          4 |      64 |       37 |       0.4022 |
| gru    | S+M    | 2005-02-02    |      128 | layers 2       |     0.17  | 0.0043  |       5.27e-05 |          4 |      16 |       27 |       0.4169 |
| gru    | S+M    | 2009-01-28    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       45 |       0.4208 |
| gru    | S+M    | 2013-01-30    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       65 |       0.3718 |
| gru    | S+M    | 2017-02-01    |      128 | layers 2       |     0.272 | 0.00491 |       0.00723  |          8 |      16 |       50 |       0.2957 |
| gru    | S+M    | 2021-01-27    |       16 | layers 2       |     0.135 | 0.00937 |       0.0076   |          2 |      64 |       83 |       0.3429 |
| gru    | S+M    | 2025-01-29    |      128 | layers 2       |     0.431 | 0.00412 |       1.62e-06 |          2 |      16 |       15 |       0.4242 |
| gru    | S+M+Co | 2001-01-03    |      128 | layers 2       |     0.215 | 0.00887 |       0.00174  |          4 |      64 |       23 |       0.4001 |
| gru    | S+M+Co | 2005-02-02    |       16 | layers 2       |     0.147 | 0.00417 |       0.00114  |          4 |      64 |       99 |       0.4286 |
| gru    | S+M+Co | 2009-01-28    |       16 | layers 2       |     0.202 | 0.00972 |       0.00782  |          4 |      64 |       60 |       0.3745 |
| gru    | S+M+Co | 2013-01-30    |      128 | layers 2       |     0.138 | 0.00471 |       0.00886  |          4 |      16 |       36 |       0.2993 |
| gru    | S+M+Co | 2017-02-01    |       64 | layers 2       |     0.437 | 0.00325 |       9.58e-06 |          2 |      16 |       53 |       0.3016 |
| gru    | S+M+Co | 2021-01-27    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       28 |       0.3854 |
| gru    | S+M+Co | 2025-01-29    |      256 | layers 2       |     0.499 | 0.00257 |       2.79e-06 |          8 |      16 |        8 |       0.4937 |
| gru    | S+M+Ho | 2001-01-03    |      128 | layers 2       |     0.138 | 0.00471 |       0.00886  |          4 |      16 |       10 |       0.3624 |
| gru    | S+M+Ho | 2005-02-02    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       73 |       0.3766 |
| gru    | S+M+Ho | 2009-01-28    |       16 | layers 2       |     0.491 | 0.00314 |       2.71e-05 |          4 |      16 |       69 |       0.4158 |
| gru    | S+M+Ho | 2013-01-30    |      128 | layers 2       |     0.138 | 0.00471 |       0.00886  |          4 |      16 |       23 |       0.3798 |
| gru    | S+M+Ho | 2017-02-01    |       16 | layers 2       |     0.203 | 0.00972 |       2.02e-05 |          4 |      16 |       39 |       0.2699 |
| gru    | S+M+Ho | 2021-01-27    |       16 | layers 2       |     0.219 | 0.00607 |       0.00716  |          4 |      16 |       41 |       0.4092 |
| gru    | S+M+Ho | 2025-01-29    |        8 | layers 1       |     0.433 | 0.0011  |       9.41e-06 |          2 |      16 |       72 |       0.4923 |
| tft    | S+M    | 2001-01-03    |       64 | heads 1        |     0.366 | 0.00972 |       0.000292 |          8 |      64 |        5 |       0.436  |
| tft    | S+M    | 2005-02-02    |       64 | heads 4        |     0.226 | 0.00458 |       7.78e-06 |          8 |      32 |       14 |       0.6437 |
| tft    | S+M    | 2009-01-28    |       16 | heads 4        |     0.38  | 0.00887 |       0.000115 |          8 |      64 |       22 |       0.6165 |
| tft    | S+M    | 2013-01-30    |        8 | heads 1        |     0.037 | 0.00805 |       0.00523  |          2 |      32 |       86 |       0.3349 |
| tft    | S+M    | 2017-02-01    |        8 | heads 1        |     0.035 | 0.00972 |       0.00465  |          2 |      32 |       39 |       0.429  |
| tft    | S+M    | 2021-01-27    |        8 | heads 4        |     0.107 | 0.0085  |       0.00465  |          4 |      32 |       34 |       0.4412 |
| tft    | S+M    | 2025-01-29    |       16 | heads 2        |     0.107 | 0.00373 |       4.56e-06 |          2 |      32 |       19 |       0.4973 |
| tft    | S+M+Co | 2001-01-03    |       64 | heads 4        |     0.369 | 0.00934 |       0.000195 |          8 |      16 |        3 |       0.4362 |
| tft    | S+M+Co | 2005-02-02    |       16 | heads 4        |     0.296 | 0.00139 |       7.8e-06  |          2 |      32 |       57 |       0.6495 |
| tft    | S+M+Co | 2009-01-28    |       16 | heads 4        |     0.399 | 0.00415 |       3.05e-06 |          8 |      32 |       24 |       0.7536 |
| tft    | S+M+Co | 2013-01-30    |        8 | heads 4        |     0.107 | 0.0085  |       0.00465  |          4 |      32 |       51 |       0.4626 |
| tft    | S+M+Co | 2017-02-01    |       16 | heads 1        |     0.078 | 0.00415 |       0.000143 |          2 |      32 |       68 |       0.3064 |
| tft    | S+M+Co | 2021-01-27    |        8 | heads 4        |     0.107 | 0.0085  |       0.00465  |          4 |      32 |       33 |       0.4299 |
| tft    | S+M+Co | 2025-01-29    |        8 | heads 4        |     0.148 | 0.00805 |       0.00523  |          4 |      32 |       30 |       0.6131 |
| tft    | S+M+Ho | 2001-01-03    |       16 | heads 2        |     0.212 | 0.00196 |       5.63e-05 |          4 |      64 |       82 |       0.6254 |
| tft    | S+M+Ho | 2005-02-02    |       64 | heads 4        |     0.307 | 0.00972 |       1.06e-06 |          8 |      32 |       20 |       0.632  |
| tft    | S+M+Ho | 2009-01-28    |        8 | heads 1        |     0.136 | 0.00972 |       0.00465  |          4 |      32 |       44 |       0.5669 |
| tft    | S+M+Ho | 2013-01-30    |        8 | heads 4        |     0.107 | 0.0085  |       0.00465  |          4 |      32 |       36 |       0.428  |
| tft    | S+M+Ho | 2017-02-01    |        8 | heads 1        |     0.035 | 0.00972 |       0.00465  |          2 |      32 |       50 |       0.2776 |
| tft    | S+M+Ho | 2021-01-27    |        8 | heads 4        |     0.079 | 0.00887 |       0.00758  |          4 |      32 |       35 |       0.4585 |
| tft    | S+M+Ho | 2025-01-29    |        8 | heads 1        |     0.037 | 0.00805 |       0.00523  |          2 |      32 |       31 |       0.5423 |

## 4. ตรวจว่า loss ถูกต้อง

**CORAL loss** ต่อแถว (K = 3 class, 2 ระดับ):

L = - Σ_k [ y_k · log σ(z_k) + (1 - y_k) · log(1 - σ(z_k)) ],  k = 1, 2

โดย y_k = 1 ถ้า rank ≥ k (dovish = (0,0), neutral = (1,0), hawkish = (1,1)) แล้วเฉลี่ยทุกแถว
(น้ำหนักของ 2 งานเท่ากัน ตามบทความ Cao, Mirjalili, Raschka 2020)

สิ่งที่ตรวจแล้ว:
1. **เทียบกับ package ของผู้เขียน**: test `test_loss_levels_and_labels_match_coral_pytorch` (tests/studies/coral/test_core.py)
   เทียบ loss, การแปลง label เป็นระดับ และการแปลงความน่าจะเป็นเป็น label กับ coral-pytorch 1.4.0 ได้ค่าตรงกัน (ผ่าน)
2. **rank consistency**: bias ต้องเรียง b_1 ≥ b_2 ทุกโมเดลของ ChronoBERT-CORAL ทั้ง 81 โมเดลเรียงถูก
   และ network ทุก seed ทุกแถวเรียงถูก (results/coral/networks/biases.csv: 0 แถวที่ไม่เรียง)
3. **neutral ไม่ถูกบีบ**: ก่อนแก้ (amendment 2) ช่องว่าง bias 0.5 ทำให้ P(neutral) ไม่เกิน 0.124 หลังแก้
   P(neutral) เฉลี่ยใน holdout อยู่ที่ 0.40-0.51 (มี test ครอบคลุม)
4. loss ของ Haiku ไม่มี เพราะ Haiku ไม่ได้ถูก train

## 5. งานใหม่ (2026-10-21): ChronoBERT ที่เรียนจาก Haiku (Hd)

input เหมือนข้อ 2.1 แต่ใช้ ChronoBERT รุ่น 2009 รุ่นเดียว และเป้าหมายเป็น "soft label" จาก Haiku:
ระดับ (P_N + P_H, P_H) ใช้ CORAL loss แบบเดียวกัน (สูตรข้อ 4 ใช้ได้กับ y_k ที่เป็นค่าระหว่าง 0-1)
train ด้วยประโยคก่อนปี 2010 จำนวน 144,211 ประโยค (มากกว่า label คนก่อนปี 2010 ประมาณ 100 เท่า)
tune learning rate {1e-5, 2e-5, 4e-5} × epochs {1, 2} โดยใช้ปี 1994-2006 train และ 2007-2009 validate
รายละเอียดใน docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md
