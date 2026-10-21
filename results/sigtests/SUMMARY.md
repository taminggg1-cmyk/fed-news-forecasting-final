# Literature significance tests on the saved forecasts (exploratory)

Design: docs/design/2026-10-21-tests-and-distill-design.md, Part A. Nothing here changes a registered verdict. With this many tests some p-values fall below 0.05 by chance; choosing a test after seeing its result is not confirmatory evidence.

Pairwise tests run: 576; in-sample tests: 8.

## Calibration on the placebo versions (a fake tone should not be significant)

|                                                                              |   placebo comparisons with p < 0.05 |   placebo comparisons |
|:-----------------------------------------------------------------------------|------------------------------------:|----------------------:|
| ('networks', 'Clark-West (plain variance)', 'squared rate error')            |                                  11 |                    12 |
| ('networks', 'Clark-West, Newey-West 8 lags', 'squared rate error')          |                                   8 |                    12 |
| ('networks', 'Giacomini-White conditional', 'log score')                     |                                   1 |                    12 |
| ('networks', 'Newey-West DM, 8 lags', 'log score')                           |                                   0 |                    12 |
| ('networks', 'Newey-West DM, 8 lags', 'ranked probability score')            |                                   1 |                    12 |
| ('networks', 'Newey-West DM, 8 lags', 'squared rate error')                  |                                   0 |                    12 |
| ('networks', 'fixed-b DM (Coroneo-Iacone)', 'log score')                     |                                   0 |                    12 |
| ('networks', 'fixed-b DM (Coroneo-Iacone)', 'ranked probability score')      |                                   1 |                    12 |
| ('networks', 'fixed-b DM (Coroneo-Iacone)', 'squared rate error')            |                                   0 |                    12 |
| ('networks', 'fixed-m DM (Coroneo-Iacone)', 'log score')                     |                                   1 |                    12 |
| ('networks', 'fixed-m DM (Coroneo-Iacone)', 'ranked probability score')      |                                   1 |                    12 |
| ('networks', 'fixed-m DM (Coroneo-Iacone)', 'squared rate error')            |                                   0 |                    12 |
| ('ordered logit', 'Clark-West (plain variance)', 'squared rate error')       |                                   0 |                     4 |
| ('ordered logit', 'Clark-West, Newey-West 8 lags', 'squared rate error')     |                                   0 |                     4 |
| ('ordered logit', 'Giacomini-White conditional', 'log score')                |                                   0 |                     4 |
| ('ordered logit', 'Newey-West DM, 8 lags', 'log score')                      |                                   0 |                     4 |
| ('ordered logit', 'Newey-West DM, 8 lags', 'ranked probability score')       |                                   0 |                     4 |
| ('ordered logit', 'Newey-West DM, 8 lags', 'squared rate error')             |                                   0 |                     4 |
| ('ordered logit', 'fixed-b DM (Coroneo-Iacone)', 'log score')                |                                   0 |                     4 |
| ('ordered logit', 'fixed-b DM (Coroneo-Iacone)', 'ranked probability score') |                                   0 |                     4 |
| ('ordered logit', 'fixed-b DM (Coroneo-Iacone)', 'squared rate error')       |                                   0 |                     4 |
| ('ordered logit', 'fixed-m DM (Coroneo-Iacone)', 'log score')                |                                   0 |                     4 |
| ('ordered logit', 'fixed-m DM (Coroneo-Iacone)', 'ranked probability score') |                                   0 |                     4 |
| ('ordered logit', 'fixed-m DM (Coroneo-Iacone)', 'squared rate error')       |                                   0 |                     4 |

A test that often calls a placebo significant is biased here and its p-values for the real tones cannot be trusted.

## Out-of-sample pairwise tests with p < 0.05 (real tones)

| period   | forecaster    | reader               |   n | loss                     | test                          | sided   |    stat |      p |
|:---------|:--------------|:---------------------|----:|:-------------------------|:------------------------------|:--------|--------:|-------:|
| from2010 | gru           | ChronoBERT (Study 4) | 134 | squared rate error       | Clark-West (plain variance)   | one     |  2.7455 | 0.003  |
| from2010 | gru           | ChronoBERT-CORAL     | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.2861 | 0.0005 |
| from2010 | gru           | ChronoBERT-CORAL     | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.8584 | 0.0316 |
| from2010 | gru           | Haiku (Study 4)      | 134 | squared rate error       | Clark-West (plain variance)   | one     |  2.475  | 0.0067 |
| from2010 | gru           | Haiku ordinal        | 134 | squared rate error       | Clark-West (plain variance)   | one     |  4.4163 | 0      |
| from2010 | gru           | Haiku ordinal        | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.916  | 0.0018 |
| from2010 | gru           | Haiku ordinal        | 134 | ranked probability score | Newey-West DM, 8 lags         | one     |  1.9092 | 0.0281 |
| from2010 | lstm          | ChronoBERT (Study 4) | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.0617 | 0.0011 |
| from2010 | lstm          | ChronoBERT (Study 4) | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.6592 | 0.0485 |
| from2010 | lstm          | Haiku (Study 4)      | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.2801 | 0.0005 |
| from2010 | lstm          | Haiku (Study 4)      | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.8243 | 0.0341 |
| from2010 | lstm          | Haiku ordinal        | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.8666 | 0.0001 |
| from2010 | lstm          | Haiku ordinal        | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.0787 | 0.0188 |
| from2010 | lstm          | Haiku ordinal        | 134 | squared rate error       | Newey-West DM, 8 lags         | one     |  1.7083 | 0.0438 |
| from2010 | ordered logit | Haiku ordinal        | 135 | squared rate error       | Clark-West (plain variance)   | one     |  1.6526 | 0.0492 |
| from2010 | tft           | ChronoBERT (Study 4) | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.7126 | 0.0001 |
| from2010 | tft           | ChronoBERT (Study 4) | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  3.1057 | 0.0009 |
| from2010 | tft           | ChronoBERT (Study 4) | 134 | log score                | Giacomini-White conditional   | two     |  8.2531 | 0.0161 |
| from2010 | tft           | ChronoBERT-CORAL     | 134 | squared rate error       | Clark-West (plain variance)   | one     |  1.7992 | 0.036  |
| from2010 | tft           | Haiku (Study 4)      | 134 | squared rate error       | Clark-West (plain variance)   | one     |  2.6221 | 0.0044 |
| from2010 | tft           | Haiku (Study 4)      | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.0878 | 0.0184 |
| from2010 | tft           | Haiku ordinal        | 134 | squared rate error       | Clark-West (plain variance)   | one     |  3.9468 | 0      |
| from2010 | tft           | Haiku ordinal        | 134 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  3.4451 | 0.0003 |
| from2010 | tft           | Haiku ordinal        | 134 | ranked probability score | Newey-West DM, 8 lags         | one     |  2.3924 | 0.0084 |
| from2010 | tft           | Haiku ordinal        | 134 | log score                | Newey-West DM, 8 lags         | one     |  2.2541 | 0.0121 |
| from2010 | tft           | Haiku ordinal        | 134 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  2.4291 | 0.0154 |
| from2010 | tft           | Haiku ordinal        | 134 | ranked probability score | fixed-m DM (Coroneo-Iacone)   | one     |  2.429  | 0.0178 |
| from2010 | tft           | Haiku ordinal        | 134 | log score                | fixed-b DM (Coroneo-Iacone)   | one     |  2.1836 | 0.025  |
| from2010 | tft           | Haiku ordinal        | 134 | log score                | fixed-m DM (Coroneo-Iacone)   | one     |  2.0831 | 0.0319 |
| from2010 | tft           | Haiku ordinal        | 134 | log score                | Giacomini-White conditional   | two     |  6.6067 | 0.0368 |
| from2010 | tft           | Haiku ordinal        | 134 | squared rate error       | Newey-West DM, 8 lags         | one     |  1.7311 | 0.0417 |
| full     | gru           | ChronoBERT (Study 4) | 206 | squared rate error       | Clark-West (plain variance)   | one     |  2.4219 | 0.0077 |
| full     | gru           | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West (plain variance)   | one     |  2.9106 | 0.0018 |
| full     | gru           | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.7433 | 0.0406 |
| full     | gru           | ChronoBERT-CORAL     | 206 | log score                | Giacomini-White conditional   | two     |  6.0923 | 0.0475 |
| full     | gru           | Haiku (Study 4)      | 206 | squared rate error       | Clark-West (plain variance)   | one     |  3.0538 | 0.0011 |
| full     | gru           | Haiku (Study 4)      | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.7538 | 0.0397 |
| full     | gru           | Haiku ordinal        | 206 | squared rate error       | Clark-West (plain variance)   | one     |  5.1974 | 0      |
| full     | gru           | Haiku ordinal        | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  3.3339 | 0.0004 |
| full     | gru           | Haiku ordinal        | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  2.2952 | 0.0109 |
| full     | gru           | Haiku ordinal        | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  2.0938 | 0.0275 |
| full     | gru           | Haiku ordinal        | 206 | ranked probability score | fixed-m DM (Coroneo-Iacone)   | one     |  1.8957 | 0.0436 |
| full     | gru           | Haiku ordinal        | 206 | squared rate error       | Newey-West DM, 8 lags         | one     |  1.6797 | 0.0465 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | squared rate error       | Clark-West (plain variance)   | one     |  4.2235 | 0      |
| full     | lstm          | ChronoBERT (Study 4) | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.5192 | 0.0059 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | squared rate error       | Newey-West DM, 8 lags         | one     |  2.0111 | 0.0222 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  1.9725 | 0.0243 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | log score                | Newey-West DM, 8 lags         | one     |  1.8387 | 0.033  |
| full     | lstm          | ChronoBERT (Study 4) | 206 | squared rate error       | fixed-b DM (Coroneo-Iacone)   | one     |  1.9249 | 0.0374 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  1.9112 | 0.0386 |
| full     | lstm          | ChronoBERT (Study 4) | 206 | log score                | fixed-b DM (Coroneo-Iacone)   | one     |  1.8423 | 0.0438 |
| full     | lstm          | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West (plain variance)   | one     |  2.6693 | 0.0038 |
| full     | lstm          | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.6686 | 0.0476 |
| full     | lstm          | Haiku (Study 4)      | 206 | squared rate error       | Clark-West (plain variance)   | one     |  4.3802 | 0      |
| full     | lstm          | Haiku (Study 4)      | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.5079 | 0.0061 |
| full     | lstm          | Haiku (Study 4)      | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  2.4537 | 0.0071 |
| full     | lstm          | Haiku (Study 4)      | 206 | squared rate error       | Newey-West DM, 8 lags         | one     |  2.3353 | 0.0098 |
| full     | lstm          | Haiku (Study 4)      | 206 | log score                | Newey-West DM, 8 lags         | one     |  2.1933 | 0.0141 |
| full     | lstm          | Haiku (Study 4)      | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  2.3063 | 0.018  |
| full     | lstm          | Haiku (Study 4)      | 206 | squared rate error       | fixed-b DM (Coroneo-Iacone)   | one     |  2.2453 | 0.0204 |
| full     | lstm          | Haiku (Study 4)      | 206 | log score                | fixed-b DM (Coroneo-Iacone)   | one     |  2.1388 | 0.0253 |
| full     | lstm          | Haiku (Study 4)      | 206 | squared rate error       | fixed-m DM (Coroneo-Iacone)   | one     |  2.0536 | 0.0336 |
| full     | lstm          | Haiku (Study 4)      | 206 | ranked probability score | fixed-m DM (Coroneo-Iacone)   | one     |  2.0178 | 0.0356 |
| full     | lstm          | Haiku (Study 4)      | 206 | log score                | fixed-m DM (Coroneo-Iacone)   | one     |  1.8826 | 0.0446 |
| full     | lstm          | Haiku ordinal        | 206 | squared rate error       | Clark-West (plain variance)   | one     |  4.7484 | 0      |
| full     | lstm          | Haiku ordinal        | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.6586 | 0.0039 |
| full     | lstm          | Haiku ordinal        | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  2.3828 | 0.0086 |
| full     | lstm          | Haiku ordinal        | 206 | squared rate error       | Newey-West DM, 8 lags         | one     |  2.3227 | 0.0101 |
| full     | lstm          | Haiku ordinal        | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  2.3035 | 0.0181 |
| full     | lstm          | Haiku ordinal        | 206 | squared rate error       | fixed-b DM (Coroneo-Iacone)   | one     |  2.2104 | 0.0218 |
| full     | lstm          | Haiku ordinal        | 206 | log score                | Newey-West DM, 8 lags         | one     |  1.9736 | 0.0242 |
| full     | lstm          | Haiku ordinal        | 206 | log score                | fixed-b DM (Coroneo-Iacone)   | one     |  1.9784 | 0.0337 |
| full     | lstm          | Haiku ordinal        | 206 | ranked probability score | fixed-m DM (Coroneo-Iacone)   | one     |  1.9949 | 0.037  |
| full     | lstm          | Haiku ordinal        | 206 | squared rate error       | fixed-m DM (Coroneo-Iacone)   | one     |  1.9555 | 0.0395 |
| full     | tft           | ChronoBERT (Study 4) | 206 | squared rate error       | Clark-West (plain variance)   | one     |  4.4904 | 0      |
| full     | tft           | ChronoBERT (Study 4) | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.8072 | 0.0025 |
| full     | tft           | ChronoBERT (Study 4) | 206 | log score                | Giacomini-White conditional   | two     | 11.211  | 0.0037 |
| full     | tft           | ChronoBERT (Study 4) | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  1.88   | 0.0301 |
| full     | tft           | ChronoBERT (Study 4) | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  1.9091 | 0.0388 |
| full     | tft           | ChronoBERT (Study 4) | 206 | squared rate error       | Newey-West DM, 8 lags         | one     |  1.7639 | 0.0389 |
| full     | tft           | ChronoBERT (Study 4) | 206 | ranked probability score | fixed-m DM (Coroneo-Iacone)   | one     |  1.9215 | 0.0418 |
| full     | tft           | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West (plain variance)   | one     |  1.9697 | 0.0244 |
| full     | tft           | ChronoBERT-CORAL     | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  1.6571 | 0.0487 |
| full     | tft           | Haiku (Study 4)      | 206 | squared rate error       | Clark-West (plain variance)   | one     |  3.3356 | 0.0004 |
| full     | tft           | Haiku (Study 4)      | 206 | log score                | Giacomini-White conditional   | two     | 11.18   | 0.0037 |
| full     | tft           | Haiku (Study 4)      | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.0318 | 0.0211 |
| full     | tft           | Haiku ordinal        | 206 | squared rate error       | Clark-West (plain variance)   | one     |  3.9263 | 0      |
| full     | tft           | Haiku ordinal        | 206 | log score                | Giacomini-White conditional   | two     | 14.663  | 0.0007 |
| full     | tft           | Haiku ordinal        | 206 | squared rate error       | Clark-West, Newey-West 8 lags | one     |  2.1387 | 0.0162 |
| full     | tft           | Haiku ordinal        | 206 | ranked probability score | Newey-West DM, 8 lags         | one     |  2.036  | 0.0209 |
| full     | tft           | Haiku ordinal        | 206 | log score                | Newey-West DM, 8 lags         | one     |  1.8448 | 0.0325 |
| full     | tft           | Haiku ordinal        | 206 | ranked probability score | fixed-b DM (Coroneo-Iacone)   | one     |  1.9297 | 0.037  |

## Registered network test (Newey-West, log score) next to the small-sample versions

|                                                       |   Newey-West DM, 8 lags |   fixed-b DM (Coroneo-Iacone) |   fixed-m DM (Coroneo-Iacone) |
|:------------------------------------------------------|------------------------:|------------------------------:|------------------------------:|
| ('from2010', 'gru', 'ChronoBERT (Study 4)')           |                  0.2216 |                        0.2413 |                        0.2745 |
| ('from2010', 'gru', 'ChronoBERT-CORAL')               |                  0.3165 |                        0.3276 |                        0.3476 |
| ('from2010', 'gru', 'Haiku (Study 4)')                |                  0.4368 |                        0.4412 |                        0.4495 |
| ('from2010', 'gru', 'Haiku ordinal')                  |                  0.14   |                        0.1703 |                        0.2007 |
| ('from2010', 'lstm', 'ChronoBERT (Study 4)')          |                  0.1191 |                        0.1355 |                        0.1624 |
| ('from2010', 'lstm', 'ChronoBERT-CORAL')              |                  0.3599 |                        0.368  |                        0.3768 |
| ('from2010', 'lstm', 'Haiku (Study 4)')               |                  0.0992 |                        0.1143 |                        0.1358 |
| ('from2010', 'lstm', 'Haiku ordinal')                 |                  0.1042 |                        0.1168 |                        0.1256 |
| ('from2010', 'ordered logit', 'ChronoBERT (Study 4)') |                  0.1325 |                        0.1442 |                        0.1271 |
| ('from2010', 'ordered logit', 'ChronoBERT-CORAL')     |                  0.2802 |                        0.2868 |                        0.3149 |
| ('from2010', 'ordered logit', 'Haiku (Study 4)')      |                  0.1268 |                        0.1447 |                        0.1694 |
| ('from2010', 'ordered logit', 'Haiku ordinal')        |                  0.1111 |                        0.1319 |                        0.1572 |
| ('from2010', 'tft', 'ChronoBERT (Study 4)')           |                  0.3115 |                        0.3166 |                        0.2971 |
| ('from2010', 'tft', 'ChronoBERT-CORAL')               |                  0.5741 |                        0.5682 |                        0.5595 |
| ('from2010', 'tft', 'Haiku (Study 4)')                |                  0.0918 |                        0.1009 |                        0.0925 |
| ('from2010', 'tft', 'Haiku ordinal')                  |                  0.0121 |                        0.025  |                        0.0319 |
| ('full', 'gru', 'ChronoBERT (Study 4)')               |                  0.3923 |                        0.4039 |                        0.4114 |
| ('full', 'gru', 'ChronoBERT-CORAL')                   |                  0.4684 |                        0.4714 |                        0.4752 |
| ('full', 'gru', 'Haiku (Study 4)')                    |                  0.4166 |                        0.4258 |                        0.4331 |
| ('full', 'gru', 'Haiku ordinal')                      |                  0.0758 |                        0.1125 |                        0.1503 |
| ('full', 'lstm', 'ChronoBERT (Study 4)')              |                  0.033  |                        0.0438 |                        0.06   |
| ('full', 'lstm', 'ChronoBERT-CORAL')                  |                  0.1477 |                        0.1808 |                        0.2139 |
| ('full', 'lstm', 'Haiku (Study 4)')                   |                  0.0141 |                        0.0253 |                        0.0446 |
| ('full', 'lstm', 'Haiku ordinal')                     |                  0.0242 |                        0.0337 |                        0.0547 |
| ('full', 'ordered logit', 'ChronoBERT (Study 4)')     |                  0.1974 |                        0.2035 |                        0.2191 |
| ('full', 'ordered logit', 'ChronoBERT-CORAL')         |                  0.4    |                        0.4015 |                        0.369  |
| ('full', 'ordered logit', 'Haiku (Study 4)')          |                  0.1537 |                        0.1687 |                        0.1755 |
| ('full', 'ordered logit', 'Haiku ordinal')            |                  0.1049 |                        0.1257 |                        0.148  |
| ('full', 'tft', 'ChronoBERT (Study 4)')               |                  0.1171 |                        0.1271 |                        0.1234 |
| ('full', 'tft', 'ChronoBERT-CORAL')                   |                  0.7575 |                        0.7524 |                        0.7471 |
| ('full', 'tft', 'Haiku (Study 4)')                    |                  0.2868 |                        0.2881 |                        0.292  |
| ('full', 'tft', 'Haiku ordinal')                      |                  0.0325 |                        0.0514 |                        0.0672 |

## In-sample ordered logit (all meetings of the panel, or from 2010)

Unpenalised maximum likelihood. The market columns nearly separate the classes, so coefficients are large (max_abs_coef, per standard deviation) and the chi-square approximation of the likelihood ratio is doubtful; from 2010 (few rate moves) the fit is not identified. p_rotation compares the real likelihood-ratio gain with the gains of the same tone circularly shifted by 8 to T-8 meetings (99 shifts), which does not rely on the chi-square approximation.

| period   | reader               |   n | identified   |   max_abs_coef |   lr_stat |     lr_p |   p_rotation |   rotation_null_95 |   hac_wald_p |   first_tone_coef |   first_tone_z_p_one_sided |
|:---------|:---------------------|----:|:-------------|---------------:|----------:|---------:|-------------:|-------------------:|-------------:|------------------:|---------------------------:|
| full     | Haiku (Study 4)      | 260 | True         |        13.5076 |    7.5729 |   0.0227 |         0.22 |            21.9794 |       0.0028 |            2.0053 |                     0.0605 |
| full     | ChronoBERT (Study 4) | 260 | True         |        11.8773 |    3.5157 |   0.1724 |         0.59 |            17.1251 |       0.0655 |           -0.4436 |                     0.5949 |
| full     | ChronoBERT-CORAL     | 260 | True         |        13.2523 |   14.9211 |   0.0006 |         0.1  |            18.87   |       0.0862 |           -3.2677 |                     0.9832 |
| full     | Haiku ordinal        | 260 | True         |        13.9015 |   10.7016 |   0.0047 |         0.16 |            18.6428 |       0.0009 |            2.9894 |                     0.0225 |
| from2010 | Haiku (Study 4)      | 135 | False        |       388.432  |  nan      | nan      |       nan    |           nan      |     nan      |          nan      |                   nan      |
| from2010 | ChronoBERT (Study 4) | 135 | False        |       360.504  |  nan      | nan      |       nan    |           nan      |     nan      |          nan      |                   nan      |
| from2010 | ChronoBERT-CORAL     | 135 | False        |       360.504  |  nan      | nan      |       nan    |           nan      |     nan      |          nan      |                   nan      |
| from2010 | Haiku ordinal        | 135 | False        |       360.504  |  nan      | nan      |       nan    |           nan      |     nan      |          nan      |                   nan      |

## Joint tests (log score; SPA p-value, consistent version; StepM superior models at 5%)

| period   | family                                 |   n |   SPA p | StepM superior   |
|:---------|:---------------------------------------|----:|--------:|:-----------------|
| full     | networks, registered 6 (Co and Ho x 3) | 206 |  0.0665 | none             |
| full     | networks, Haiku ordinal x 3            | 206 |  0.0372 | lstm S+M+Ho      |
| full     | networks, ChronoBERT-CORAL x 3         | 206 |  0.3987 | none             |
| full     | networks, Study 4 tones (C and H x 3)  | 206 |  0.068  | none             |
| full     | ordered logit, S+M+Co, S+M+Ho          | 212 |  0.1137 | none             |
| from2010 | networks, registered 6 (Co and Ho x 3) | 134 |  0.1964 | none             |
| from2010 | networks, Haiku ordinal x 3            | 134 |  0.1419 | none             |
| from2010 | networks, ChronoBERT-CORAL x 3         | 134 |  0.3588 | none             |
| from2010 | networks, Study 4 tones (C and H x 3)  | 134 |  0.2247 | none             |
| from2010 | ordered logit, S+M+Co, S+M+Ho          | 135 |  0.1153 | none             |

## 90% model confidence sets (log score)

| period   | forecaster    | members                           |
|:---------|:--------------|:----------------------------------|
| full     | ordered logit | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| full     | lstm          | S+M+Co, S+M+Ho, S+M+C, S+M+H      |
| full     | gru           | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| full     | tft           | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| from2010 | ordered logit | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| from2010 | lstm          | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| from2010 | gru           | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
| from2010 | tft           | S+M, S+M+Co, S+M+Ho, S+M+C, S+M+H |
