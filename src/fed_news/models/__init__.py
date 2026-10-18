"""Shared model code.

    linear.py       ridge on the rate change (T1), penalised ordered logit (T2), Taylor rule, penalty selection
                    (this file was fed_news/models.py until 2026-10-17)
    walkforward.py  the linear walk-forward engine: ladder of specs, t-2 training rule, placebo and shuffled controls
    neural.py       LSTM / GRU regressors, sequence building and training loops
    tft.py          the compact TFT-inspired network

The old module name ``fed_news.models`` is now this package; it re-exports linear.py so old code
(``from fed_news import models; models.OrderedLogit``) keeps working.
"""
from fed_news.models.linear import (OrderedLogit, choose_penalty, classes_from_change, expanding_folds,  # noqa: F401
                                    logloss_score, move_sizes, ridge_score, taylor_predict)
