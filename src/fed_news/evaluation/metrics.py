"""EVALUATE: forecast accuracy and the pass / fail rule.

RMSE, MAE, mean error, decisions called correctly, macro F1; and the decision rule with placebo and shuffled controls.

Details: Metrics, the pre-registered H1 decision, and RESULTS.md (spec section 8).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fed_news import config


def metrics(preds: pd.DataFrame) -> dict:
    """Accuracy numbers for one forecast table: n, MSE, RMSE, MAE, mean error, decisions correct, macro F1."""
    err = preds["y_pred"] - preds["y_true"]
    return {"n": len(preds), "mse": float(np.mean(err ** 2)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "mbe": float(np.mean(err)),
            "accuracy": float((preds["class_pred"] == preds["class_true"]).mean()),
            "macro_f1": float(f1_score(preds["class_true"], preds["class_pred"], average="macro"))}


def decide(main_p, placebo_p, shuffled_p, alpha: float = config.ALPHA, shuffled_alpha: float = 0.01) -> str:
    """'supported', 'not supported' or 'void' (a placebo or the shuffled control passed) from the p-values."""
    for p in list(main_p) + list(placebo_p) + list(shuffled_p):
        if np.isnan(p):
            raise ValueError("NaN p-value in decision inputs")
    if any(p < alpha for p in placebo_p) or any(p < shuffled_alpha for p in shuffled_p):
        return "void"
    return "supported" if all(p < alpha for p in main_p) else "not supported"
