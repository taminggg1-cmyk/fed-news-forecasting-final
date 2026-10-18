"""EVALUATE: the Clark-West test for nested forecasts (a model with news nests the same model without it).

Details: Exploratory, added after results: Clark-West (2007) test for nested models, linear S vs S+N.

S+N nests S (zero text coefficients give S), and the Diebold-Mariano test is undersized for nested
comparisons (Clark and McCracken, 2001). Clark and West adjust the squared-error differential by the
squared forecast difference. This is not part of the pre-registered decision; it reads the stored Plan 1
forecasts and writes results/exploratory/clark_west.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

from fed_news.evaluation.significance import dm_hln


def clark_west(y, pred_small, pred_large) -> tuple[float, float]:
    """One-sided Clark-West test that the larger model adds predictive information (horizon 1)."""
    y, small, large = (np.asarray(a, float) for a in (y, pred_small, pred_large))
    f = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
    sd = f.std(ddof=1)
    if sd == 0:
        return 0.0, 0.5
    t = float(f.mean() / (sd / np.sqrt(len(f))))
    return t, float(1 - stats.norm.cdf(t))


def compare(small: pd.DataFrame, large: pd.DataFrame) -> dict:
    """Diebold-Mariano and Clark-West p-values for two forecast tables on the same dates."""
    if len(small) != len(large) or not (small["date"].to_numpy() == large["date"].to_numpy()).all():
        raise ValueError("forecast dates differ")
    y = small["y_true"].to_numpy(float)
    e_small, e_large = y - small["y_pred"].to_numpy(float), y - large["y_pred"].to_numpy(float)
    t, p = clark_west(y, small["y_pred"], large["y_pred"])
    return {"n": len(y), "dm_hln_p": dm_hln(e_small ** 2, e_large ** 2)[1], "clark_west_t": t, "clark_west_p": p}
