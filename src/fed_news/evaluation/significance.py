"""EVALUATE: significance tests.

Diebold-Mariano with the HLN correction, McNemar, Holm correction, and a Newey-West test of a mean.

Details: Pre-registered tests (spec section 8).
"""
import numpy as np
from scipy import stats


def dm_hln(loss_base, loss_new, h: int = 1) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected Diebold-Mariano; one-sided p that loss_new < loss_base."""
    d = np.asarray(loss_base, float) - np.asarray(loss_new, float)
    n = len(d)
    centred = d - d.mean()
    var = np.mean(centred ** 2) + 2 * sum(np.mean(centred[k:] * centred[:-k]) for k in range(1, h))
    if var <= 0:
        return 0.0, 1.0 if d.mean() <= 0 else 0.0
    dm = d.mean() / np.sqrt(var / n)
    stat = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(1 - stats.t.cdf(stat, df=n - 1))


def mcnemar_greater(correct_base, correct_new) -> tuple[int, int, float]:
    """McNemar test on decisions: (fixed, broken, one-sided p that the new model gets more right)."""
    base, new = np.asarray(correct_base, bool), np.asarray(correct_new, bool)
    fixes, breaks = int((~base & new).sum()), int((base & ~new).sum())
    if fixes + breaks == 0:
        return 0, 0, 1.0
    return fixes, breaks, float(stats.binomtest(fixes, fixes + breaks, 0.5, alternative="greater").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for a family of tests."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


# Copied unchanged from fed_news/studies/study5_horizon/study5.py (fed-news-forecasting ad2183c),
# where NW_LAGS is also 8.
NW_LAGS = 8


def nw_summary(d, lags: int = NW_LAGS) -> dict:
    """Newey-West test that the mean of `d` is above zero: mean, standard error, one-sided p, 90% interval."""
    import statsmodels.api as sm
    d = np.asarray(d, float)
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean, se = float(fit.params[0]), float(fit.bse[0])
    return {"n": int(len(d)), "mean": mean, "se": se, "p_one_sided": float(1 - stats.norm.cdf(mean / se)),
            "ci90": [mean - 1.645 * se, mean + 1.645 * se]}
