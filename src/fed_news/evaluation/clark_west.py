"""Exploratory, added after results: Clark-West (2007) test for nested models, linear S vs S+N.

S+N nests S (zero text coefficients give S), and the Diebold-Mariano test is undersized for nested
comparisons (Clark and McCracken, 2001). Clark and West adjust the squared-error differential by the
squared forecast difference. This is not part of the pre-registered decision; it reads the stored Plan 1
forecasts and writes results/exploratory/clark_west.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.evaluation.stats_tests import dm_hln

TASKS = ("T1", "T2")


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
    if len(small) != len(large) or not (small["date"].to_numpy() == large["date"].to_numpy()).all():
        raise ValueError("forecast dates differ")
    y = small["y_true"].to_numpy(float)
    e_small, e_large = y - small["y_pred"].to_numpy(float), y - large["y_pred"].to_numpy(float)
    t, p = clark_west(y, small["y_pred"], large["y_pred"])
    return {"n": len(y), "dm_hln_p": dm_hln(e_small ** 2, e_large ** 2)[1], "clark_west_t": t, "clark_west_p": p}


def main() -> pd.DataFrame:
    load = lambda task, name: pd.read_csv(config.RESULTS / f"preds_{task}_{name}.csv", parse_dates=["date"])
    table = pd.DataFrame([{"task": task, **compare(load(task, "S"), load(task, "S+N"))} for task in TASKS])
    out = config.RESULTS / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "clark_west.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
