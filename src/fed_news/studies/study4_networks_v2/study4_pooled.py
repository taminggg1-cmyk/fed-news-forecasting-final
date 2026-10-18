"""Exploratory pooled news test for the v2 networks (DEVIATIONS.md, recorded 2026-10-13 before running).

For each network and task, the per-row squared-error differential (no-news model minus news model, rate units)
is averaged over the four news-vs-no-news pairs and its mean is tested once (one-sided Newey-West, lags = the
refit interval) with a moving-block bootstrap interval. Chosen after seeing the results; not part of the Holm
family of the primary comparisons.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.study4_networks_v2 import tune
from fed_news.studies.study4_haiku_logit.study4_sensitivity import block_bootstrap
from fed_news.evaluation.stats_tests import nw_summary

OUT = config.RESULTS / "study4" / "exploratory_pooled_news.json"
ARCHS = ("lstm", "gru", "tft")
SPLIT = pd.Timestamp("2016-01-01")
# (news, no-news) pairs per task, grouped by reader
PAIRS = {"T2": {"ChronoBERT": [("S+M+C", "S+M"), ("strict+C", "strict")],
                "Haiku": [("S+M+H", "S+M"), ("strict+H", "strict")]},
         "T1": {"ChronoBERT": [("S+C", "S"), ("strict+C", "strict")],
                "Haiku": [("S+H", "S"), ("strict+H", "strict")]}}


def pooled_differential(pairs) -> pd.Series:
    """Row-wise mean over (base, news) prediction frames of base squared error minus news squared error."""
    cols = []
    for base, new in pairs:
        b, n = base.set_index("date").sort_index(), new.set_index("date").sort_index()
        if not b.index.equals(n.index):
            raise ValueError("base and news forecasts cover different rows")
        cols.append((b["y_true"] - b["y_pred"]) ** 2 - (n["y_true"] - n["y_pred"]) ** 2)
    return pd.concat(cols, axis=1).mean(axis=1)


def summary(d: pd.Series, pair_means, lags: int) -> dict:
    nw = nw_summary(d.to_numpy(float), lags=lags)
    boot = block_bootstrap(d.to_numpy(float), lags, tune.BOOT_DRAWS, 0)
    return {"n": nw["n"], "mean": nw["mean"], "p_one_sided": nw["p_one_sided"], "ci90": nw["ci90"],
            "boot_ci90": boot["ci90"],
            "pairs_positive": f"{int(np.sum(np.asarray(pair_means) > 0))} of {len(pair_means)}"}


def _subsets(task: str, dates: pd.DatetimeIndex) -> dict:
    cfg = tune.TASKS[task]
    if task == "T2":
        unsched = pd.read_pickle(config.DATA / "t2_full.pkl")["unscheduled"]
        sched = unsched.reindex(dates).fillna(0).to_numpy() == 0
        main = {"scheduled": sched, "all meetings": np.ones(len(dates), bool)}
        return {**main, "scheduled, before 2016": sched & (dates < SPLIT),
                "scheduled, from 2016": sched & (dates >= SPLIT)}
    lo, hi = pd.Timestamp(config.PAPER_T1[0]), pd.Timestamp(config.PAPER_T1[1])
    return {"all test months": np.ones(len(dates), bool), "paper window": (dates >= lo) & (dates <= hi),
            "before 2016": dates < SPLIT, "from 2016": dates >= SPLIT}


def run() -> dict:
    res = {}
    for task, groups in PAIRS.items():
        cfg = tune.TASKS[task]
        for arch in ARCHS:
            load = lambda fs: pd.read_csv(cfg.out / f"preds_{arch}_{fs}.csv", parse_dates=["date"])
            frames = {grp: [(load(base), load(new)) for new, base in pairs] for grp, pairs in groups.items()}
            sets = {"all four": frames["ChronoBERT"] + frames["Haiku"], **frames}
            for name, pairs in sets.items():
                d = pooled_differential(pairs)
                per_pair = [pooled_differential([p]) for p in pairs]
                for sub, mask in _subsets(task, d.index).items():
                    key = f"{task} {arch} {name} | {sub}"
                    res[key] = summary(d[mask], [s[mask].mean() for s in per_pair], cfg.refit_every)
    OUT.write_text(json.dumps(res, indent=2))
    table = pd.DataFrame(res).T[["n", "mean", "p_one_sided", "boot_ci90", "pairs_positive"]]
    pd.set_option("display.width", 200)
    print(table.to_string(), flush=True)
    return res


if __name__ == "__main__":
    run()
