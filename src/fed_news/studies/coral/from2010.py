"""CORAL STUDY, amendment 3: the saved walk-forward forecasts scored only on the test meetings from 2010-01-01.

Nothing is refit. Every forecast was made only from data before its meeting, so scoring the meetings from 2010
equals training up to the end of 2009 and forecasting from 2010 with the same walk-forward.
Reads:   results/coral/logit/preds_T2_*.csv; results/coral/networks/preds_<arch>_<arm>.csv.
Writes:  results/coral/logit_from2010/comparison.csv, verdict.json;
         results/coral/networks_from2010/comparison.csv, summary.json.
Run:     python -m fed_news.studies.coral.from2010 logit   (or networks, once the networks have forecast)

Details: CORAL study, section 15. The same tests, controls, Holm families and wording as sections 6, 7 and 9,
computed on the subset. The network loop repeats the comparison part of run_networks.forecast_all instead of
changing that module while its tuning run is going.
"""
import argparse
import json

import pandas as pd

from fed_news import config
from fed_news.studies.coral import run_logit as rl

START = "2010-01-01"
OUT_LOGIT = config.RESULTS / "coral" / "logit_from2010"
OUT_NETWORKS = config.RESULTS / "coral" / "networks_from2010"


def subset(preds: dict, start: str = START) -> dict:
    """Every version's forecasts for the test meetings dated `start` or later; all versions must keep the same
    meetings."""
    out = {k: p[p["date"] >= pd.Timestamp(start)].reset_index(drop=True) for k, p in preds.items()}
    dates = [tuple(p["date"]) for p in out.values()]
    if not dates or not dates[0]:
        raise ValueError(f"no test meetings from {start}")
    if any(d != dates[0] for d in dates):
        raise ValueError(f"the versions do not have the same meetings from {start}")
    return out


def run_logit(start: str = START) -> dict:
    """Section 6 tests on the meetings from `start`: writes the comparison table and the verdict."""
    preds = {name: pd.read_csv(rl.OUT / f"preds_T2_{name}.csv", parse_dates=["date"]) for name in rl.specs()}
    sub = subset(preds, start)
    table, ev = rl.evaluate(sub)
    base = sub["S+M"]
    res = {"start": start, "n_test_meetings": int(len(base)), "first": str(base["date"].min().date()),
           "last": str(base["date"].max().date()), "n_rate_moves": int((base["class_true"] != 1).sum()), **ev}
    OUT_LOGIT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_LOGIT / "comparison.csv", index=False)
    (OUT_LOGIT / "verdict.json").write_text(json.dumps(res, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(json.dumps(res, indent=2, default=str), flush=True)
    return res


def run_networks(start: str = START) -> pd.DataFrame:
    """Section 7 primary tests (Holm over the 6) and placebo reading on the meetings from `start`: writes the
    comparison table and the summary."""
    from fed_news.studies.coral import run_networks as rn  # imports torch and optuna; only this step needs them
    preds = {(arch, arm): pd.read_csv(rn.OUT / f"preds_{arch}_{arm}.csv", parse_dates=["date"])
             for arch in rn.ARCHS for arm in rn.ARMS if (rn.OUT / f"preds_{arch}_{arm}.csv").exists()}
    if not preds:
        raise FileNotFoundError(f"no network forecasts in {rn.OUT}; run the forecast step first")
    sub, pan, comps = subset(preds, start), rn.panels(), []
    for arch in rn.ARCHS:
        for new, base in rn.PAIRS:
            if (arch, new) not in sub or (arch, base) not in sub:
                continue
            role = ("primary" if (new, base) in rn.PRIMARY else "placebo" if "placebo" in new + base else "old_tone")
            b, n = sub[(arch, base)], sub[(arch, new)]
            for name, mask in rn._subsets(rn.T2, pan, b["date"]).items():
                comps.append({"arch": arch, "model": new, "against": base, "role": role, "subset": name,
                              **rn.pair_stats_coral(b[mask].reset_index(drop=True), n[mask].reset_index(drop=True))})
    comp = pd.DataFrame(comps)
    comp["nw_p_holm"] = rn.registered_holm_coral(comp)
    OUT_NETWORKS.mkdir(parents=True, exist_ok=True)
    comp.to_csv(OUT_NETWORKS / "comparison.csv", index=False)
    summary = {"start": start, "n_meetings": int(len(next(iter(sub.values())))), "results": rn._summary(comp)}
    (OUT_NETWORKS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(comp.drop(columns=[c for c in comp.columns if c.endswith("ci90")]).round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("forecaster", choices=["logit", "networks"])
    args = ap.parse_args()
    run_logit() if args.forecaster == "logit" else run_networks()
