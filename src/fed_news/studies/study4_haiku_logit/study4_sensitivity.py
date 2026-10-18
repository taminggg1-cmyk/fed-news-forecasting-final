"""Study 4, section 8: sensitivity checks after the external review (exploratory, added after results)
(docs/superpowers/specs/2026-10-12-study4-haiku-index-tuned-design.md).

8.1 real-time data: drop the S columns whose series need the ALFRED fallback, and score the meetings after the
last fallback row. 8.2 which news columns carry the Plan 1 gain, in one pipeline. 8.3 counts by class and
turning point, HAC and block-bootstrap versions of the primary test, McNemar one- and two-sided.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.evaluation.evaluate import metrics
from fed_news.studies.study4_haiku_logit.haiku_compare import RESULTS, log_loss, panels
from fed_news.features.panel import M_COLS, S_COLS
from fed_news.evaluation.stats_tests import dm_hln, mcnemar_greater
from fed_news.models.walkforward import Spec, run_spec

OUT = RESULTS / "sensitivity"
FALLBACK_COLS = ["s_earn_yoy", "s_u6_mom", "s_u6_yoy", "s_hpi_yoy"]  # earnings, U6, house prices: vintages from 2011-12
REALTIME_FROM = "2012-06-20"  # first test meeting after the last fallback row (2012-04-25)
STANCE = ["n_stance", "n_stance_chg"]
GROUPS = {0: "cut", 1: "hold", 2: "hike"}


def realtime(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in FALLBACK_COLS]


def _spec(name: str, cols: list[str]) -> Spec:
    return Spec(name, "T2", "primary", cols)


def ablation_specs() -> dict[str, Spec]:
    sm = S_COLS + M_COLS
    out, cols = {"S+M": _spec("S+M", sm)}, sm + STANCE
    out["S+M+stance"] = _spec("S+M+stance", cols)
    name = "S+M+stance"
    for col, tag in (("n_beige", "beige"), ("n_news", "news"), ("n_gpr", "gpr")):
        cols, name = cols + [col], f"{name}+{tag}"
        out[name] = _spec(name, cols)
    out["S+M+N"] = _spec("S+M+N", cols + ["n_epu"])
    for col, tag in (("n_beige", "beige"), ("n_news", "news"), ("n_gpr", "gpr"), ("n_epu", "epu")):
        out[f"S+M+{tag}"] = _spec(f"S+M+{tag}", sm + [col])
    return out


def realtime_specs() -> dict[str, tuple[str, Spec]]:
    sm = realtime(S_COLS) + M_COLS
    return {"rt S+M": ("C", _spec("rt S+M", sm)), "rt S+M+H": ("H", _spec("rt S+M+H", sm + STANCE)),
            "rt S+M+C": ("C", _spec("rt S+M+C", sm + STANCE)),
            "rt S+M+N": ("C", _spec("rt S+M+N", sm + STANCE + ["n_beige", "n_news", "n_gpr", "n_epu"]))}


def class_counts(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    y = base["class_true"].to_numpy(int)
    ok_b = base["class_pred"].to_numpy(int) == y
    ok_n = new["class_pred"].to_numpy(int) == y
    turn = np.r_[False, y[1:] != y[:-1]]
    rows = []
    for name, mask in [(GROUPS[k], y == k) for k in (0, 1, 2)] + [("turning point", turn), ("all", np.ones_like(turn))]:
        rows.append({"group": name, "n": int(mask.sum()), "base_correct": int(ok_b[mask].sum()),
                     "new_correct": int(ok_n[mask].sum()), "net": int(ok_n[mask].sum() - ok_b[mask].sum())})
    return pd.DataFrame(rows)


def hac_p(d, lags: int) -> float:
    """One-sided p that the mean of d is positive, Newey-West (Bartlett) standard error."""
    import statsmodels.api as sm
    d = np.asarray(d, float)
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(1 - stats.norm.cdf(fit.tvalues[0]))


def block_bootstrap(d, block: int, draws: int, seed: int) -> dict:
    d = np.asarray(d, float)
    n, rng = len(d), np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = d[idx].mean(1)
    return {"mean": float(d.mean()), "ci90": [float(np.percentile(means, 5)), float(np.percentile(means, 95))],
            "share_at_or_below_zero": float((means <= 0).mean())}


def _run(pan_frame, spec) -> pd.DataFrame:
    path = OUT / f"preds_{spec.name.replace(' ', '_')}.csv"
    if not path.exists():
        run_spec(pan_frame, spec, config.TEST_START, proba=True).to_csv(path, index=False)
        print("done", spec.name, flush=True)
    return pd.read_csv(path, parse_dates=["date"])


def _row(name, p, base):
    ll, ll_b = log_loss(p), log_loss(base)
    return {"model": name, "n": len(p), "accuracy": metrics(p)["accuracy"], "log_loss": float(ll.mean()),
            "dm_logloss_p_vs_base": np.nan if p is base else dm_hln(ll_b, ll)[1]}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pan = panels()
    res = {}

    # 8.2 ablation (ChronoBERT panel = Plan 1 panel)
    abl = {name: _run(pan["C"], spec) for name, spec in ablation_specs().items()}
    t82 = pd.DataFrame([_row(k, v, abl["S+M"]) for k, v in abl.items()])
    t82.to_csv(OUT / "ablation_T2.csv", index=False)

    # 8.1 real-time data
    rt = {name: _run(pan[which], spec) for name, (which, spec) in realtime_specs().items()}
    orig = {k: pd.read_csv(RESULTS / f"preds_T2_{k}.csv", parse_dates=["date"]) for k in ("S+M", "S+M+H", "S+M+C")}
    orig["S+M+N"] = abl["S+M+N"]
    rows = []
    for window, start in (("all test meetings", config.TEST_START), ("after the last fallback row", REALTIME_FROM)):
        for label, group, base in (("original columns", orig, "S+M"), ("fallback columns dropped", rt, "rt S+M")):
            sub = {k: v[v["date"] >= pd.Timestamp(start)].reset_index(drop=True) for k, v in group.items()}
            for k, v in sub.items():
                rows.append({"window": window, "columns": label, **_row(k, v, sub[base])})
    t81 = pd.DataFrame(rows)
    t81.to_csv(OUT / "realtime_T2.csv", index=False)

    # 8.3 reporting
    base = orig["S+M"]
    counts = pd.concat([class_counts(base, orig[k]).assign(model=k) for k in ("S+M+H", "S+M+C", "S+M+N")])
    counts.to_csv(OUT / "class_counts_T2.csv", index=False)
    d = log_loss(base) - log_loss(orig["S+M+H"])
    ok_b = base["class_pred"] == base["class_true"]
    ok_h = orig["S+M+H"]["class_pred"] == orig["S+M+H"]["class_true"]
    fixes, breaks, p_one = mcnemar_greater(ok_b, ok_h)
    res["primary_dependence"] = {"hac_p_lag4": hac_p(d, 4), "hac_p_lag8": hac_p(d, 8),
                                 "block_bootstrap_8": block_bootstrap(d, 8, 10_000, 0)}
    res["mcnemar_haiku"] = {"fixes": fixes, "breaks": breaks, "p_one_sided": p_one,
                            "p_two_sided": float(stats.binomtest(fixes, fixes + breaks, 0.5).pvalue)}
    (OUT / "sensitivity.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 200)
    print(t82.round(4).to_string(index=False))
    print(t81.round(4).to_string(index=False))
    print(counts.to_string(index=False))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    run()
