"""Study 4, section 4: does the Haiku news index improve FOMC decision forecasts over the no-news model?
(docs/superpowers/specs/2026-10-12-study4-haiku-index-tuned-design.md)

Penalised ordered logit (walkforward.run_spec) on identical meetings for S+M, S+M+H (Haiku stance), S+M+C
(ChronoBERT stance, leak-free reference) and the placebo and shuffled controls. Primary: one-sided HLN
Diebold-Mariano on per-meeting log loss, S+M+H against S+M. Memory flag: Haiku's gain minus ChronoBERT's gain
regressed on whether Haiku recalled the decision from its date (memory exam), HC3 standard errors.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.evaluation.clark_west import compare
from fed_news.evaluation.evaluate import decide, metrics
from fed_news.features.haiku_index import OUT_PATH as HAIKU_PATH, RESULTS
from fed_news.features.stance_panel import naive_panel
from fed_news.features.panel import M_COLS, S_COLS
from fed_news.evaluation.stats_tests import dm_hln, mcnemar_greater
from fed_news.models.walkforward import Spec, common_rows, run_spec

D = ["n_stance", "n_stance_chg"]
PROBS = ["p_cut", "p_hold", "p_hike"]
CUTOFF = "2026-07-01"  # Haiku 5.5 training data ends June 2026
NEWS = ("S+M+H", "S+M+C", "placebo-H", "shuffled-H")


def log_loss(preds: pd.DataFrame) -> np.ndarray:
    p = np.clip(preds[PROBS].to_numpy(float), 1e-12, None)
    return -np.log(p[np.arange(len(p)), preds["class_true"].to_numpy(int)])


def brier(preds: pd.DataFrame) -> np.ndarray:
    onehot = np.eye(3)[preds["class_true"].to_numpy(int)]
    return ((preds[PROBS].to_numpy(float) - onehot) ** 2).sum(1)


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = a.index.intersection(b.index)
    return a.loc[rows], b.loc[rows]


def memory_flag(gain_h, gain_c, indicator) -> dict:
    import statsmodels.api as sm
    y = np.asarray(gain_h, float) - np.asarray(gain_c, float)
    x = sm.add_constant(np.asarray(indicator, float))
    fit = sm.OLS(y, x).fit(cov_type="HC3")
    coef, p2 = float(fit.params[1]), float(fit.pvalues[1])
    p1 = p2 / 2 if coef > 0 else 1 - p2 / 2
    return {"n": int(len(y)), "n_indicator": int(np.sum(indicator)), "coef": coef, "se": float(fit.bse[1]),
            "p_one_sided": p1, "flag": bool(coef > 0 and p1 < config.ALPHA)}


def verdict(p_main: float, p_placebo: float, p_shuffled: float, flag: bool) -> dict:
    decision = decide([p_main], [p_placebo], [p_shuffled])
    if decision == "supported":
        wording = ("the improvement is consistent with memory" if flag else
                   "the Haiku index improves forecasts; memory cannot be ruled out (pre-cutoff skill is not "
                   "identified), but no memory pattern was detected")
    elif decision == "void":
        wording = "a control beat the baseline, so the comparison is void"
    else:
        wording = "the Haiku index does not improve forecasts over the no-news model"
    return {"decision": decision, "wording": wording}


def specs() -> dict[str, tuple[str, Spec]]:
    sm_cols = S_COLS + M_COLS
    return {"S+M": ("H", Spec("S+M", "T2", "primary", sm_cols)),
            "S+M+H": ("H", Spec("S+M+H", "T2", "primary", sm_cols + D)),
            "S+M+C": ("C", Spec("S+M+C", "T2", "primary", sm_cols + D)),
            "placebo-H": ("H", Spec("placebo-H", "T2", "primary", sm_cols + D, placebo=True)),
            "shuffled-H": ("H", Spec("shuffled-H", "T2", "primary", sm_cols + D, shuffle_labels=True))}


def panels() -> dict[str, pd.DataFrame]:
    h = common_rows(naive_panel("t2", pd.read_pickle(HAIKU_PATH)))
    c = common_rows(naive_panel("t2", pd.read_pickle(config.DATA / "doc_stance.pkl")))
    h, c = same_rows(h, c)
    return {"H": h, "C": c}


def _memory_indicators(dates) -> pd.DataFrame:
    exam = pd.read_csv(config.RESULTS / "inputs_from_earlier" / "memory_exam.csv", parse_dates=["date"]).set_index("date")
    exam = exam.reindex(pd.DatetimeIndex(dates))
    return pd.DataFrame({"recalled": exam["recall"] == exam["persistence"],
                         "dated": (exam["year_guess"] - exam.index.year).abs() <= 1,
                         "has_exam": exam["recall"].notna()}, index=exam.index)


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pan, preds = panels(), {}
    for name, (which, spec) in specs().items():
        preds[name] = run_spec(pan[which], spec, config.TEST_START, proba=True)
        preds[name].to_csv(RESULTS / f"preds_T2_{name}.csv", index=False)
    base = preds["S+M"]
    ll = {k: log_loss(v) for k, v in preds.items()}
    rows = []
    for name in ("S+M",) + NEWS:
        p = preds[name]
        row = {"model": name, **metrics(p), "log_loss": float(ll[name].mean()), "brier": float(brier(p).mean())}
        if name != "S+M":
            fixes, breaks, p_mc = mcnemar_greater(base["class_pred"] == base["class_true"],
                                                  p["class_pred"] == p["class_true"])
            row.update({"dm_logloss_p": dm_hln(ll["S+M"], ll[name])[1], **compare(base, p),
                        "mcnemar_fixes": fixes, "mcnemar_breaks": breaks, "mcnemar_p": p_mc})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "comparison_T2.csv", index=False)
    get = lambda n, c: float(table.loc[table["model"] == n, c].iloc[0])

    ind = _memory_indicators(base["date"])
    keep = ind["has_exam"].to_numpy()
    gain_h, gain_c = ll["S+M"] - ll["S+M+H"], ll["S+M"] - ll["S+M+C"]
    flag = memory_flag(gain_h[keep], gain_c[keep], ind["recalled"].to_numpy()[keep])
    dated = memory_flag(gain_h[keep], gain_c[keep], ind["dated"].to_numpy()[keep])
    dated.pop("flag")  # reported without a rule

    post = base["date"] >= pd.Timestamp(CUTOFF)
    post_rows = pd.DataFrame({"date": base.loc[post, "date"], "class_true": base.loc[post, "class_true"],
                              **{f"log_loss_{k}": ll[k][post.to_numpy()] for k in ("S+M", "S+M+H", "S+M+C")}})
    post_rows.to_csv(RESULTS / "post_cutoff_T2.csv", index=False)

    res = {"n_test_meetings": int(len(base)), "first": str(base["date"].min().date()),
           "last": str(base["date"].max().date()),
           "primary_dm_logloss_p": get("S+M+H", "dm_logloss_p"),
           "placebo_dm_logloss_p": get("placebo-H", "dm_logloss_p"),
           "shuffled_dm_logloss_p": get("shuffled-H", "dm_logloss_p"),
           "memory_flag_recalled": flag, "memory_dated_descriptive": dated,
           "haiku_vs_chrono_dm_logloss_p": dm_hln(ll["S+M+C"], ll["S+M+H"])[1],
           "chrono_dm_logloss_p": get("S+M+C", "dm_logloss_p"),
           "post_cutoff": post_rows.assign(date=post_rows["date"].dt.date.astype(str)).to_dict("records")}
    res.update(verdict(res["primary_dm_logloss_p"], res["placebo_dm_logloss_p"], res["shuffled_dm_logloss_p"],
                       flag["flag"]))
    (RESULTS / "verdict_T2.json").write_text(json.dumps(res, indent=2, default=str))
    print(table.round(4).to_string(index=False))
    print(json.dumps(res, indent=2, default=str), flush=True)
    return res


if __name__ == "__main__":
    run()
