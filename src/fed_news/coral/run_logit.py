"""CORAL STUDY: the ordered logit (unchanged model) with the ChronoBERT-CORAL and Haiku ordinal tones.

Seven versions on the same meetings: no news (S+M), + ChronoBERT-CORAL tone (S+M+Co), + Haiku ordinal tone (S+M+Ho),
and for each tone a placebo and a shuffled-label control. Primary test: Clark-West, Holm over the two tones.
Reads:   <data>/t2.pkl, t2_full.pkl, doc_stance_haiku.pkl, coral/doc_stance_chronobert_coral.pkl,
         coral/doc_stance_haiku_ordinal.pkl; results/study4/preds_T2_S+M.csv, comparison_T2.csv (reference).
Writes:  results/coral/logit/preds_T2_*.csv, comparison.csv, verdict.json.
Run:     python -m fed_news.coral.run_logit

Details: CORAL study, section 6. Same engine and settings as Study 4 section 4 (walk-forward, penalty by
expanding-window CV inside the training rows, class probabilities saved). S+M is rerun on the common rows of all
panels and checked against the Study 4 forecasts. Primary: one-sided Clark-West on the rate forecast, S+M+Co and
S+M+Ho against S+M, Holm over the 2. A tone's result is void if its placebo beats S+M at p < 0.05 or its shuffled
control at p < 0.01 (Clark-West). Reported without a rule: DM-HLN on log loss, Newey-West (lags 4 and 8) on the
log-loss difference, accuracy, Brier, McNemar. Memory flag for the Haiku tone as in Study 4.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.coral import chronobert_coral, haiku_ordinal
from fed_news.evaluation.clark_west import compare
from fed_news.evaluation.metrics import metrics
from fed_news.evaluation.significance import dm_hln, holm, mcnemar_greater, nw_summary
from fed_news.features.haiku_tone import OUT_PATH as HAIKU_PATH
from fed_news.features.panels import M_COLS, S_COLS
from fed_news.features.swap_tone import naive_panel
from fed_news.models.walk_forward import Spec, common_rows, run_spec
from fed_news.studies.simple_model.run_simple_model import PROBS, _memory_indicators, brier, log_loss, memory_flag

D = ["n_stance", "n_stance_chg"]
OUT = config.RESULTS / "coral" / "logit"
READERS = {"S+M+Co": "ChronoBERT-CORAL", "S+M+Ho": "Haiku ordinal"}


def specs() -> dict[str, tuple[str, Spec]]:
    """The seven versions and which panel each uses (H: Study 4 Haiku panel, Co, Ho: the CORAL tones)."""
    sm = S_COLS + M_COLS
    out = {"S+M": ("H", Spec("S+M", "T2", "primary", sm))}
    for tone in ("Co", "Ho"):
        out[f"S+M+{tone}"] = (tone, Spec(f"S+M+{tone}", "T2", "primary", sm + D))
        out[f"placebo-{tone}"] = (tone, Spec(f"placebo-{tone}", "T2", "primary", sm + D, placebo=True))
        out[f"shuffled-{tone}"] = (tone, Spec(f"shuffled-{tone}", "T2", "primary", sm + D, shuffle_labels=True))
    return out


def common(pans: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every panel restricted to the meetings present in all of them."""
    rows = None
    for p in pans.values():
        rows = p.index if rows is None else rows.intersection(p.index)
    return {k: p.loc[rows] for k, p in pans.items()}


def panels() -> dict[str, pd.DataFrame]:
    """The Study 4 Haiku panel (for S+M) and the two CORAL-tone panels, on their common meetings."""
    return common({"H": common_rows(naive_panel("t2", pd.read_pickle(HAIKU_PATH))),
                   "Co": common_rows(naive_panel("t2", pd.read_pickle(chronobert_coral.OUT_PATH))),
                   "Ho": common_rows(naive_panel("t2", pd.read_pickle(haiku_ordinal.OUT_PATH)))})


def primary_holm(p_co: float, p_ho: float) -> dict[str, float]:
    """Holm-adjusted Clark-West p-values over the two primary comparisons."""
    return holm({"S+M+Co": p_co, "S+M+Ho": p_ho})


def void(cw_placebo_p: float, cw_shuffled_p: float) -> bool:
    """A control beat the no-news model: placebo at p < 0.05 or shuffled labels at p < 0.01 (Clark-West)."""
    return bool(cw_placebo_p < config.ALPHA or cw_shuffled_p < 0.01)


MEMORY_CAVEAT = ("; memory cannot be ruled out: Claude Haiku 5.5 may have seen these meetings' outcomes in training "
                 "(pre-cutoff skill is not identified)")


def verdict(reader: str, p_holm: float, is_void: bool, haiku: bool = False) -> dict:
    """Decision and wording fixed in the design (section 9); any Haiku result carries the memory caveat."""
    caveat = MEMORY_CAVEAT if haiku else ""
    if is_void:
        return {"decision": "void",
                "wording": f"a control beat the no-news model, so the {reader} comparison is void{caveat}"}
    if p_holm < config.ALPHA:
        return {"decision": "supported",
                "wording": f"in this exploratory study, the {reader} tone improved ordered-logit forecasts of FOMC "
                           f"decisions (p = {p_holm:.3g}, Holm-adjusted){caveat}"}
    return {"decision": "not supported",
            "wording": f"not significant at 5% in this exploratory study (p = {p_holm:.3g}, Holm-adjusted); this "
                       f"does not show that the {reader} tone does not help{caveat}"}


def compare_row(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Every reported statistic of one news version against S+M (positive differences favour the news version)."""
    ll_b, ll_n = log_loss(base), log_loss(new)
    cw = compare(base, new)
    fixes, breaks, p_mc = mcnemar_greater(base["class_pred"] == base["class_true"],
                                          new["class_pred"] == new["class_true"])
    return {**metrics(new), "log_loss": float(ll_n.mean()), "brier": float(brier(new).mean()),
            "clark_west_t": cw["clark_west_t"], "clark_west_p": cw["clark_west_p"], "dm_sqerr_p": cw["dm_hln_p"],
            "dm_logloss_p": dm_hln(ll_b, ll_n)[1],
            "nw4_logloss_p": nw_summary(ll_b - ll_n, lags=4)["p_one_sided"],
            "nw8_logloss_p": nw_summary(ll_b - ll_n, lags=8)["p_one_sided"],
            "mcnemar_fixes": fixes, "mcnemar_breaks": breaks, "mcnemar_p": p_mc}


def _same_as_study4(base: pd.DataFrame) -> bool:
    """Do the rerun S+M forecasts equal the Study 4 ones (same meetings, same forecasts)?"""
    old = pd.read_csv(config.RESULTS / "study4" / "preds_T2_S+M.csv", parse_dates=["date"])
    if len(old) != len(base) or not (old["date"].to_numpy() == base["date"].to_numpy()).all():
        return False
    cols = ["y_pred"] + PROBS
    return bool(np.allclose(old[cols].to_numpy(float), base[cols].to_numpy(float), atol=1e-9))


def run() -> dict:
    """Run the seven versions, the tests, the controls and the memory flag; save forecasts, table and verdict."""
    OUT.mkdir(parents=True, exist_ok=True)
    pan, preds = panels(), {}
    for name, (which, spec) in specs().items():
        preds[name] = run_spec(pan[which], spec, config.TEST_START, proba=True)
        preds[name].to_csv(OUT / f"preds_T2_{name}.csv", index=False)
    base = preds["S+M"]
    rows = [{"model": "S+M", **metrics(base), "log_loss": float(log_loss(base).mean()),
             "brier": float(brier(base).mean())}]
    rows += [{"model": name, **compare_row(base, p)} for name, p in preds.items() if name != "S+M"]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    cw = dict(zip(table["model"], table.get("clark_west_p")))
    adj = primary_holm(cw["S+M+Co"], cw["S+M+Ho"])
    res = {"n_test_meetings": int(len(base)), "first": str(base["date"].min().date()),
           "last": str(base["date"].max().date()), "n_panel_rows": int(len(pan["H"])),
           "s_m_equals_study4": _same_as_study4(base), "primary_clark_west_p": {k: cw[k] for k in READERS},
           "primary_holm_p": adj}
    for name, reader in READERS.items():
        tone = name.split("+")[-1]
        is_void = void(cw[f"placebo-{tone}"], cw[f"shuffled-{tone}"])
        res[name] = {"placebo_cw_p": cw[f"placebo-{tone}"], "shuffled_cw_p": cw[f"shuffled-{tone}"],
                     **verdict(reader, adj[name], is_void, haiku=tone == "Ho")}
    ll = {k: log_loss(v) for k, v in preds.items()}
    ind = _memory_indicators(base["date"])
    keep = ind["has_exam"].to_numpy()
    gain_ho, gain_co = ll["S+M"] - ll["S+M+Ho"], ll["S+M"] - ll["S+M+Co"]
    res["memory_flag_haiku"] = memory_flag(gain_ho[keep], gain_co[keep], ind["recalled"].to_numpy()[keep])
    res["study4_reference"] = pd.read_csv(config.RESULTS / "study4" / "comparison_T2.csv").set_index("model")[
        ["log_loss", "accuracy", "dm_logloss_p", "clark_west_p"]].to_dict("index")
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(json.dumps(res, indent=2, default=str), flush=True)
    return res


if __name__ == "__main__":
    run()
