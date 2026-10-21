"""Haiku-distilled ChronoBERT study: the ordered logit (unchanged model) with the distilled tone ("Hd").

Four versions on the CORAL meetings: no news (S+M), + distilled tone (S+M+Hd), its placebo and its shuffled-label
control. Primary test: one-sided Clark-West, S+M+Hd against S+M, test meetings from 2010-01-01, one test (no
multiple-testing correction).
Reads:   <data>/t2.pkl, t2_full.pkl, doc_stance_haiku.pkl, coral/doc_stance_chronobert_coral.pkl,
         coral/doc_stance_haiku_ordinal.pkl, distill/doc_stance_distill.pkl; results/coral/logit/preds_T2_S+M.csv.
Writes:  results/distill/logit/preds_T2_*.csv, comparison_from2010.csv, comparison_full.csv, verdict.json.
Run:     python -m fed_news.studies.distill.run_logit

Details: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, sections B5-B7. Same engine and settings as the CORAL
study section 6 (walk-forward from 2001, penalty by cross-validation inside the training rows). The panels are
restricted to the CORAL meetings, so S+M must equal the CORAL S+M forecasts (checked). The void rule and the
reported statistics are those of the CORAL study; the full period from 2001 is reported without a rule.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.coral import chronobert_coral, haiku_ordinal
from fed_news.studies.coral import run_logit as rl
from fed_news.studies.coral.from2010 import START, subset
from fed_news.studies.distill.chronobert_distill import OUT_PATH as DISTILL_PATH
from fed_news.evaluation.metrics import metrics
from fed_news.features.haiku_tone import OUT_PATH as HAIKU_PATH
from fed_news.features.panels import M_COLS, S_COLS
from fed_news.features.swap_tone import naive_panel
from fed_news.models.walk_forward import Spec, common_rows, run_spec
from fed_news.studies.simple_model.run_simple_model import PROBS, brier, log_loss

OUT = config.RESULTS / "distill" / "logit"
READER = "Haiku-distilled ChronoBERT"


def specs() -> dict[str, tuple[str, Spec]]:
    """The four versions and which panel each uses (H: the Study 4 Haiku panel, Hd: the distilled tone)."""
    sm = S_COLS + M_COLS
    return {"S+M": ("H", Spec("S+M", "T2", "primary", sm)),
            "S+M+Hd": ("Hd", Spec("S+M+Hd", "T2", "primary", sm + rl.D)),
            "placebo-Hd": ("Hd", Spec("placebo-Hd", "T2", "primary", sm + rl.D, placebo=True)),
            "shuffled-Hd": ("Hd", Spec("shuffled-Hd", "T2", "primary", sm + rl.D, shuffle_labels=True))}


def panels() -> dict[str, pd.DataFrame]:
    """The Study 4 Haiku panel (for S+M) and the distilled-tone panel, on the meetings common to the CORAL panels."""
    pans = rl.common({"H": common_rows(naive_panel("t2", pd.read_pickle(HAIKU_PATH))),
                      "Co": common_rows(naive_panel("t2", pd.read_pickle(chronobert_coral.OUT_PATH))),
                      "Ho": common_rows(naive_panel("t2", pd.read_pickle(haiku_ordinal.OUT_PATH))),
                      "Hd": common_rows(naive_panel("t2", pd.read_pickle(DISTILL_PATH)))})
    return {"H": pans["H"], "Hd": pans["Hd"]}


def verdict(p: float, is_void: bool) -> dict:
    """Decision and wording fixed in the design (section B7): one test, no correction."""
    if is_void:
        return {"decision": "void", "wording": f"a control beat the no-news model, so the {READER} comparison is void"}
    if p < config.ALPHA:
        return {"decision": "supported",
                "wording": f"in this exploratory attempt, the {READER} tone improved ordered-logit forecasts of FOMC "
                           f"decisions from 2010 (p = {p:.3g})"}
    return {"decision": "not supported",
            "wording": f"not significant at 5% (p = {p:.3g}); this does not show that the {READER} tone does not help"}


def evaluate(preds: dict[str, pd.DataFrame], start: str | None) -> tuple[pd.DataFrame, dict]:
    """The comparison table and the primary test, controls and verdict on the meetings from `start` (all if None)."""
    sub = subset(preds, start) if start else preds
    base = sub["S+M"]
    rows = [{"model": "S+M", **metrics(base), "log_loss": float(log_loss(base).mean()),
             "brier": float(brier(base).mean())}]
    rows += [{"model": k, **rl.compare_row(base, p)} for k, p in sub.items() if k != "S+M"]
    table = pd.DataFrame(rows)
    cw = dict(zip(table["model"], table["clark_west_p"]))
    is_void = rl.void(cw["placebo-Hd"], cw["shuffled-Hd"])
    res = {"start": start, "n_test_meetings": int(len(base)), "first": str(base["date"].min().date()),
           "last": str(base["date"].max().date()), "n_rate_moves": int((base["class_true"] != 1).sum()),
           "primary_clark_west_p": cw["S+M+Hd"], "placebo_cw_p": cw["placebo-Hd"], "shuffled_cw_p": cw["shuffled-Hd"],
           **verdict(cw["S+M+Hd"], is_void)}
    return table, res


def _same_as_coral(base: pd.DataFrame) -> bool:
    """Do the S+M forecasts equal the CORAL study's (same meetings, same forecasts)?"""
    old = pd.read_csv(rl.OUT / "preds_T2_S+M.csv", parse_dates=["date"])
    if len(old) != len(base) or not (old["date"].to_numpy() == base["date"].to_numpy()).all():
        return False
    cols = ["y_pred"] + PROBS
    return bool(np.allclose(old[cols].to_numpy(float), base[cols].to_numpy(float), atol=1e-9))


def run() -> dict:
    """Forecast the four versions, then test from 2010 (primary) and over the full period (reported only)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pan, preds = panels(), {}
    for name, (which, spec) in specs().items():
        preds[name] = run_spec(pan[which], spec, config.TEST_START, proba=True)
        preds[name].to_csv(OUT / f"preds_T2_{name}.csv", index=False)
    out = {"s_m_equals_coral": _same_as_coral(preds["S+M"])}
    for label, start in (("from2010", START), ("full", None)):
        table, res = evaluate(preds, start)
        table.to_csv(OUT / f"comparison_{label}.csv", index=False)
        out[label] = res
        pd.set_option("display.width", 250)
        print(table.round(4).to_string(index=False), flush=True)
    out["primary"] = out["from2010"]
    (OUT / "verdict.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
    return out


if __name__ == "__main__":
    run()
