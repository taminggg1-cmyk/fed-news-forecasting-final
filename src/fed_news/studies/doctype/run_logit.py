"""Attempt 14: the ordered logit (unchanged model) with the distilled tone split by document type ("Hs").

Four versions on the CORAL meetings: no news (S+M), + the 6 split-tone columns (S+M+Hs), its placebo and its
shuffled-label control. Primary test: one-sided Clark-West, S+M+Hs against S+M, test meetings from 2010-01-01, one
test (no multiple-testing correction). Reported without a rule: the full period from 2001, and the split tone
against the single distilled tone (S+M+Hs against S+M+Hd, attempt 12's forecasts).
Reads:   <data>/t2.pkl, t2_full.pkl, doc_stance_haiku.pkl, coral/doc_stance_chronobert_coral.pkl,
         coral/doc_stance_haiku_ordinal.pkl, distill/doc_stance_distill.pkl; results/coral/logit/preds_T2_S+M.csv,
         results/distill/logit/preds_T2_S+M+Hd.csv.
Writes:  results/doctype/logit/preds_T2_*.csv, comparison_from2010.csv, comparison_full.csv, verdict.json.
Run:     python -m fed_news.studies.doctype.run_logit

Details: docs/studies/5_doctype/2026-10-21-doctype-tone-design.md, sections 2-4. Same engine and settings as
attempt 12; the placebo shuffles the 6 Hs columns (Spec.news_cols). The panels are restricted to the CORAL
meetings, so S+M must equal the CORAL S+M forecasts (checked).
"""
import json

import pandas as pd

from fed_news import config
from fed_news.evaluation.metrics import metrics
from fed_news.features.haiku_tone import OUT_PATH as HAIKU_PATH
from fed_news.features.panels import M_COLS, S_COLS
from fed_news.features.swap_tone import naive_panel
from fed_news.models.walk_forward import Spec, common_rows, run_spec
from fed_news.studies.coral import chronobert_coral, haiku_ordinal
from fed_news.studies.coral import run_logit as rl
from fed_news.studies.coral.from2010 import START, subset
from fed_news.studies.distill import run_logit as dl
from fed_news.studies.distill.chronobert_distill import OUT_PATH as DISTILL_PATH
from fed_news.studies.doctype.group_tone import HS, split_panel
from fed_news.studies.simple_model.run_simple_model import brier, log_loss

OUT = config.RESULTS / "doctype" / "logit"
READER = "distilled tone split by document type"
SINGLE = dl.OUT / "preds_T2_S+M+Hd.csv"


def specs() -> dict[str, tuple[str, Spec]]:
    """The four versions and which panel each uses (H: the Study 4 Haiku panel, Hs: the split tone)."""
    sm = S_COLS + M_COLS
    return {"S+M": ("H", Spec("S+M", "T2", "primary", sm)),
            "S+M+Hs": ("Hs", Spec("S+M+Hs", "T2", "primary", sm + HS)),
            "placebo-Hs": ("Hs", Spec("placebo-Hs", "T2", "primary", sm + HS, placebo=True, news_cols=HS)),
            "shuffled-Hs": ("Hs", Spec("shuffled-Hs", "T2", "primary", sm + HS, shuffle_labels=True))}


def panels() -> dict[str, pd.DataFrame]:
    """The Study 4 Haiku panel (for S+M) and the split-tone panel, on the meetings common to the CORAL panels."""
    distill = pd.read_pickle(DISTILL_PATH)
    pans = rl.common({"H": common_rows(naive_panel("t2", pd.read_pickle(HAIKU_PATH))),
                      "Co": common_rows(naive_panel("t2", pd.read_pickle(chronobert_coral.OUT_PATH))),
                      "Ho": common_rows(naive_panel("t2", pd.read_pickle(haiku_ordinal.OUT_PATH))),
                      "Hs": common_rows(split_panel(distill))})
    if pans["Hs"][HS].isna().any().any():
        raise ValueError("missing split-tone values on the CORAL meetings")
    return {"H": pans["H"], "Hs": pans["Hs"]}


def verdict(p: float, is_void: bool) -> dict:
    """Decision and wording fixed in the design (section 3): one test, no correction."""
    if is_void:
        return {"decision": "void", "wording": f"a control beat the no-news model, so the {READER} comparison is void"}
    if p < config.ALPHA:
        return {"decision": "supported",
                "wording": f"in this exploratory attempt, the {READER} improved ordered-logit forecasts of FOMC "
                           f"decisions from 2010 (p = {p:.3g})"}
    return {"decision": "not supported",
            "wording": f"not significant at 5% (p = {p:.3g}); this does not show that the {READER} does not help"}


def evaluate(preds: dict[str, pd.DataFrame], start: str | None) -> tuple[pd.DataFrame, dict]:
    """The comparison table, the primary test, the controls, the verdict and the split against the single tone
    (preds["S+M+Hd"]) on the meetings from `start` (all if None)."""
    sub = subset(preds, start or "1900-01-01")  # also checks that every version has the same meetings
    base = sub["S+M"]
    rows = [{"model": "S+M", **metrics(base), "log_loss": float(log_loss(base).mean()),
             "brier": float(brier(base).mean())}]
    rows += [{"model": k, **rl.compare_row(base, p)} for k, p in sub.items() if k != "S+M"]
    table = pd.DataFrame(rows)
    cw = dict(zip(table["model"], table["clark_west_p"]))
    is_void = rl.void(cw["placebo-Hs"], cw["shuffled-Hs"])
    split = rl.compare_row(sub["S+M+Hd"], sub["S+M+Hs"])
    res = {"start": start, "n_test_meetings": int(len(base)), "first": str(base["date"].min().date()),
           "last": str(base["date"].max().date()), "n_rate_moves": int((base["class_true"] != 1).sum()),
           "primary_clark_west_p": cw["S+M+Hs"], "placebo_cw_p": cw["placebo-Hs"], "shuffled_cw_p": cw["shuffled-Hs"],
           **verdict(cw["S+M+Hs"], is_void),
           "split_vs_single": {k: split[k] for k in ("clark_west_p", "dm_logloss_p", "nw8_logloss_p", "log_loss",
                                                    "accuracy", "mcnemar_fixes", "mcnemar_breaks")}}
    return table, res


def run() -> dict:
    """Forecast the four versions, then test from 2010 (primary) and over the full period (reported only)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pan, preds = panels(), {}
    for name, (which, spec) in specs().items():
        preds[name] = run_spec(pan[which], spec, config.TEST_START, proba=True)
        preds[name].to_csv(OUT / f"preds_T2_{name}.csv", index=False)
    preds["S+M+Hd"] = pd.read_csv(SINGLE, parse_dates=["date"])
    out = {"s_m_equals_coral": dl._same_as_coral(preds["S+M"])}
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
