"""EVALUATION (exploratory): the literature's forecast-comparison tests on every saved forecast.

Reads:   results/coral/logit/preds_T2_*.csv, results/study4/preds_T2_S+M+{H,C}.csv,
         results/distill/logit/preds_T2_S+M+Hd.csv (when present), results/coral/networks/preds_<arch>_<arm>.csv,
         <data>/t2_full.pkl (scheduled meetings) and the tone panels (in-sample tests).
Writes:  results/sigtests/pairwise.csv, in_sample.csv, joint.json, SUMMARY.md.
Run:     python -m fed_news.studies.sigtests.run_battery

Details: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, Part A. Every news version against S+M of the same
forecaster, on the full test period and on the meetings from 2010-01-01 (networks: scheduled meetings). Nothing here
changes a registered verdict; picking a test after seeing its p-value is not confirmatory evidence.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.sigtests import tests as st
from fed_news.studies.simple_model.run_simple_model import log_loss

OUT = config.RESULTS / "sigtests"
LOGIT = {"S+M": config.RESULTS / "coral" / "logit" / "preds_T2_S+M.csv",
         "S+M+Co": config.RESULTS / "coral" / "logit" / "preds_T2_S+M+Co.csv",
         "S+M+Ho": config.RESULTS / "coral" / "logit" / "preds_T2_S+M+Ho.csv",
         "S+M+C": config.RESULTS / "study4" / "preds_T2_S+M+C.csv",
         "S+M+H": config.RESULTS / "study4" / "preds_T2_S+M+H.csv",
         "S+M+Hd": config.RESULTS / "distill" / "logit" / "preds_T2_S+M+Hd.csv",
         "placebo-Co": config.RESULTS / "coral" / "logit" / "preds_T2_placebo-Co.csv",
         "placebo-Ho": config.RESULTS / "coral" / "logit" / "preds_T2_placebo-Ho.csv",
         "placebo-Hd": config.RESULTS / "distill" / "logit" / "preds_T2_placebo-Hd.csv"}
NETWORKS = config.RESULTS / "coral" / "networks"
ARCHS = ("lstm", "gru", "tft")
NET_ARMS = ("S+M", "S+M+Co", "S+M+Ho", "S+M+C", "S+M+H", "placebo-Co", "placebo-Ho")
# Placebo versions calibrate the tests: a test that finds a fake tone "significant" is not trustworthy here.
REAL = ("S+M", "S+M+Co", "S+M+Ho", "S+M+C", "S+M+H", "S+M+Hd")
PERIODS = {"full": None, "from2010": "2010-01-01"}
READERS = {"S+M+Co": "ChronoBERT-CORAL", "S+M+Ho": "Haiku ordinal", "S+M+C": "ChronoBERT (Study 4)",
           "S+M+H": "Haiku (Study 4)", "S+M+Hd": "Haiku-distilled ChronoBERT",
           "placebo-Co": "PLACEBO of ChronoBERT-CORAL", "placebo-Ho": "PLACEBO of Haiku ordinal",
           "placebo-Hd": "PLACEBO of Haiku-distilled ChronoBERT"}


def losses(p: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-meeting log score, squared rate error and ranked probability score."""
    sq = (p["y_true"].to_numpy(float) - p["y_pred"].to_numpy(float)) ** 2
    return {"log score": log_loss(p), "squared rate error": sq, "ranked probability score": st.rps(p)}


def pair_rows(base: pd.DataFrame, new: pd.DataFrame) -> list[dict]:
    """Every test of one news version against its base (positive differences favour news)."""
    lb, ln = losses(base), losses(new)
    rows = []
    for loss in lb:
        d = lb[loss] - ln[loss]
        for test, r in (("Newey-West DM, 8 lags", st.newey_west(d)), ("fixed-b DM (Coroneo-Iacone)", st.fixed_b(d)),
                        ("fixed-m DM (Coroneo-Iacone)", st.fixed_m(d))):
            rows.append({"loss": loss, "test": test, "mean_diff": float(d.mean()), "stat": r["stat"],
                         "p": r["p_one_sided"], "sided": "one"})
    gw = st.giacomini_white(lb["log score"] - ln["log score"])
    rows.append({"loss": "log score", "test": "Giacomini-White conditional", "mean_diff": gw["mean_d"],
                 "stat": gw["stat"], "p": gw["p"], "sided": "two"})
    y, small, large = base["y_true"], base["y_pred"], new["y_pred"]
    for lags, name in ((0, "Clark-West (plain variance)"), (st.BLOCK, "Clark-West, Newey-West 8 lags")):
        r = st.clark_west_nw(y, small, large, lags=lags)
        rows.append({"loss": "squared rate error", "test": name, "mean_diff": np.nan, "stat": r["stat"],
                     "p": r["p_one_sided"], "sided": "one"})
    return rows


def _check_dates(preds: dict) -> None:
    """All versions of one forecaster must forecast the same meetings."""
    dates = [tuple(p["date"]) for p in preds.values()]
    if any(d != dates[0] for d in dates):
        raise ValueError("versions forecast different meetings")


def _cut(preds: dict, start: str | None, mask=None) -> dict:
    """Restrict every version to a row mask and to the meetings from `start`."""
    out = {}
    for k, p in preds.items():
        keep = np.ones(len(p), bool) if mask is None else mask.copy()
        if start:
            keep &= (p["date"] >= pd.Timestamp(start)).to_numpy()
        out[k] = p[keep].reset_index(drop=True)
    return out


def load_logit() -> dict:
    """The ordered-logit forecasts of every version that exists."""
    preds = {k: pd.read_csv(v, parse_dates=["date"]) for k, v in LOGIT.items() if v.exists()}
    _check_dates(preds)
    return preds


def load_networks() -> dict:
    """{arch: {arm: forecasts}} and the scheduled-meeting mask."""
    out = {a: {m: pd.read_csv(NETWORKS / f"preds_{a}_{m}.csv", parse_dates=["date"]) for m in NET_ARMS}
           for a in ARCHS}
    for a in ARCHS:
        _check_dates(out[a])
    dates = pd.DatetimeIndex(out[ARCHS[0]]["S+M"]["date"])
    unsched = pd.read_pickle(config.DATA / "t2_full.pkl")["unscheduled"].reindex(dates).fillna(0).to_numpy()
    return out, unsched == 0


def pairwise(logit: dict, nets: dict, sched: np.ndarray) -> pd.DataFrame:
    """The pairwise table: every forecaster, news version, period and test."""
    rows = []
    for period, start in PERIODS.items():
        groups = [("ordered logit", _cut(logit, start))]
        groups += [(a, _cut(nets[a], start, sched)) for a in ARCHS]
        for forecaster, preds in groups:
            for model, p in preds.items():
                if model == "S+M":
                    continue
                for r in pair_rows(preds["S+M"], p):
                    rows.append({"period": period, "forecaster": forecaster, "model": model,
                                 "reader": READERS[model], "n": len(p), **r})
    return pd.DataFrame(rows)


def in_sample_table() -> pd.DataFrame:
    """In-sample ordered logit tests of each tone on all panel meetings and on the meetings from 2010."""
    from fed_news.studies.coral import run_networks as rn
    from fed_news.studies.coral.run_logit import D, common
    from fed_news.studies.distill.chronobert_distill import OUT_PATH as DISTILL
    from fed_news.features.panels import M_COLS, S_COLS
    from fed_news.features.swap_tone import naive_panel
    from fed_news.models.walk_forward import common_rows
    pan = {k: v for k, v in rn.panels().items() if k in ("H", "C", "Co", "Ho")}
    if DISTILL.exists():
        pan = common({**pan, "Hd": common_rows(naive_panel("t2", pd.read_pickle(DISTILL)))})
    rows = []
    for period, start in PERIODS.items():
        for tone, df in pan.items():
            sub = df if start is None else df[df.index >= pd.Timestamp(start)]
            r = st.in_sample(sub, S_COLS + M_COLS, D)
            if r["identified"]:  # the chi-square LR is doubtful near separation: also a rotation (placebo) test
                rot = st.rotation_lr(sub, S_COLS + M_COLS, D)
                r.update({"p_rotation": rot["p_rotation"], "n_shifts": rot["n_shifts"],
                          "rotation_null_95": rot["null_95"]})
            rows.append({"period": period, "model": f"S+M+{tone}", "reader": READERS[f"S+M+{tone}"],
                         "first": str(sub.index.min().date()), "last": str(sub.index.max().date()), **r})
    return pd.DataFrame(rows)


def joint_tests(logit: dict, nets: dict, sched: np.ndarray) -> dict:
    """SPA and StepM over each family of comparisons (log score), and the model confidence set per forecaster."""
    out = {}
    for period, start in PERIODS.items():
        lg = _cut(logit, start)
        nt = {a: _cut(nets[a], start, sched) for a in ARCHS}
        d = lambda preds, m: log_loss(preds["S+M"]) - log_loss(preds[m])
        fams = {"networks, registered 6 (Co and Ho x 3)": [(a, m) for a in ARCHS for m in ("S+M+Co", "S+M+Ho")],
                "networks, Haiku ordinal x 3": [(a, "S+M+Ho") for a in ARCHS],
                "networks, ChronoBERT-CORAL x 3": [(a, "S+M+Co") for a in ARCHS],
                "networks, Study 4 tones (C and H x 3)": [(a, m) for a in ARCHS for m in ("S+M+C", "S+M+H")]}
        res = {}
        for fam, members in fams.items():
            D = np.column_stack([d(nt[a], m) for a, m in members])
            res[fam] = {"n": int(len(D)), **st.joint(D, [f"{a} {m}" for a, m in members])}
        logit_members = [m for m in ("S+M+Co", "S+M+Ho", "S+M+Hd") if m in lg]
        D = np.column_stack([d(lg, m) for m in logit_members])
        res["ordered logit, " + ", ".join(logit_members)] = {"n": int(len(D)), **st.joint(D, logit_members)}
        real = lambda preds: pd.DataFrame({m: log_loss(p) for m, p in preds.items() if m in REAL})
        mcs = {"ordered logit": st.mcs(real(lg))}
        mcs.update({a: st.mcs(real(nt[a])) for a in ARCHS})
        out[period] = {"spa_stepm": res, "model_confidence_set_90": mcs}
    return out


def summary(pair: pd.DataFrame, ins: pd.DataFrame, joint: dict) -> str:
    """SUMMARY.md: what was run, every p below 0.05, and the caveats."""
    lines = ["# Literature significance tests on the saved forecasts (exploratory)", "",
             "Design: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, Part A. Nothing here changes a registered "
             "verdict. With this many tests some p-values fall below 0.05 by chance; choosing a test after seeing its "
             "result is not confirmatory evidence.", "",
             f"Pairwise tests run: {len(pair)}; in-sample tests: {len(ins)}.", "",
             "## Calibration on the placebo versions (a fake tone should not be significant)", ""]
    pl = pair[pair["model"].str.startswith("placebo")]
    kind = np.where(pl["forecaster"] == "ordered logit", "ordered logit", "networks")
    cal = (pl.assign(false_positive=pl["p"] < config.ALPHA, kind=kind)
           .groupby(["kind", "test", "loss"])["false_positive"].agg(["sum", "size"])
           .rename(columns={"sum": "placebo comparisons with p < 0.05", "size": "placebo comparisons"}))
    lines += [cal.to_markdown(), "",
              "A test that often calls a placebo significant is biased here and its p-values for the real tones "
              "cannot be trusted.", "", "## Out-of-sample pairwise tests with p < 0.05 (real tones)", ""]
    real_rows = pair[~pair["model"].str.startswith("placebo")]
    sig = real_rows[real_rows["p"] < config.ALPHA].sort_values(["period", "forecaster", "model", "p"])
    cols = ["period", "forecaster", "reader", "n", "loss", "test", "sided", "stat", "p"]
    lines += [sig[cols].round(4).to_markdown(index=False) if len(sig) else "None.", ""]
    lines += ["## Registered network test (Newey-West, log score) next to the small-sample versions", ""]
    keep = pair[(pair["loss"] == "log score") & pair["test"].str.contains("DM")
                & ~pair["model"].str.startswith("placebo")]
    piv = keep.pivot_table(index=["period", "forecaster", "reader"], columns="test", values="p").round(4)
    lines += [piv.to_markdown(), ""]
    lines += ["## In-sample ordered logit (all meetings of the panel, or from 2010)", "",
              "Unpenalised maximum likelihood. The market columns nearly separate the classes, so coefficients are "
              "large (max_abs_coef, per standard deviation) and the chi-square approximation of the likelihood ratio "
              "is doubtful; from 2010 (few rate moves) the fit is not identified. p_rotation compares the real likelihood-ratio "
              "gain with the gains of the same tone circularly shifted by 8 to T-8 meetings (99 shifts), which does "
              "not rely on the chi-square approximation.", ""]
    lines += [ins.reindex(columns=["period", "reader", "n", "identified", "max_abs_coef", "lr_stat", "lr_p",
                               "p_rotation", "rotation_null_95", "hac_wald_p", "first_tone_coef",
                               "first_tone_z_p_one_sided"]).round(4).to_markdown(index=False), ""]
    lines += ["## Joint tests (log score; SPA p-value, consistent version; StepM superior models at 5%)", ""]
    jrows = [{"period": per, "family": fam, "n": r["n"], "SPA p": round(r["spa_p_consistent"], 4),
              "StepM superior": ", ".join(r["stepm_superior"]) or "none"}
             for per, v in joint.items() for fam, r in v["spa_stepm"].items()]
    lines += [pd.DataFrame(jrows).to_markdown(index=False), "", "## 90% model confidence sets (log score)", ""]
    mrows = [{"period": per, "forecaster": f, "members": ", ".join(r["included"])}
             for per, v in joint.items() for f, r in v["model_confidence_set_90"].items()]
    lines += [pd.DataFrame(mrows).to_markdown(index=False), ""]
    return "\n".join(lines)


def run() -> None:
    """Run every test and write the tables and the summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    logit = load_logit()
    nets, sched = load_networks()
    pair = pairwise(logit, nets, sched)
    pair.to_csv(OUT / "pairwise.csv", index=False)
    ins = in_sample_table()
    ins.to_csv(OUT / "in_sample.csv", index=False)
    joint = joint_tests(logit, nets, sched)
    (OUT / "joint.json").write_text(json.dumps(joint, indent=2))
    text = summary(pair, ins, joint)
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    run()
