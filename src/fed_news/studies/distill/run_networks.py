"""Haiku-distilled ChronoBERT study, secondary: the CORAL LSTM, GRU and TFT with the distilled tone ("Hd").

Two new arms per network (S+M+Hd and its placebo), tuned and forecast exactly as in the CORAL study; the S+M arm is
reused from the CORAL study (same meetings, checked). Primary for this part: Newey-West (8 lags) on the log-loss
difference, S+M+Hd against S+M, scheduled meetings from 2010-01-01, Holm over the 3 networks; placebo reading.
Reads:   the CORAL panels and <data>/distill/doc_stance_distill.pkl; results/coral/networks/preds_<arch>_S+M.csv.
Writes:  results/distill/networks/settings_*.json, preds_*.csv, metrics.csv, comparison.csv, summary.json.
Run:     python -m fed_news.studies.distill.run_networks tune      (CUDA)
         python -m fed_news.studies.distill.run_networks forecast  (CUDA)

Details: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, section B8; CORAL study section 7 for the networks.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.coral import run_networks as rn
from fed_news.studies.coral.from2010 import START
from fed_news.studies.coral.run_logit import common
from fed_news.studies.distill.chronobert_distill import OUT_PATH as DISTILL_PATH
from fed_news.evaluation.metrics import metrics
from fed_news.evaluation.significance import holm
from fed_news.features.swap_tone import naive_panel
from fed_news.models.walk_forward import common_rows
from fed_news.studies.simple_model.run_simple_model import log_loss

ARMS = {"S+M+Hd": ("Hd", rn.SM + rn.D), "placebo-Hd": ("Hdp", rn.SM + rn.D)}
PAIRS = [("S+M+Hd", "S+M"), ("placebo-Hd", "S+M"), ("S+M+Hd", "placebo-Hd")]
OUT = config.RESULTS / "distill" / "networks"
PERIODS = {"from2010": START, "full": None}


def panels() -> dict[str, pd.DataFrame]:
    """The CORAL panels plus the distilled-tone panel and its placebo, all on the same meetings."""
    pan = rn.panels()
    base = {k: pan[k] for k in ("H", "C", "Co", "Ho")}
    out = common({**base, "Hd": common_rows(naive_panel("t2", pd.read_pickle(DISTILL_PATH)))})
    if not out["H"].index.equals(pan["H"].index):
        raise ValueError("the distilled tone panel drops meetings of the CORAL panels; S+M could not be reused")
    out["Hdp"] = rn.placebo_panel(out["Hd"])
    return out


def _settings_path(arch: str, arm: str):
    """File of the tuned settings for one network and arm."""
    return OUT / f"settings_{arch}_{arm}.json"


def tune(pan: dict, device: str, n_trials: int = rn.N_TRIALS) -> None:
    """Tune both arms of every network at every tuning point (resumes from saved settings)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for arch in rn.ARCHS:
        for arm, (which, cols) in ARMS.items():
            df, path = pan[which], _settings_path(arch, arm)
            done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
            rn._check_dates(done, df, arch, arm, cols)
            fp = rn.panel_fingerprint(df, cols, arch)
            X, ranks = df[cols].to_numpy(float), df["y_class"].to_numpy(int)
            for pt in rn.tuning_points(df.index):
                if pt in done:
                    continue
                t0 = time.time()
                done[pt] = rn.tune_point_coral(X, ranks, pt, arch, device, n_trials) | {
                    "date": str(df.index[pt].date()), "panel": fp}
                rn._write_json(path, done)
                print(f"{arch} {arm} {df.index[pt].date()}: {done[pt]} ({time.time() - t0:.0f}s)", flush=True)


def primary_holm(comp: pd.DataFrame) -> pd.Series:
    """Holm over the three primary tests (S+M+Hd against S+M, scheduled meetings from 2010); NaN elsewhere."""
    prim = comp[(comp["model"] == "S+M+Hd") & (comp["against"] == "S+M") & (comp["period"] == "from2010")
                & (comp["subset"] == "scheduled")]
    adj = holm({i: p for i, p in prim["nw_p"].items()})
    return pd.Series({i: adj.get(i, np.nan) for i in comp.index}, dtype=float)


def reading(comp: pd.DataFrame, period: str = "from2010") -> dict[str, bool]:
    """Placebo reading per network: the gain is from the tone's content when the real tone beats its placebo at
    p < 0.05 and the placebo does not beat S+M at p < 0.05 (scheduled meetings)."""
    c = comp[(comp["period"] == period) & (comp["subset"] == "scheduled")]
    get = lambda a, m, b: float(c[(c["arch"] == a) & (c["model"] == m) & (c["against"] == b)]["nw_p"].iloc[0])
    return {a: bool(get(a, "S+M+Hd", "placebo-Hd") < config.ALPHA and get(a, "placebo-Hd", "S+M") >= config.ALPHA)
            for a in c["arch"].unique()}


def forecast(pan: dict, device: str) -> pd.DataFrame:
    """Forecast both arms (8 seeds, mean levels), compare with the CORAL S+M forecasts, write tables and summary."""
    preds = {}
    for arch in rn.ARCHS:
        base = pd.read_csv(rn.OUT / f"preds_{arch}_S+M.csv", parse_dates=["date"])
        preds[(arch, "S+M")] = base
        for arm, (which, cols) in ARMS.items():
            df = pan[which]
            settings = {int(k): v for k, v in json.loads(_settings_path(arch, arm).read_text()).items()}
            rn._check_dates(settings, df, arch, arm, cols)
            missing = sorted(set(rn.tuning_points(df.index)) - set(settings))
            if missing:
                raise ValueError(f"{arch} {arm}: tuning incomplete, missing tuning points {missing}")
            frames = [rn.seed_forecast(df, cols, arch, settings, seed, device,
                                       OUT / f"preds_{arch}_{arm}_seed{seed}.csv") for seed in rn.SEEDS]
            ens = rn.ensemble(frames)
            if not (ens["date"].to_numpy() == base["date"].to_numpy()).all():
                raise ValueError(f"{arch} {arm}: forecast meetings differ from the CORAL S+M forecasts")
            ens.to_csv(OUT / f"preds_{arch}_{arm}.csv", index=False)
            preds[(arch, arm)] = ens
    rows, comps = [], []
    for (arch, arm), p in preds.items():
        for period, start in PERIODS.items():
            for sub, mask in rn._subsets(rn.T2, pan, p["date"]).items():
                keep = mask & ((p["date"] >= pd.Timestamp(start)).to_numpy() if start else True)
                q = p[keep]
                rows.append({"arch": arch, "features": arm, "period": period, "subset": sub, **metrics(q),
                             "log_loss": float(log_loss(q).mean())})
    for arch in rn.ARCHS:
        for new, old in PAIRS:
            b, n = preds[(arch, old)], preds[(arch, new)]
            for period, start in PERIODS.items():
                for sub, mask in rn._subsets(rn.T2, pan, b["date"]).items():
                    keep = mask & ((b["date"] >= pd.Timestamp(start)).to_numpy() if start else True)
                    comps.append({"arch": arch, "model": new, "against": old, "period": period, "subset": sub,
                                  **rn.pair_stats_coral(b[keep].reset_index(drop=True),
                                                        n[keep].reset_index(drop=True))})
    comp = pd.DataFrame(comps)
    comp["nw_p_holm"] = primary_holm(comp)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    comp.to_csv(OUT / "comparison.csv", index=False)
    prim = comp[comp["nw_p_holm"].notna()]
    summary = {"primary_from2010_scheduled": {r.arch: {"nw_p": r.nw_p, "nw_p_holm": r.nw_p_holm,
                                                       "logloss_diff_mean": r.logloss_diff_mean}
                                              for r in prim.itertuples()},
               "gain_from_tone_content_from2010": reading(comp, "from2010"),
               "gain_from_tone_content_full": reading(comp, "full")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "forecast"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    p = panels()
    tune(p, args.device) if args.step == "tune" else forecast(p, args.device)
