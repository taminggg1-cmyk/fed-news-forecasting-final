"""Attempt 14, declared secondary: the CORAL LSTM, GRU and TFT with the distilled tone split by document type ("Hs").

Two new arms per network (S+M+Hs and its placebo), tuned and forecast exactly as in attempt 12b; the S+M arm is reused
from the CORAL study and the single-tone arm S+M+Hd from attempt 12b (same meetings, checked). Secondary test:
one-sided Newey-West (8 lags) on the log-loss difference, S+M+Hs against S+M, scheduled meetings from 2010-01-01,
Holm over the 3 networks; placebo reading. Reported without a rule: the full period, S+M+Hs against S+M+Hd, and the
small-sample checks (fixed-b, fixed-m, ranked probability score).
Reads:   the CORAL panels and <data>/distill/doc_stance_distill.pkl; results/coral/networks/preds_<arch>_S+M.csv,
         results/distill/networks/preds_<arch>_S+M+Hd.csv.
Writes:  results/doctype/networks/settings_*.json, preds_*.csv, metrics.csv, comparison.csv, small_sample.csv,
         summary.json.
Run:     python -m fed_news.studies.doctype.run_networks tune      (CUDA)
         python -m fed_news.studies.doctype.run_networks forecast  (CUDA)

Details: docs/studies/5_doctype/2026-10-21-doctype-tone-design.md, sections 2-4; CORAL study section 7 for the
networks.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.evaluation.metrics import metrics
from fed_news.evaluation.significance import holm
from fed_news.models.walk_forward import common_rows
from fed_news.studies.coral import run_networks as rn
from fed_news.studies.coral.from2010 import START
from fed_news.studies.coral.run_logit import common
from fed_news.studies.distill import run_networks as dn
from fed_news.studies.distill.chronobert_distill import OUT_PATH as DISTILL_PATH
from fed_news.studies.doctype.group_tone import HS, split_panel
from fed_news.studies.sigtests import tests as st
from fed_news.studies.simple_model.run_simple_model import log_loss

ARMS = {"S+M+Hs": ("Hs", rn.SM + HS), "placebo-Hs": ("Hsp", rn.SM + HS)}
PAIRS = [("S+M+Hs", "S+M"), ("placebo-Hs", "S+M"), ("S+M+Hs", "placebo-Hs"), ("S+M+Hs", "S+M+Hd")]
REUSED = {"S+M": rn.OUT, "S+M+Hd": dn.OUT}
OUT = config.RESULTS / "doctype" / "networks"
PERIODS = {"from2010": START, "full": None}


def placebo(df: pd.DataFrame) -> pd.DataFrame:
    """The panel with the 6 Hs columns shifted by half its length (the placebo used by every earlier study)."""
    return rn.placebo_panel(df, cols=HS)


def panels() -> dict[str, pd.DataFrame]:
    """The CORAL panels plus the split-tone panel and its placebo, all on the same meetings."""
    pan = rn.panels()
    base = {k: pan[k] for k in ("H", "C", "Co", "Ho")}
    out = common({**base, "Hs": common_rows(split_panel(pd.read_pickle(DISTILL_PATH)))})
    if not out["H"].index.equals(pan["H"].index):
        raise ValueError("the split-tone panel drops meetings of the CORAL panels; S+M could not be reused")
    if out["Hs"][HS].isna().any().any():
        raise ValueError("missing split-tone values on the CORAL meetings")
    out["Hsp"] = placebo(out["Hs"])
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
    """Holm over the three secondary tests (S+M+Hs against S+M, scheduled meetings from 2010); NaN elsewhere."""
    prim = comp[(comp["model"] == "S+M+Hs") & (comp["against"] == "S+M") & (comp["period"] == "from2010")
                & (comp["subset"] == "scheduled")]
    adj = holm({i: p for i, p in prim["nw_p"].items()})
    return pd.Series({i: adj.get(i, np.nan) for i in comp.index}, dtype=float)


def reading(comp: pd.DataFrame, period: str = "from2010") -> dict[str, bool]:
    """Placebo reading per network: the gain is from the tone's content when the real tone beats its placebo at
    p < 0.05 and the placebo does not beat S+M at p < 0.05 (scheduled meetings)."""
    c = comp[(comp["period"] == period) & (comp["subset"] == "scheduled")]
    get = lambda a, m, b: float(c[(c["arch"] == a) & (c["model"] == m) & (c["against"] == b)]["nw_p"].iloc[0])
    return {a: bool(get(a, "S+M+Hs", "placebo-Hs") < config.ALPHA and get(a, "placebo-Hs", "S+M") >= config.ALPHA)
            for a in c["arch"].unique()}


def small_sample(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Fixed-b and fixed-m on the log-loss difference; Newey-West and fixed-b on the ranked probability score
    difference (positive differences favour `new`; one-sided p)."""
    d_ll, d_rps = log_loss(base) - log_loss(new), st.rps(base) - st.rps(new)
    return {"fixed_b_p": st.fixed_b(d_ll)["p_one_sided"], "fixed_m_p": st.fixed_m(d_ll)["p_one_sided"],
            "rps_nw_p": st.newey_west(d_rps)["p_one_sided"], "rps_fixed_b_p": st.fixed_b(d_rps)["p_one_sided"]}


def _keep(dates: pd.Series, start, sub_mask) -> np.ndarray:
    """Rows of the subset that are dated `start` or later (all rows if start is None)."""
    return sub_mask & ((dates >= pd.Timestamp(start)).to_numpy() if start else True)


def forecast(pan: dict, device: str) -> pd.DataFrame:
    """Forecast both arms (8 seeds, mean levels), compare with the reused S+M and S+M+Hd forecasts, write tables and
    the summary."""
    preds = {}
    for arch in rn.ARCHS:
        for arm, folder in REUSED.items():
            preds[(arch, arm)] = pd.read_csv(folder / f"preds_{arch}_{arm}.csv", parse_dates=["date"])
        base = preds[(arch, "S+M")]
        if not (preds[(arch, "S+M+Hd")]["date"].to_numpy() == base["date"].to_numpy()).all():
            raise ValueError(f"{arch}: the S+M+Hd forecasts are not on the CORAL S+M meetings")
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
    rows, comps, small = [], [], []
    for (arch, arm), p in preds.items():
        for period, start in PERIODS.items():
            for sub, mask in rn._subsets(rn.T2, pan, p["date"]).items():
                q = p[_keep(p["date"], start, mask)]
                rows.append({"arch": arch, "features": arm, "period": period, "subset": sub, **metrics(q),
                             "log_loss": float(log_loss(q).mean())})
    for arch in rn.ARCHS:
        for new, old in PAIRS:
            b, n = preds[(arch, old)], preds[(arch, new)]
            for period, start in PERIODS.items():
                for sub, mask in rn._subsets(rn.T2, pan, b["date"]).items():
                    keep = _keep(b["date"], start, mask)
                    bk, nk = b[keep].reset_index(drop=True), n[keep].reset_index(drop=True)
                    comps.append({"arch": arch, "model": new, "against": old, "period": period, "subset": sub,
                                  **rn.pair_stats_coral(bk, nk)})
                    if (new, old) == ("S+M+Hs", "S+M") and sub == "scheduled":
                        small.append({"arch": arch, "period": period, "n": len(bk), **small_sample(bk, nk)})
    comp = pd.DataFrame(comps)
    comp["nw_p_holm"] = primary_holm(comp)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    comp.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(small).to_csv(OUT / "small_sample.csv", index=False)
    prim = comp[comp["nw_p_holm"].notna()]
    sel = lambda m, a: comp[(comp["model"] == m) & (comp["against"] == a) & (comp["period"] == "from2010")
                            & (comp["subset"] == "scheduled")].set_index("arch")["nw_p"].to_dict()
    summary = {"secondary_from2010_scheduled": {r.arch: {"nw_p": r.nw_p, "nw_p_holm": r.nw_p_holm,
                                                         "logloss_diff_mean": r.logloss_diff_mean}
                                                for r in prim.itertuples()},
               "gain_from_tone_content_from2010": reading(comp, "from2010"),
               "gain_from_tone_content_full": reading(comp, "full"),
               "split_vs_single_nw_p_from2010": sel("S+M+Hs", "S+M+Hd")}
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
