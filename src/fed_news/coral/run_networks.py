"""CORAL STUDY: LSTM, GRU and TFT with a CORAL output layer, tuned per arm, with and without Fed tone (CUDA only).

Step "time": seconds per Optuna trial, projected hours. Step "tune": Optuna searches each network's settings inside
each training window, objective = validation log loss. Step "forecast": walk-forward forecasts with 8 seeds, then
metrics and tests. Saved settings and forecasts are reused, so every step resumes.
Reads:   <data>/t2.pkl, t2_full.pkl, doc_stance.pkl, doc_stance_haiku.pkl, coral/doc_stance_chronobert_coral.pkl,
         coral/doc_stance_haiku_ordinal.pkl; results/study4/neural/preds_*.csv (version 2, for comparison).
Writes:  results/coral/networks/: settings_*.json, preds_*.csv, metrics.csv, comparison.csv, confusion.csv,
         biases.csv, summary.json, timing.json.
Run:     python -m fed_news.coral.run_networks time
         python -m fed_news.coral.run_networks tune [--archs lstm gru tft] [--arms ...]
         python -m fed_news.coral.run_networks forecast

Details: CORAL study, section 7. The version 2 protocol (Study 4 section 12, meetings) with a CORAL head and loss:
tuning points = the first test meeting of 2001, 2005, ..., 2025; 4 shared expanding folds (identical validation rows
for every candidate); 20 TPE trials (seed 0) per architecture, arm and point; at most 300 epochs, patience 20, early
stopping and objective on the validation log loss of the class probabilities; final fit for the median best epoch.
Walk-forward refits every 8 meetings and at each tuning point, 8 seeds; the seeds' level probabilities are averaged,
the predicted class follows the paper's rule, and the rate forecast is the last rate plus the class probabilities
times the training rows' mean move per class. Primary: S+M+Co and S+M+Ho against S+M per architecture, one-sided
Newey-West (lags 8) on the log-loss difference, scheduled meetings, Holm over the 6.
"""
import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from fed_news import config
from fed_news.coral import chronobert_coral, haiku_ordinal
from fed_news.coral.core import biases_ordered, class_probs, predict_rank
from fed_news.coral.networks import build_coral_model, fit_fixed_coral, fit_validate_coral, predict_levels
from fed_news.coral.run_logit import common
from fed_news.evaluation.metrics import metrics
from fed_news.evaluation.significance import dm_hln, holm, mcnemar_greater, nw_summary
from fed_news.features.panels import M_COLS, S_COLS
from fed_news.features.swap_tone import naive_panel
from fed_news.models.linear import move_sizes
from fed_news.models.neural import make_sequences, standardise
from fed_news.models.walk_forward import common_rows, train_mask
from fed_news.studies.networks.run_networks import (ARCHS, BOOT_DRAWS, GROUPS, MAX_EPOCHS, MIN_TRAIN_SEQ, PATIENCE,
                                                    REFIT_EVERY, SEEDS, TASKS, _subsets, class_report, folds_for,
                                                    placebo_panel, point_for, suggest, tuning_points)
from fed_news.studies.simple_model.run_simple_model import log_loss
from fed_news.studies.simple_model.run_simple_model import panels as study4_panels
from fed_news.studies.simple_model.sensitivity_checks import block_bootstrap

N_TRIALS, LOOKBACKS = 20, (2, 4, 8, 12)
D = ["n_stance", "n_stance_chg"]
SM = S_COLS + M_COLS
ARMS = {"S+M": ("H", SM), "S+M+C": ("C", SM + D), "S+M+H": ("H", SM + D), "S+M+Co": ("Co", SM + D),
        "S+M+Ho": ("Ho", SM + D), "placebo-Co": ("Cop", SM + D), "placebo-Ho": ("Hop", SM + D)}
PRIMARY = [("S+M+Co", "S+M"), ("S+M+Ho", "S+M")]
PAIRS = PRIMARY + [("S+M+C", "S+M"), ("S+M+H", "S+M"), ("placebo-Co", "S+M"), ("placebo-Ho", "S+M"),
                   ("S+M+Co", "placebo-Co"), ("S+M+Ho", "placebo-Ho")]
V2_ARMS = ("S+M", "S+M+C", "S+M+H")
OUT = config.RESULTS / "coral" / "networks"
T2 = TASKS["T2"]


def panels(coral: bool = True) -> dict[str, pd.DataFrame]:
    """The Study 4 panels (H, C) and, with coral=True, the CORAL-tone panels (Co, Ho) and their placebos, all on
    the meetings present in every panel."""
    pan = study4_panels()
    if coral:
        pan = common({**pan, "Co": common_rows(naive_panel("t2", pd.read_pickle(chronobert_coral.OUT_PATH))),
                      "Ho": common_rows(naive_panel("t2", pd.read_pickle(haiku_ordinal.OUT_PATH)))})
        pan["Cop"], pan["Hop"] = placebo_panel(pan["Co"]), placebo_panel(pan["Ho"])
    return pan


def cv_score_coral(X, ranks, point: int, arch: str, p: dict, device: str, max_lookback: int,
                   max_epochs: int = MAX_EPOCHS, seed: int = 0) -> tuple[float, int]:
    """Mean validation log loss over the shared folds and the median best epoch; infinite if any fold has fewer
    than MIN_TRAIN_SEQ training sequences for this candidate's lookback."""
    lb = p["lookback"]
    vals, epochs = [], []
    for train_end, va in folds_for(point, 0, max_lookback):
        tr = np.arange(lb - 1, train_end)
        if len(tr) < MIN_TRAIN_SEQ:
            return np.inf, 1
        mask = np.zeros(len(X), bool)
        mask[tr] = True
        Z = standardise(X, mask)
        model = build_coral_model(arch, X.shape[1], p["hidden"], p["dropout"], seed, layers=p.get("layers", 1),
                                  heads=p.get("heads", 4))
        v, e = fit_validate_coral(model, make_sequences(Z, tr, lb), ranks[tr], make_sequences(Z, va, lb), ranks[va],
                                  p["lr"], p["weight_decay"], p["batch"], max_epochs, PATIENCE, device, seed)
        vals.append(v)
        epochs.append(e)
    return float(np.mean(vals)), max(1, int(np.median(epochs)))


def tune_point_coral(X, ranks, point: int, arch: str, device: str, n_trials: int = N_TRIALS) -> dict:
    """Optuna (TPE, seed 0) at one tuning point; the best settings, their epochs and validation log loss."""
    def objective(trial):
        """Validation log loss of one candidate."""
        p = suggest(trial, arch, LOOKBACKS)
        score, epochs = cv_score_coral(X, ranks, point, arch, p, device, max(LOOKBACKS))
        trial.set_user_attr("epochs", epochs)
        return score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    n_feasible = sum(bool(np.isfinite(t.value)) for t in study.trials if t.value is not None)
    return {**best.params, "epochs": int(best.user_attrs["epochs"]), "cv_logloss": float(best.value),
            "n_feasible_trials": int(n_feasible)}


def run_tuned_coral(df: pd.DataFrame, cols, arch: str, settings: dict, seed: int, device: str,
                    test_start) -> pd.DataFrame:
    """Walk-forward forecasts of one CORAL network with the tuned settings and one seed."""
    X = df[cols].to_numpy(float)
    y_change, lag, y_level = (df[c].to_numpy(float) for c in ("y_change", "s_rate_lag", "y_level"))
    ranks, dates, n = df["y_class"].to_numpy(int), df.index, len(df)
    test, points = np.flatnonzero(dates >= pd.Timestamp(test_start)), sorted(settings)
    out, model, current, sizes, ordered = [], None, None, None, None
    for k, i in enumerate(test):
        pt = point_for(i, points)
        if k % REFIT_EVERY == 0 or pt != current:
            current, p = pt, settings[pt]
            train = train_mask(n, i, "T2")
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            rows = rows[rows >= p["lookback"] - 1]
            model = build_coral_model(arch, len(cols), p["hidden"], p["dropout"], seed, layers=p.get("layers", 1),
                                      heads=p.get("heads", 4))
            fit_fixed_coral(model, make_sequences(Z, rows, p["lookback"]), ranks[rows], p["lr"], p["epochs"], device,
                            batch_size=p["batch"], seed=seed, weight_decay=p["weight_decay"])
            sizes, ordered = move_sizes(ranks[rows], y_change[rows]), biases_ordered(model.head)
        lp = predict_levels(model, make_sequences(Z, np.array([i]), p["lookback"]), device)
        if not np.isfinite(lp).all():
            raise ValueError(f"{arch}: non-finite forecast at {dates[i].date()}")
        probs = class_probs(lp)[0]
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + float(probs @ sizes), "lag": lag[i],
                    "class_true": int(ranks[i]), "class_pred": int(predict_rank(lp)[0]),
                    "lp1": float(lp[0, 0]), "lp2": float(lp[0, 1]), "p_cut": probs[0], "p_hold": probs[1],
                    "p_hike": probs[2], "size_cut": sizes[0], "size_hike": sizes[2], "biases_ordered": ordered})
    return pd.DataFrame(out)


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Seed-mean level probabilities, then class probabilities, the paper's class and the rate forecast."""
    ens = frames[0][["date", "y_true", "lag", "class_true", "size_cut", "size_hike"]].copy()
    lp = np.mean([f[["lp1", "lp2"]].to_numpy(float) for f in frames], axis=0)
    p = class_probs(lp)
    ens["lp1"], ens["lp2"] = lp[:, 0], lp[:, 1]
    ens["p_cut"], ens["p_hold"], ens["p_hike"] = p[:, 0], p[:, 1], p[:, 2]
    ens["class_pred"] = predict_rank(lp)
    ens["y_pred"] = ens["lag"] + p[:, 0] * ens["size_cut"] + p[:, 2] * ens["size_hike"]
    return ens


def pair_stats_coral(base: pd.DataFrame, new: pd.DataFrame, block: int = REFIT_EVERY) -> dict:
    """Log-loss difference (base minus new; positive favours new) with a Newey-West test and a moving-block
    bootstrap; the same for squared rate errors; accuracy difference; DM-HLN on log loss; McNemar counts."""
    ll_b, ll_n = log_loss(base), log_loss(new)
    y = base["y_true"].to_numpy(float)
    e_b, e_n = (y - base["y_pred"].to_numpy(float)) ** 2, (y - new["y_pred"].to_numpy(float)) ** 2
    ok_b = base["class_pred"].to_numpy() == base["class_true"].to_numpy()
    ok_n = new["class_pred"].to_numpy() == new["class_true"].to_numpy()
    nw, sq = nw_summary(ll_b - ll_n, lags=block), nw_summary(e_b - e_n, lags=block)
    fixes, breaks, p_mc = mcnemar_greater(ok_b, ok_n)
    return {"n": int(len(y)), "logloss_diff_mean": nw["mean"], "nw_p": nw["p_one_sided"], "logloss_ci90": nw["ci90"],
            "logloss_boot_ci90": block_bootstrap(ll_b - ll_n, block, BOOT_DRAWS, 0)["ci90"],
            "sq_diff_mean": sq["mean"], "sq_nw_p": sq["p_one_sided"],
            "sq_boot_ci90": block_bootstrap(e_b - e_n, block, BOOT_DRAWS, 0)["ci90"],
            "acc_diff": float(ok_n.mean() - ok_b.mean()),
            "acc_diff_ci90": block_bootstrap(ok_n.astype(float) - ok_b.astype(float), block, BOOT_DRAWS, 0)["ci90"],
            "dm_hln_logloss_p": dm_hln(ll_b, ll_n)[1], "mcnemar_fixes": fixes, "mcnemar_breaks": breaks,
            "mcnemar_p": p_mc}


def registered_holm_coral(comp: pd.DataFrame) -> list:
    """Holm over the registered family (3 architectures x 2 primary pairs, scheduled meetings); NaN for every row
    while any member is missing."""
    family = [(arch, new, base) for arch in ARCHS for new, base in PRIMARY]
    prim = comp[(comp["role"] == "primary") & (comp["subset"] == "scheduled")]
    found = {(r.arch, r.model, r.against): r.nw_p for r in prim.itertuples()}
    if any(k not in found for k in family):
        return [np.nan] * len(comp)
    adj = holm({f"{a} {new}": found[(a, new, base)] for a, new, base in family})
    return [adj[f"{r.arch} {r.model}"] if r.role == "primary" and r.subset == "scheduled" else np.nan
            for r in comp.itertuples()]


def _settings_path(arch: str, arm: str):
    """File of the tuned settings for one network and arm."""
    return OUT / f"settings_{arch}_{arm}.json"


def _load_settings(arch: str, arm: str, df: pd.DataFrame) -> dict:
    """Saved settings keyed by row index, checked against the panel's dates (the panel must not have changed)."""
    settings = {int(k): v for k, v in json.loads(_settings_path(arch, arm).read_text()).items()}
    for pt, s in settings.items():
        if str(df.index[pt].date()) != s["date"]:
            raise ValueError(f"{arch} {arm}: tuning point {pt} is {df.index[pt].date()} now, {s['date']} when tuned")
    return settings


def tune_all(pan: dict, device: str, n_trials: int = N_TRIALS, archs=ARCHS, arms=None) -> None:
    """Tune every network and arm at every tuning point (resumes from saved settings)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        for arm in arms or tuple(ARMS):
            which, cols = ARMS[arm]
            df = pan[which]
            path = _settings_path(arch, arm)
            done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
            X, ranks = df[cols].to_numpy(float), df["y_class"].to_numpy(int)
            for pt in tuning_points(df.index):
                if pt in done:
                    continue
                t0 = time.time()
                done[pt] = tune_point_coral(X, ranks, pt, arch, device, n_trials) | {"date": str(df.index[pt].date())}
                path.write_text(json.dumps(done, indent=2))
                print(f"{arch} {arm} {df.index[pt].date()}: {done[pt]} ({time.time() - t0:.0f}s)", flush=True)


def time_trial(pan: dict, device: str, archs=ARCHS) -> dict:
    """Seconds per trial at the last tuning point (the most rows) and the projected hours for every arm."""
    df = pan["C"]
    X, ranks = df[SM + D].to_numpy(float), df["y_class"].to_numpy(int)
    pts = tuning_points(df.index)
    secs = {}
    for arch in archs:
        t0 = time.time()
        tune_point_coral(X, ranks, pts[-1], arch, device, 3)
        secs[arch] = (time.time() - t0) / 3
    hours = {a: s * N_TRIALS * len(pts) * len(ARMS) / 3600 for a, s in secs.items()}
    res = {"device": device, "seconds_per_trial_at_last_point": secs, "hours_upper_bound_per_arch": hours,
           "hours_upper_bound_total_sequential": float(sum(hours.values())),
           "ask_user": bool(sum(hours.values()) > 36)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def _v2_rows(preds: dict, pan: dict, archs) -> list[dict]:
    """Descriptive: each CORAL arm against the version 2 MSE network of the same arm (squared errors, accuracy)."""
    rows = []
    for arch in archs:
        for arm in V2_ARMS:
            path = config.RESULTS / "study4" / "neural" / f"preds_{arch}_{arm}.csv"
            if (arch, arm) not in preds or not path.exists():
                continue
            old = pd.read_csv(path, parse_dates=["date"])
            new = preds[(arch, arm)]
            old = old.set_index("date").loc[new["date"]].reset_index()
            y = new["y_true"].to_numpy(float)
            for sub, mask in _subsets(T2, pan, new["date"]).items():
                e_o, e_n = ((y - old["y_pred"].to_numpy(float)) ** 2)[mask], ((y - new["y_pred"].to_numpy(float)) ** 2)[mask]
                ok_o = (old["class_pred"] == old["class_true"]).to_numpy()[mask]
                ok_n = (new["class_pred"] == new["class_true"]).to_numpy()[mask]
                nw = nw_summary(e_o - e_n, lags=REFIT_EVERY)
                rows.append({"arch": arch, "model": f"{arm} CORAL", "against": f"{arm} version 2 (MSE)",
                             "role": "vs_v2", "subset": sub, "n": int(mask.sum()), "sq_diff_mean": nw["mean"],
                             "sq_nw_p": nw["p_one_sided"], "acc_diff": float(ok_n.mean() - ok_o.mean())})
    return rows


def forecast_all(pan: dict, device: str, archs=ARCHS) -> pd.DataFrame:
    """Forecast every tuned network and arm (8 seeds, mean levels), then write metrics, tests and summaries."""
    preds, bias_rows = {}, []
    for arch in archs:
        for arm, (which, cols) in ARMS.items():
            if not _settings_path(arch, arm).exists():
                continue
            df = pan[which]
            settings = _load_settings(arch, arm, df)
            frames = []
            for seed in SEEDS:
                path = OUT / f"preds_{arch}_{arm}_seed{seed}.csv"
                if not path.exists():
                    run_tuned_coral(df, cols, arch, settings, seed, device, config.TEST_START).to_csv(path, index=False)
                f = pd.read_csv(path, parse_dates=["date"])
                frames.append(f)
                bias_rows.append({"arch": arch, "arm": arm, "seed": seed,
                                  "n_rows_biases_unordered": int((~f["biases_ordered"].astype(bool)).sum())})
            ens = ensemble(frames)
            ens.to_csv(OUT / f"preds_{arch}_{arm}.csv", index=False)
            preds[(arch, arm)] = ens
    rows, comps, confusions = [], [], []
    for (arch, arm), p in preds.items():
        for sub, mask in _subsets(T2, pan, p["date"]).items():
            q = p[mask]
            rows.append({"arch": arch, "features": arm, "subset": sub, **metrics(q),
                         "log_loss": float(log_loss(q).mean()), **class_report(q)})
            if sub == "scheduled":
                cm = confusion_matrix(q["class_true"], q["class_pred"], labels=[0, 1, 2])
                confusions += [{"arch": arch, "features": arm, "true": GROUPS[i],
                                **{f"pred_{GROUPS[j]}": int(cm[i, j]) for j in range(3)}} for i in range(3)]
    for arch in archs:
        for new, base in PAIRS:
            if (arch, new) not in preds or (arch, base) not in preds:
                continue
            role = ("primary" if (new, base) in PRIMARY else "placebo" if "placebo" in new + base else "old_tone")
            b, n = preds[(arch, base)], preds[(arch, new)]
            for sub, mask in _subsets(T2, pan, b["date"]).items():
                comps.append({"arch": arch, "model": new, "against": base, "role": role, "subset": sub,
                              **pair_stats_coral(b[mask].reset_index(drop=True), n[mask].reset_index(drop=True))})
    comp = pd.DataFrame(comps)
    if len(comp):
        comp["nw_p_holm"] = registered_holm_coral(comp)
    comp = pd.concat([comp, pd.DataFrame(_v2_rows(preds, pan, archs))], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    comp.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(confusions).to_csv(OUT / "confusion.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(OUT / "biases.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(_summary(comp), indent=2, default=str))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(comp.drop(columns=[c for c in comp.columns if c.endswith("ci90")]).round(4).to_string(index=False), flush=True)
    return comp


def _summary(comp: pd.DataFrame) -> dict:
    """Primary results with Holm, and the placebo reading (Study 4 section 10) for every architecture and tone."""
    sched = comp[comp["subset"] == "scheduled"]
    get = lambda a, m, b: sched[(sched["arch"] == a) & (sched["model"] == m) & (sched["against"] == b)]
    out = {}
    for arch in ARCHS:
        for tone in ("Co", "Ho"):
            prim, real_vs_pl, pl_vs_sm = get(arch, f"S+M+{tone}", "S+M"), get(arch, f"S+M+{tone}", f"placebo-{tone}"), \
                get(arch, f"placebo-{tone}", "S+M")
            if prim.empty:
                continue
            entry = {"nw_p": float(prim["nw_p"].iloc[0]), "nw_p_holm": float(prim["nw_p_holm"].iloc[0]),
                     "logloss_diff_mean": float(prim["logloss_diff_mean"].iloc[0])}
            if not real_vs_pl.empty and not pl_vs_sm.empty:
                entry["gain_from_tone_content"] = bool(real_vs_pl["nw_p"].iloc[0] < config.ALPHA
                                                       and pl_vs_sm["nw_p"].iloc[0] >= config.ALPHA)
            out[f"{arch} S+M+{tone}"] = entry
    return out


def main(argv=None) -> None:
    """Command line: time, tune or forecast. Training runs only on a CUDA GPU."""
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["time", "tune", "forecast"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    args = ap.parse_args(argv)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required: the CORAL networks train only on the GPU (--device cuda)")
    if args.step == "time":
        time_trial(panels(coral=False), args.device, args.archs)
    elif args.step == "tune":
        tune_all(panels(), args.device, args.trials, args.archs, args.arms)
    else:
        forecast_all(panels(), args.device, args.archs)


if __name__ == "__main__":
    main()
