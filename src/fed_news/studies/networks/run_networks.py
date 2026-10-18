"""STUDY 4 VERSION 2 MAIN RUN: LSTM, GRU and TFT, tuned, with and without Fed tone.

Step "tune": Optuna searches each network's settings inside each training window (the committed runs used --trials 20;
docs/DEVIATIONS.md). Step "forecast": walk-forward forecasts with 8 seeds, then metrics and tests. Forecast files that
already exist are reused, so "forecast" on the saved files only recomputes the tables.
Reads:   <data>/t2.pkl, t2_full.pkl (meetings) or t1b.pkl, t1b_full.pkl (months), doc_stance.pkl, doc_stance_haiku.pkl,
         <data>/alfred/, results/inputs_from_earlier/paper_settings/.
Writes:  results/study4/neural/ (meetings, --task T2) or results/study4/neural_t1/ (months, --task T1):
         settings_*.json, preds_*.csv, metrics_neural_*.csv, comparison_neural_*.csv.
Run:     python -m fed_news.studies.networks.run_networks tune --task T2 --trials 20
         python -m fed_news.studies.networks.run_networks forecast --task T2 --device cpu

Details: Study 4, sections 5, 10-12: LSTM, GRU and a TFT-inspired network tuned on our own data (secondary)
(docs/design/2026-10-12-study4-haiku-index-tuned-design.md; version 2 protocol, section 12).

At each tuning point (the first test row of 2001, 2005, ..., 2025) Optuna's TPE sampler searches the settings
on rows before that point only. Every candidate is scored on the same 4 expanding validation folds (section
12.2), in rate units (12.3); candidates without 10 training sequences in every fold are infeasible. The chosen
settings forecast every test row until the next point: walk-forward refits every 8 meetings or 12 months (and
at each point), 8 seeds, seed-mean forecasts. Every feature set gets the same search space and budget.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from fed_news import config
from fed_news.models import linear as models
from fed_news.evaluation.metrics import metrics
from fed_news.models.neural import build_model, fit_fixed, fit_validate, make_sequences, predict
from fed_news.models.neural import standardise
from fed_news.features.panels import M_COLS, S_COLS
from fed_news.evaluation.significance import dm_hln, holm, mcnemar_greater, nw_summary
from fed_news.studies.simple_model.sensitivity_checks import block_bootstrap
from fed_news.features.strict_columns import B as STRICT_T2, KEPT_SERIES, REALTIME_S as STRICT_S, check_vintages
from fed_news.models.walk_forward import train_mask

ARCHS = ("lstm", "gru", "tft")
YEARS = (2001, 2005, 2009, 2013, 2017, 2021, 2025)
N_TRIALS, N_FOLDS, MAX_EPOCHS, PATIENCE, REFIT_EVERY, SEEDS = 30, 4, 300, 20, 8, tuple(range(8))
MIN_TRAIN_SEQ, BOOT_DRAWS = 10, 10_000
D = ["n_stance", "n_stance_chg"]
FEATURES = {"S+M": ("H", S_COLS + M_COLS), "S+M+H": ("H", S_COLS + M_COLS + D), "S+M+C": ("C", S_COLS + M_COLS + D),
            # section 10: the same columns with the stance shifted circularly by half the panel
            "placebo-C": ("Cp", S_COLS + M_COLS + D), "placebo-H": ("Hp", S_COLS + M_COLS + D),
            # section 12.4: only series whose vintages cover every row, no fallback value anywhere
            "strict": ("C", STRICT_T2), "strict+C": ("C", STRICT_T2 + D), "strict+H": ("H", STRICT_T2 + D)}
PAIRS = [("S+M+C", "S+M"), ("S+M+H", "S+M"), ("strict+C", "strict"), ("strict+H", "strict"),
         ("placebo-C", "S+M"), ("placebo-H", "S+M"), ("S+M+C", "placebo-C"), ("S+M+H", "placebo-H")]
OUT = config.RESULTS / "study4" / "neural"
# section 11: the monthly task with Plan 2's inputs (the paper's Table 1 statistics) and the stance columns
FEATURES_T1 = {"S": ("C", S_COLS), "S+C": ("C", S_COLS + D), "S+H": ("H", S_COLS + D),
               "placebo-C": ("Cp", S_COLS + D), "placebo-H": ("Hp", S_COLS + D),
               "strict": ("C", STRICT_S), "strict+C": ("C", STRICT_S + D), "strict+H": ("H", STRICT_S + D)}
PAIRS_T1 = [("S+C", "S"), ("S+H", "S"), ("strict+C", "strict"), ("strict+H", "strict"),
            ("placebo-C", "S"), ("placebo-H", "S"), ("S+C", "placebo-C"), ("S+H", "placebo-H")]
GROUPS = {0: "cut", 1: "hold", 2: "hike"}


@dataclass(frozen=True)
class TaskCfg:
    """Settings of one task: panel, output folder, feature sets, comparisons, lookbacks, refit interval."""
    name: str
    panel: str
    out: Path
    features: dict
    pairs: list
    primary: tuple
    lookbacks: tuple
    refit_every: int
    gap: int  # rows before a tuning point that are not yet known (monthly: month t-1, the t-2 rule)
    base: str


TASKS = {"T2": TaskCfg("T2", "t2", OUT, FEATURES, PAIRS, (("S+M+C", "S+M"), ("S+M+H", "S+M")), (2, 4, 8, 12),
                       REFIT_EVERY, 0, "S+M"),
         "T1": TaskCfg("T1", "t1b", config.RESULTS / "study4" / "neural_t1", FEATURES_T1, PAIRS_T1,
                       (("S+C", "S"), ("S+H", "S")), (3, 6, 12, 24), 12, 1, "S")}


def placebo_panel(df: pd.DataFrame, cols=D) -> pd.DataFrame:
    """The panel with the tone columns shifted by half its length (placebo)."""
    out = df.copy()
    for c in cols:
        out[c] = np.roll(df[c].to_numpy(), len(df) // 2)
    return out


def tuning_points(dates, years=YEARS, test_start=config.TEST_START) -> list[int]:
    """Row index of the first test row in each tuning year (2001, 2005, ..., 2025)."""
    dates = pd.DatetimeIndex(dates)
    out = []
    for y in years:
        start = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(test_start))
        hits = np.flatnonzero(dates >= start)
        if len(hits) and int(hits[0]) not in out:
            out.append(int(hits[0]))
    return out


def point_for(i: int, points) -> int:
    """The latest tuning point at or before row i."""
    return max(p for p in points if p <= i)


def suggest(trial, arch: str, lookbacks=(2, 4, 8, 12)) -> dict:
    """One Optuna candidate: hidden size, dropout, learning rate, weight decay, lookback, batch, layers or heads."""
    p = {"hidden": trial.suggest_categorical("hidden", [8, 16, 32, 64] if arch == "tft" else [8, 16, 32, 64, 128, 256]),
         "dropout": trial.suggest_float("dropout", 0.0, 0.5),
         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "lookback": trial.suggest_categorical("lookback", list(lookbacks)),
         "batch": trial.suggest_categorical("batch", [16, 32, 64])}
    if arch == "tft":
        p["heads"] = trial.suggest_categorical("heads", [1, 2, 4])
    else:
        p["layers"] = trial.suggest_categorical("layers", [1, 2])
    return p


def _model(arch: str, n_features: int, p: dict, seed: int):
    """Build a network from a settings dict."""
    return build_model(arch, n_features, p["hidden"], p["dropout"], seed, layers=p.get("layers", 1),
                       heads=p.get("heads", 4))


def _target(y, rows):
    """Mean and standard deviation of the target on the training rows."""
    mu, sd = float(y[rows].mean()), float(y[rows].std())
    return mu, sd if sd > 0 else 1.0


def folds_for(point: int, gap: int = 0, max_lookback: int = 1, n_folds: int = N_FOLDS) -> list:
    """Section 12.2: (training end, validation rows) per fold, identical for every candidate lookback. The rows
    every lookback can use, before the point (minus the gap), are cut into 5 blocks; fold k validates block k+1
    and trains on rows before it minus the gap."""
    eligible = np.arange(max_lookback - 1, point - gap)
    return [(int(eligible[va][0]) - gap, eligible[va]) for _, va in models.expanding_folds(len(eligible), n_folds)]


def cv_score(X, y_change, point: int, arch: str, p: dict, device: str, max_epochs: int = MAX_EPOCHS,
             seed: int = 0, gap: int = 0, max_lookback: int | None = None) -> tuple[float, int]:
    """Mean validation MSE in rate units over the shared folds, and the median best epoch; infinite if any fold
    has fewer than MIN_TRAIN_SEQ training sequences for this candidate's lookback."""
    lb = p["lookback"]
    vals, epochs = [], []
    for train_end, va in folds_for(point, gap, max_lookback or lb):
        tr = np.arange(lb - 1, train_end)
        if len(tr) < MIN_TRAIN_SEQ:
            return np.inf, 1
        mask = np.zeros(len(X), bool)
        mask[tr] = True
        Z = standardise(X, mask)
        mu, sd = _target(y_change, tr)
        model = _model(arch, X.shape[1], p, seed)
        v, e = fit_validate(model, make_sequences(Z, tr, lb), (y_change[tr] - mu) / sd, make_sequences(Z, va, lb),
                            (y_change[va] - mu) / sd, p["lr"], p["weight_decay"], p["batch"], max_epochs, PATIENCE,
                            device, seed)
        vals.append(v * sd ** 2)  # standardised MSE back to rate units
        epochs.append(e)
    return float(np.mean(vals)), max(1, int(np.median(epochs)))


def tune_point(X, y_change, point: int, arch: str, device: str, n_trials: int = N_TRIALS, lookbacks=(2, 4, 8, 12),
               gap: int = 0) -> dict:
    """Run Optuna at one tuning point; return the best settings and their validation score."""
    def objective(trial):
        """Validation score of one candidate."""
        p = suggest(trial, arch, lookbacks)
        score, epochs = cv_score(X, y_change, point, arch, p, device, gap=gap, max_lookback=max(lookbacks))
        trial.set_user_attr("epochs", epochs)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    n_feasible = sum(bool(np.isfinite(t.value)) for t in study.trials if t.value is not None)
    return {**best.params, "epochs": int(best.user_attrs["epochs"]), "cv_mse": float(best.value),
            "n_feasible_trials": int(n_feasible)}


def run_tuned(df: pd.DataFrame, cols, arch: str, settings: dict, seed: int, device: str, test_start,
              on_refit=None, task: str = "T2", refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """Walk-forward forecasts of one network with the tuned settings and one seed."""
    X = df[cols].to_numpy(float)
    y_change, lag, y_level = (df[c].to_numpy(float) for c in ("y_change", "s_rate_lag", "y_level"))
    y_class, dates, n = df["y_class"].to_numpy(int), df.index, len(df)
    test, points = np.flatnonzero(dates >= pd.Timestamp(test_start)), sorted(settings)
    out, model, current = [], None, None
    for k, i in enumerate(test):
        pt = point_for(i, points)
        if k % refit_every == 0 or pt != current:
            current, p = pt, settings[pt]
            train = train_mask(n, i, task)
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            rows = rows[rows >= p["lookback"] - 1]
            mu, sd = _target(y_change, rows)
            model = _model(arch, len(cols), p, seed)
            fit_fixed(model, make_sequences(Z, rows, p["lookback"]), (y_change[rows] - mu) / sd, p["lr"],
                      p["epochs"], device, batch_size=p["batch"], seed=seed, weight_decay=p["weight_decay"])
            if on_refit:
                on_refit(i, p)
        change = float(predict(model, make_sequences(Z, np.array([i]), p["lookback"]), device)[0]) * sd + mu
        if not np.isfinite(change):
            raise ValueError(f"{arch}: non-finite forecast at {dates[i].date()}")
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(y_class[i]),
                    "class_pred": int(models.classes_from_change(np.array([change]))[0])})
    return pd.DataFrame(out)


def _settings_path(arch, fs, out=OUT):
    """File of the tuned settings for one network and feature set."""
    return out / f"settings_{arch}_{fs}.json"


def tune_all(pan: dict, device: str, n_trials: int = N_TRIALS, archs=ARCHS, features=None,
             cfg: TaskCfg = TASKS["T2"]) -> None:
    """Tune every network and feature set at every tuning point (resumes from saved settings)."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        for fs in features or tuple(cfg.features):
            which, cols = cfg.features[fs]
            df = pan[which]
            path = _settings_path(arch, fs, cfg.out)
            done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
            X, y = df[cols].to_numpy(float), df["y_change"].to_numpy(float)
            for pt in tuning_points(df.index):
                if pt in done:
                    continue
                t0 = time.time()
                done[pt] = (tune_point(X, y, pt, arch, device, n_trials, cfg.lookbacks, cfg.gap)
                            | {"date": str(df.index[pt].date())})
                path.write_text(json.dumps(done, indent=2))
                print(f"{cfg.name} {arch} {fs} {df.index[pt].date()}: {done[pt]} ({time.time() - t0:.0f}s)", flush=True)


def time_trial(pan: dict, device: str, cfg: TaskCfg, archs=ARCHS) -> dict:
    """Seconds per trial at the last tuning point (the most rows), per architecture: an upper bound per arm."""
    df = pan["C"]
    X, y = df[cfg.features[cfg.base][1] + D].to_numpy(float), df["y_change"].to_numpy(float)
    pts = tuning_points(df.index)
    secs = {}
    for arch in archs:
        t0 = time.time()
        tune_point(X, y, pts[-1], arch, device, 3, cfg.lookbacks, cfg.gap)
        secs[arch] = (time.time() - t0) / 3
    res = {"task": cfg.name, "device": device, "seconds_per_trial_at_last_point": secs,
           "hours_per_arm_upper_bound": {a: s * N_TRIALS * len(pts) / 3600 for a, s in secs.items()},
           "arms_per_architecture": len(cfg.features)}
    print(json.dumps(res, indent=2), flush=True)
    return res


def class_report(p: pd.DataFrame) -> dict:
    """Accuracy, balanced accuracy and recall per class."""
    t, c = p["class_true"].to_numpy(int), p["class_pred"].to_numpy(int)
    rec = recall_score(t, c, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float((t == c).mean()), "balanced_accuracy": float(balanced_accuracy_score(t, c)),
            **{f"recall_{GROUPS[k]}": float(rec[k]) for k in range(3)}}


def pair_stats(base: pd.DataFrame, new: pd.DataFrame, block: int) -> dict:
    """Section 12.6: squared-error difference (base minus new; positive favours new) with a Newey-West test and
    a paired moving-block bootstrap; accuracy difference with the same bootstrap; HLN-DM and McNemar counts."""
    y = base["y_true"].to_numpy(float)
    e_b, e_n = (y - base["y_pred"].to_numpy(float)) ** 2, (y - new["y_pred"].to_numpy(float)) ** 2
    ok_b = base["class_pred"].to_numpy() == base["class_true"].to_numpy()
    ok_n = new["class_pred"].to_numpy() == new["class_true"].to_numpy()
    nw = nw_summary(e_b - e_n, lags=block)
    fixes, breaks, p_mc = mcnemar_greater(ok_b, ok_n)
    return {"n": int(len(y)), "sq_diff_mean": nw["mean"], "nw_p": nw["p_one_sided"], "sq_diff_ci90": nw["ci90"],
            "sq_diff_boot_ci90": block_bootstrap(e_b - e_n, block, BOOT_DRAWS, 0)["ci90"],
            "acc_diff": float(ok_n.mean() - ok_b.mean()),
            "acc_diff_ci90": block_bootstrap(ok_n.astype(float) - ok_b.astype(float), block, BOOT_DRAWS, 0)["ci90"],
            "hln_dm_p": dm_hln(e_b, e_n)[1], "mcnemar_fixes": fixes, "mcnemar_breaks": breaks, "mcnemar_p": p_mc}


def _subsets(cfg: TaskCfg, pan: dict, dates: pd.Series) -> dict:
    """Row masks for the reported subsets (scheduled / all meetings; all months / the paper's window)."""
    if cfg.name == "T2":
        unsched = pan["C"]["unscheduled"].reindex(pd.DatetimeIndex(dates)).fillna(0).to_numpy() == 1
        return {"scheduled": ~unsched, "all meetings": np.ones(len(dates), bool)}
    paper = (dates >= pd.Timestamp(config.PAPER_T1[0])) & (dates <= pd.Timestamp(config.PAPER_T1[1]))
    return {"all test months": np.ones(len(dates), bool), "paper window": paper.to_numpy()}


def registered_holm(comp: pd.DataFrame, cfg: TaskCfg) -> list:
    """Section 12.6: Holm across the registered family, every architecture in ARCHS times every primary pair, in
    the task's main subset. The family is fixed by the design, not by which arms have finished: while any member
    is missing, no adjusted p-value is reported (an interim family would understate the adjustment)."""
    main_sub = "scheduled" if cfg.name == "T2" else "all test months"
    family = [(arch, new, base) for arch in ARCHS for new, base in cfg.primary]
    prim = comp[(comp["role"] == "primary") & (comp["subset"] == main_sub)]
    found = {(r.arch, r.model, r.against): r.nw_p for r in prim.itertuples()}
    missing = [k for k in family if k not in found]
    if missing:
        print(f"Holm not computed: registered family incomplete, missing {missing}", flush=True)
        return [np.nan] * len(comp)
    adj = holm({f"{a} {new}": found[(a, new, base)] for a, new, base in family})
    return [adj[f"{r.arch} {r.model}"] if r.role == "primary" and r.subset == main_sub else np.nan
            for r in comp.itertuples()]


def forecast_all(pan: dict, device: str, archs=ARCHS, cfg: TaskCfg = TASKS["T2"]) -> pd.DataFrame:
    """Forecast every network and feature set (8 seeds, mean), then write metrics and comparisons."""
    out, preds = cfg.out, {}
    for arch in archs:
        for fs, (which, cols) in cfg.features.items():
            if not _settings_path(arch, fs, out).exists():
                continue
            df = pan[which]
            settings = {int(k): v for k, v in json.loads(_settings_path(arch, fs, out).read_text()).items()}
            frames = []
            for seed in SEEDS:
                path = out / f"preds_{arch}_{fs}_seed{seed}.csv"
                if not path.exists():
                    run_tuned(df, cols, arch, settings, seed, device, config.TEST_START, task=cfg.name,
                              refit_every=cfg.refit_every).to_csv(path, index=False)
                frames.append(pd.read_csv(path, parse_dates=["date"]))
            ens = frames[0][["date", "y_true", "lag", "class_true"]].copy()
            ens["y_pred"] = np.mean([f["y_pred"].to_numpy() for f in frames], axis=0)
            ens["class_pred"] = models.classes_from_change(ens["y_pred"].to_numpy() - ens["lag"].to_numpy())
            ens.to_csv(out / f"preds_{arch}_{fs}.csv", index=False)
            preds[(arch, fs)] = ens
    rows, comps, confusions = [], [], []
    for (arch, fs), p in preds.items():
        for sub, mask in _subsets(cfg, pan, p["date"]).items():
            q = p[mask]
            m = metrics(q)
            rows.append({"task": cfg.name, "arch": arch, "features": fs, "subset": sub, "n": m["n"], "rmse": m["rmse"],
                         "mae": m["mae"], "mbe": m["mbe"], **class_report(q)})
            if cfg.name == "T2" and sub == "scheduled":
                cm = confusion_matrix(q["class_true"], q["class_pred"], labels=[0, 1, 2])
                confusions += [{"arch": arch, "features": fs, "true": GROUPS[i],
                                **{f"pred_{GROUPS[j]}": int(cm[i, j]) for j in range(3)}} for i in range(3)]
    for arch in archs:
        for new, base in cfg.pairs:
            if (arch, new) not in preds or (arch, base) not in preds:
                continue
            role = "primary" if (new, base) in cfg.primary else "strict" if new.startswith("strict") else "placebo"
            b, n = preds[(arch, base)], preds[(arch, new)]
            for sub, mask in _subsets(cfg, pan, b["date"]).items():
                comps.append({"task": cfg.name, "arch": arch, "model": new, "against": base, "role": role,
                              "subset": sub, **pair_stats(b[mask].reset_index(drop=True),
                                                          n[mask].reset_index(drop=True), cfg.refit_every)})
    comp = pd.DataFrame(comps)
    if len(comp):
        comp["nw_p_holm"] = registered_holm(comp, cfg)
    if cfg.name == "T1":
        comp = pd.concat([comp, paper_settings_rows(preds, archs, cfg)], ignore_index=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"metrics_neural_{cfg.name}.csv", index=False)
    comp.to_csv(out / f"comparison_neural_{cfg.name}.csv", index=False)
    if confusions:
        pd.DataFrame(confusions).to_csv(out / f"confusion_neural_{cfg.name}.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(comp.drop(columns=["sq_diff_ci90", "sq_diff_boot_ci90", "acc_diff_ci90"], errors="ignore").round(4)
          .to_string(index=False), flush=True)
    return comp


def paper_settings_rows(preds: dict, archs, cfg: TaskCfg) -> pd.DataFrame:
    """Section 11 (a), descriptive under 12.6: tuned S against Plan 2's paper-settings S forecasts on the same
    months (Plan 2 used the version 1 cutoff; its LSTM settings do not match the paper version read 2026-10-12)."""
    rows = []
    for arch in archs:
        if (arch, "S") not in preds:
            continue
        old = pd.read_csv(config.RESULTS / "inputs_from_earlier" / "paper_settings" / f"preds_T1_{arch}-S.csv",
                          parse_dates=["date"])
        new = preds[(arch, "S")]
        for window, (lo, hi) in (("all test months", (config.TEST_START, "2100-01-01")),
                                 ("paper window", config.PAPER_T1)):
            o = old[(old["date"] >= lo) & (old["date"] <= hi)]
            n = new[new["date"].isin(o["date"])].sort_values("date").reset_index(drop=True)
            o = o[o["date"].isin(n["date"])].sort_values("date").reset_index(drop=True)
            rows.append({"task": "T1", "arch": arch, "model": "S tuned (v2)", "against": "S paper settings (Plan 2)",
                         "role": "settings", "subset": window, "rmse_against": metrics(o)["rmse"],
                         "rmse_model": metrics(n)["rmse"], **pair_stats(o, n, cfg.refit_every)})
    return pd.DataFrame(rows)


def panels_for(cfg: TaskCfg) -> dict:
    """The Haiku (H) and ChronoBERT (C) panels for a task, plus their placebo versions."""
    from fed_news.download import alfred_vintages
    if cfg.name == "T2":
        from fed_news.studies.simple_model.run_simple_model import panels
        pan = panels()
    else:
        from fed_news.studies.simple_model.run_simple_model import same_rows
        from fed_news.features.haiku_tone import OUT_PATH as HAIKU_PATH
        from fed_news.features.swap_tone import naive_panel
        from fed_news.models.walk_forward import common_rows
        c = common_rows(naive_panel(cfg.panel, pd.read_pickle(config.DATA / "doc_stance.pkl")))
        h = common_rows(naive_panel(cfg.panel, pd.read_pickle(HAIKU_PATH)))
        c, h = same_rows(c, h)
        pan = {"C": c, "H": h}
    first = {k: v["realtime_start"].min() for k, v in alfred_vintages.load_all().items()}
    check_vintages(first, KEPT_SERIES, pd.Timestamp(pan["C"]["cutoff"].iloc[0]))  # strict arms: no fallback
    pan["Cp"], pan["Hp"] = placebo_panel(pan["C"]), placebo_panel(pan["H"])
    return pan


def main(argv=None) -> None:
    """Command line: time, tune or forecast."""
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["time", "tune", "forecast"])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    ap.add_argument("--features", nargs="+", default=None, choices=sorted(set(FEATURES) | set(FEATURES_T1)))
    ap.add_argument("--task", default="T2", choices=sorted(TASKS))
    args = ap.parse_args(argv)
    cfg = TASKS[args.task]
    pan = panels_for(cfg)
    if args.step == "time":
        time_trial(pan, args.device, cfg, args.archs)
    elif args.step == "tune":
        tune_all(pan, args.device, args.trials, args.archs, args.features, cfg)
    else:
        forecast_all(pan, args.device, args.archs, cfg)


if __name__ == "__main__":
    main()
