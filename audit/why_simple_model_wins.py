"""Descriptive diagnostics (2026-10-18, not a registered test): why the ordered logit beats the tuned networks on
the 206 scheduled meetings (Study 4 version 2, base S+M, normal data).

Run from the repository root: PYTHONPATH=src python audit/2026-10-18/why_simple_model_wins.py
Needs the cached panels (D:/fed_news or FED_NEWS_DATA). Output: why_simple_model_wins.txt in this folder.
"""
import glob
import json

import numpy as np
import pandas as pd

from fed_news.studies.study4_networks_v2 import tune

R = "results/study4"
cfg = tune.TASKS["T2"]
pan = tune.panels_for(cfg)
print("features:", {k: (v[0], len(v[1])) for k, v in cfg.features.items()})

logit = pd.read_csv(f"{R}/preds_T2_S+M.csv", parse_dates=["date"]).set_index("date")
unsched = pan["C"]["unscheduled"].reindex(logit.index).fillna(0).to_numpy() == 1
sched = ~unsched
L = logit[sched]
e_l = (L.y_pred - L.y_true) ** 2
print("scheduled meetings", sched.sum(), "logit RMSE", round(np.sqrt(e_l.mean()), 4),
      "acc", round((L.class_pred == L.class_true).mean(), 3))
hold = (L.class_true == 1).to_numpy()
print("holds", hold.sum(), "moves", (~hold).sum())

n_feat = len(cfg.features["S+M"][1]) if "S+M" in cfg.features else None
for arch in ("lstm", "gru", "tft"):
    N = pd.read_csv(f"{R}/neural/preds_{arch}_S+M.csv", parse_dates=["date"]).set_index("date").loc[L.index]
    e_n = (N.y_pred - N.y_true) ** 2
    excess = e_n - e_l
    top = excess.sort_values(ascending=False)
    seeds = []
    for f in sorted(glob.glob(f"{R}/neural/preds_{arch}_S+M_seed*.csv")):
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date").loc[L.index]
        seeds.append(np.sqrt(((s.y_pred - s.y_true) ** 2).mean()))
    # error on holds: how far the forecast strays from "no change" when the Fed held
    print(f"\n{arch}: RMSE {np.sqrt(e_n.mean()):.4f}  acc {(N.class_pred == N.class_true).mean():.3f}")
    print(f"  single-seed RMSE min {min(seeds):.4f} max {max(seeds):.4f} mean {np.mean(seeds):.4f} (n={len(seeds)})")
    print(f"  total excess SE {excess.sum():.4f}; on holds {excess[hold].sum():.4f}; on moves {excess[~hold].sum():.4f}")
    print(f"  top-5 meetings share of excess {top.iloc[:5].sum() / excess.sum():.2f}:",
          [(d.date().isoformat(), round(v, 3)) for d, v in top.iloc[:5].items()])
    print(f"  mean |pred change| on holds: net {np.abs(N.y_pred - N.lag)[hold].mean():.4f} "
          f"logit {np.abs(L.y_pred - L.lag)[hold].mean():.4f}")
    print(f"  false moves on holds: net {(N.class_pred[hold] != 1).sum()} logit {(L.class_pred[hold] != 1).sum()}")

    st = json.load(open(f"{R}/neural/settings_{arch}_S+M.json"))
    keys = sorted(st, key=int)
    hid = [st[k]["hidden"] for k in keys]
    lb = [st[k]["lookback"] for k in keys]
    ep = [st[k].get("epochs") for k in keys]
    lay = [st[k].get("layers", 1) for k in keys]
    print(f"  refits {len(keys)}; training rows first {keys[0]} last {keys[-1]}")
    print(f"  hidden {hid}\n  lookback {lb}\n  epochs {ep}\n  layers {lay}")
    if n_feat:
        params = []
        for k in keys:
            m = tune._model(arch, n_feat, st[k], 0)
            params.append(sum(p.numel() for p in m.parameters()))
        print(f"  parameters per refit: min {min(params)} median {int(np.median(params))} max {max(params)}")
