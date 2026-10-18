"""Study 4 v2 verification before the overnight batch (design section 12.7).

(a) monthly cutoff rule: every v2 monthly cutoff is the second US business day, after the first business day on
    which the last daily effective rate of the previous month is published;
(b) folds: identical validation rows for every candidate lookback, the monthly t-2 gap, infeasible lookbacks;
(c) rate-unit scoring on real data;
(d) end-to-end truncation of features: rebuilding sampled rows from raw inputs cut at their cutoff gives the
    stored values (fallback series excepted, and counted);
(e) end-to-end truncation of forecasts: a forecast is unchanged when every later row is deleted.
"""
import json

import numpy as np
import pandas as pd

from fed_news.collect import alfred, market
from fed_news import config
from fed_news.features import panel
from fed_news.studies.study4_networks_v2 import tune
from fed_news.features.text_features import fed_stance

OUT = config.RESULTS / "study4" / "v2_verification.json"
FALLBACK_SERIES_COLS = {"s_pce_yoy", "s_pce_core_yoy", "s_earn_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy",
                        "s_cpi_core_yoy", "s_hpi_yoy"}


def check_cutoffs(t1b: pd.DataFrame) -> dict:
    bad = []
    for month, cutoff in t1b["cutoff"].items():
        first = panel._BDAY.rollforward(month)
        last_prev = panel._BDAY.rollback(month - pd.Timedelta(days=1))
        # the daily rate for last_prev is published on the next business day (first), before the cutoff
        if not (cutoff == panel.second_business_day(month) and last_prev < first < cutoff
                and first == last_prev + panel._BDAY):
            bad.append(str(month.date()))
    return {"months": int(len(t1b)), "violations": bad}


def check_folds(cfg, pan) -> list:
    df = pan["C"]
    rows = []
    for pt in tune.tuning_points(df.index):
        folds = tune.folds_for(pt, cfg.gap, max(cfg.lookbacks))
        gaps_ok = all(end == va.min() - cfg.gap for end, va in folds)
        feasible = {lb: min(len(np.arange(lb - 1, end)) for end, _ in folds) >= tune.MIN_TRAIN_SEQ
                    for lb in cfg.lookbacks}
        rows.append({"task": cfg.name, "point": str(df.index[pt].date()), "rows_before": pt,
                     "validation_rows": [int(folds[0][1].min()), int(folds[-1][1].max())],
                     "last_validation_row_below_point_minus_gap": int(folds[-1][1].max()) < pt - cfg.gap,
                     "train_end_is_block_start_minus_gap": gaps_ok,
                     "feasible_lookbacks": [lb for lb, ok in feasible.items() if ok]})
    return rows


def check_rate_units(cfg, pan) -> dict:
    df = pan["C"]
    cols = cfg.features[cfg.base][1]
    X, y = df[cols].to_numpy(float), df["y_change"].to_numpy(float)
    pt = tune.tuning_points(df.index)[3]
    p = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "lookback": cfg.lookbacks[0], "batch": 32,
         "layers": 1}
    s1, _ = tune.cv_score(X, y, pt, "gru", p, "cpu", max_epochs=20, gap=cfg.gap, max_lookback=max(cfg.lookbacks))
    s100, _ = tune.cv_score(X, y * 100, pt, "gru", p, "cpu", max_epochs=20, gap=cfg.gap,
                            max_lookback=max(cfg.lookbacks))
    return {"task": cfg.name, "score": s1, "score_in_basis_points_over_10000": s100 / 1e4,
            "equal": bool(np.isclose(s1, s100 / 1e4, rtol=1e-6))}


def _truncate_alfred(obs: dict, cutoff) -> dict:
    return {k: v[v["realtime_start"] < cutoff] for k, v in obs.items()}


def check_feature_truncation(cfg, pan, daily, obs, docs, n_rows: int = 8) -> dict:
    df = pan["C"]
    pick = df.index[np.linspace(len(df) // 10, len(df) - 1, n_rows).astype(int)]
    mismatches, fallback_mismatches, checked = [], 0, 0
    for d in pick:
        row, cutoff = df.loc[d], pd.Timestamp(df.loc[d, "cutoff"])
        try:
            macro, _ = panel.macro_at(_truncate_alfred(obs, cutoff), cutoff)
        except Exception:  # a series with no vintage before the cutoff (fallback rows)
            macro = {}
        mk = panel.market_at(daily[daily.index < cutoff], cutoff)
        stance = fed_stance(docs[docs["release"] < cutoff], pd.Timestamp(row["win_start"]), cutoff)
        rebuilt = {**macro, **mk}
        if not np.isnan(stance):
            rebuilt["n_stance"] = stance
        for col, val in rebuilt.items():
            if col not in df.columns:
                continue
            checked += 1
            same = (np.isnan(val) and np.isnan(row[col])) or np.isclose(val, row[col], rtol=1e-9, atol=1e-12)
            if not same:
                if col in FALLBACK_SERIES_COLS:
                    fallback_mismatches += 1
                else:
                    mismatches.append({"date": str(d.date()), "col": col, "stored": float(row[col]),
                                       "rebuilt": float(val)})
        for col in FALLBACK_SERIES_COLS - set(macro):
            fallback_mismatches += int(col in df.columns)
    return {"task": cfg.name, "rows": [str(d.date()) for d in pick], "values_checked": checked,
            "mismatches_outside_fallback_series": mismatches, "fallback_series_values_not_rebuildable": fallback_mismatches}


def check_forecast_truncation(cfg, pan) -> dict:
    df = pan["C"]
    cols = cfg.features["strict+C"][1]
    p = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "lookback": cfg.lookbacks[1], "batch": 32,
         "layers": 1, "epochs": 15}
    pts = tune.tuning_points(df.index)
    settings = {pt: p for pt in pts}
    full = tune.run_tuned(df, cols, "gru", settings, 0, "cpu", config.TEST_START, task=cfg.name,
                          refit_every=cfg.refit_every)
    out = []
    for i in (pts[2] + 5, len(df) - 1):
        cut = tune.run_tuned(df.iloc[: i + 1], cols, "gru", settings, 0, "cpu", config.TEST_START, task=cfg.name,
                             refit_every=cfg.refit_every)
        a = float(cut["y_pred"].iloc[-1])
        b = float(full.loc[full["date"] == df.index[i], "y_pred"].iloc[0])
        out.append({"date": str(df.index[i].date()), "with_later_rows": b, "without": a, "same": abs(a - b) < 1e-9})
    return {"task": cfg.name, "checks": out}


def run() -> dict:
    daily, obs = market.load_daily(), alfred.load_all()
    docs = pd.read_pickle(config.DATA / "doc_stance.pkl")
    t1b = pd.read_pickle(config.DATA / "t1b.pkl")
    res = {"a_cutoffs": check_cutoffs(t1b), "b_folds": [], "c_rate_units": [], "d_feature_truncation": [],
           "e_forecast_truncation": []}
    for name in ("T1", "T2"):
        cfg = tune.TASKS[name]
        pan = tune.panels_for(cfg)
        res["b_folds"] += check_folds(cfg, pan)
        res["c_rate_units"].append(check_rate_units(cfg, pan))
        res["d_feature_truncation"].append(check_feature_truncation(cfg, pan, daily, obs, docs))
        res["e_forecast_truncation"].append(check_forecast_truncation(cfg, pan))
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str), flush=True)
    return res


if __name__ == "__main__":
    run()
