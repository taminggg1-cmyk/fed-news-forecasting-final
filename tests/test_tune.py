import numpy as np
import pytest
import optuna
import pandas as pd
import torch

from fed_news.models import neural
from fed_news.studies.study4_networks_v2 import tune


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1996-01-01", periods=n, freq="45D")
    df = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["a", "b", "c"])
    df["y_change"] = 0.25 * np.sign(df["a"]) * (df["a"].abs() > 0.8)
    df["s_rate_lag"] = 3.0
    df["y_level"] = df["s_rate_lag"] + df["y_change"]
    df["y_class"] = np.where(df["y_change"] > 0.1, 2, np.where(df["y_change"] < -0.1, 0, 1))
    return df


def test_rnn_can_stack_layers_with_dropout():
    m = neural.build_model("gru", 3, 8, 0.3, seed=0, layers=2)
    assert m.rnn.num_layers == 2 and m.rnn.dropout == 0.3
    assert m(torch.zeros(5, 4, 3)).shape == (5,)
    assert neural.build_model("lstm", 3, 8, 0.3, seed=0).rnn.num_layers == 1


def test_fit_validate_reports_the_best_validation_epoch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2, 3)).astype(np.float32)
    y = X[:, -1, 0] * 2
    m = neural.build_model("lstm", 3, 8, 0.0, seed=0)
    best, epoch = neural.fit_validate(m, X[:60], y[:60], X[60:], y[60:], lr=1e-2, weight_decay=0.0, batch_size=16,
                                      max_epochs=50, patience=10, device="cpu", seed=0)
    assert 1 <= epoch <= 50 and best < float(np.mean(y[60:] ** 2))


def test_tuning_points_are_the_first_test_meeting_of_each_year():
    dates = pd.to_datetime(["2000-12-01", "2001-02-01", "2001-03-01", "2005-01-20", "2009-06-01"])
    assert tune.tuning_points(dates, years=(2001, 2005, 2009), test_start="2001-01-01") == [1, 3, 4]
    assert tune.point_for(2, [1, 3, 4]) == 1 and tune.point_for(3, [1, 3, 4]) == 3


def test_search_space_gives_valid_settings_for_each_architecture():
    for arch in tune.ARCHS:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        p = tune.suggest(study.ask(), arch)
        assert p["lookback"] in (2, 4, 8, 12) and 1e-4 <= p["lr"] <= 1e-2
        assert ("heads" in p) == (arch == "tft") and ("layers" in p) == (arch != "tft")


def test_cv_score_never_reads_rows_at_or_after_the_tuning_point():
    df = _df(90)
    df.iloc[70:, :3] = np.nan  # anything after the point would turn the score into NaN
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 2, "batch": 16, "layers": 1}
    score, epoch = tune.cv_score(df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float), 70, "lstm",
                                 p, device="cpu", max_epochs=5)
    assert np.isfinite(score) and epoch >= 1


def test_run_tuned_uses_the_settings_of_the_latest_point_and_forecasts_every_test_row():
    df = _df()
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 2, "batch": 16, "layers": 1,
         "epochs": 3}
    settings = {30: dict(p), 45: dict(p, lookback=4)}
    used = []
    out = tune.run_tuned(df, ["a", "b", "c"], "gru", settings, seed=0, device="cpu", test_start=df.index[30],
                         on_refit=lambda i, s: used.append((i, s["lookback"])))
    assert len(out) == 30 and np.isfinite(out["y_pred"]).all()
    assert (30, 2) in used and (45, 4) in used and all(i < 45 or lb == 4 for i, lb in used)


def test_placebo_panel_shifts_only_the_stance_columns_by_half_the_rows():
    df = _df(10).assign(n_stance=np.arange(10.0), n_stance_chg=np.arange(10.0) * 2)
    out = tune.placebo_panel(df)
    assert out["n_stance"].tolist() == [5.0, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert out["n_stance_chg"].tolist() == (np.roll(np.arange(10.0) * 2, 5)).tolist()
    assert out.drop(columns=["n_stance", "n_stance_chg"]).equals(df.drop(columns=["n_stance", "n_stance_chg"]))


def test_placebo_arms_use_the_real_columns_and_the_comparisons_cover_both_questions():
    for reader in ("C", "H"):
        assert tune.FEATURES[f"placebo-{reader}"][1] == tune.FEATURES[f"S+M+{reader}"][1]
        assert tune.FEATURES[f"placebo-{reader}"][0] == reader + "p"
        assert (f"placebo-{reader}", "S+M") in tune.PAIRS and (f"S+M+{reader}", f"placebo-{reader}") in tune.PAIRS


def test_monthly_task_uses_monthly_lookbacks_and_the_paper_inputs():
    cfg = tune.TASKS["T1"]
    assert cfg.lookbacks == (3, 6, 12, 24) and cfg.refit_every == 12 and cfg.gap == 1 and cfg.base == "S"
    assert cfg.features["S"][1] == tune.S_COLS and cfg.features["placebo-H"][1] == tune.S_COLS + tune.D
    assert ("S+C", "placebo-C") in cfg.pairs and ("placebo-H", "S") in cfg.pairs
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    assert tune.suggest(study.ask(), "lstm", cfg.lookbacks)["lookback"] in (3, 6, 12, 24)


def test_monthly_tuning_skips_the_month_before_the_point():
    df = _df(90)
    df.iloc[69, :3] = np.nan  # month t-1 of a point at 70 is not known yet
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 3, "batch": 16, "layers": 1}
    score, _ = tune.cv_score(df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float), 70, "lstm", p,
                             device="cpu", max_epochs=3, gap=1)
    assert np.isfinite(score)


def test_monthly_walk_forward_trains_on_rows_up_to_t_minus_2():
    df = _df()
    df.iloc[29, df.columns.get_loc("y_change")] = np.nan  # the row just before the first forecast
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 3, "batch": 16, "layers": 1,
         "epochs": 2}
    out = tune.run_tuned(df, ["a", "b", "c"], "gru", {30: p}, seed=0, device="cpu", test_start=df.index[30],
                         task="T1", refit_every=100)
    assert len(out) == 30
    with pytest.raises(ValueError):
        tune.run_tuned(df, ["a", "b", "c"], "gru", {30: p}, seed=0, device="cpu", test_start=df.index[30],
                       task="T2", refit_every=100)


def test_every_candidate_gets_the_same_validation_rows_and_the_monthly_gap():
    folds = tune.folds_for(69, gap=1, max_lookback=24)
    assert len(folds) == 4
    vals = np.concatenate([va for _, va in folds])
    assert vals.min() > 23 and vals.max() == 67  # rows 23..67 are usable by every lookback; 68 is month t-1
    for train_end, va in folds:
        assert train_end == va.min() - 1  # monthly: train on rows before the validation block minus one


def test_a_lookback_without_ten_training_sequences_in_every_fold_is_infeasible():
    df = _df(60)
    X, y = df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float)
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 12, "batch": 16, "layers": 1}
    score, _ = tune.cv_score(X, y, 48, "lstm", p, device="cpu", max_epochs=3, max_lookback=12)
    assert score == np.inf


def test_validation_score_is_in_rate_units():
    df = _df(90)
    X, y = df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float)
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 2, "batch": 16, "layers": 1}
    s1, _ = tune.cv_score(X, y, 70, "lstm", p, device="cpu", max_epochs=5)
    s10, _ = tune.cv_score(X, y * 10, 70, "lstm", p, device="cpu", max_epochs=5)
    assert s10 == pytest.approx(100 * s1, rel=1e-4)


def test_strict_arms_use_only_series_with_complete_vintages():
    from fed_news.features import realtime as study5
    t2, t1 = tune.TASKS["T2"].features, tune.TASKS["T1"].features
    assert t2["strict"][1] == study5.B and t2["strict+C"][1] == study5.B + tune.D and t2["strict+H"][0] == "H"
    assert t1["strict"][1] == study5.REALTIME_S
    fallback = {"s_pce_yoy", "s_pce_core_yoy", "s_earn_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy", "s_cpi_core_yoy",
                "s_hpi_yoy"}
    for cfg in (t1, t2):
        for name in ("strict", "strict+C", "strict+H"):
            assert not fallback & set(cfg[name][1])
    assert tune.TASKS["T1"].panel == "t1b"


def test_pair_stats_report_newey_west_and_bootstrap_without_clark_west():
    rng = np.random.default_rng(0)
    n = 120
    base = pd.DataFrame({"y_true": rng.normal(size=n), "class_true": rng.integers(0, 3, n)})
    base["y_pred"], base["class_pred"] = base["y_true"] + rng.normal(0, 1, n), base["class_true"]
    new = base.copy()
    new["y_pred"] = base["y_true"] + rng.normal(0, 0.5, n)
    r = tune.pair_stats(base, new, block=8)
    assert r["nw_p"] < 0.05 and r["sq_diff_mean"] > 0 and "clark_west_p" not in r
    assert set(r) >= {"nw_p", "sq_diff_ci90", "acc_diff_ci90", "hln_dm_p", "mcnemar_fixes", "mcnemar_breaks"}


def test_class_report_gives_balanced_accuracy_and_recalls():
    p = pd.DataFrame({"class_true": [0, 0, 1, 1, 2, 2], "class_pred": [0, 1, 1, 1, 2, 0]})
    r = tune.class_report(p)
    assert r["recall_cut"] == 0.5 and r["recall_hold"] == 1.0 and r["recall_hike"] == 0.5
    assert r["balanced_accuracy"] == pytest.approx(2 / 3)


def _primary_rows(cfg, archs, p=0.01):
    sub = "scheduled" if cfg.name == "T2" else "all test months"
    return pd.DataFrame([{"arch": a, "model": new, "against": base, "role": "primary", "subset": sub, "nw_p": p}
                         for a in archs for new, base in cfg.primary])


def test_holm_family_is_the_registered_six_even_when_arms_are_missing():
    cfg = tune.TASKS["T2"]
    interim = _primary_rows(cfg, ("lstm", "gru"))
    assert np.isnan(tune.registered_holm(interim, cfg)).all()
    full = _primary_rows(cfg, tune.ARCHS, p=0.0034)
    assert np.allclose(tune.registered_holm(full, cfg), 6 * 0.0034)


def test_registered_holm_reproduces_the_committed_v2_adjustments():
    for name, sub in (("T2", "neural"), ("T1", "neural_t1")):
        cfg = tune.TASKS[name]
        comp = pd.read_csv(tune.config.RESULTS / "study4" / sub / f"comparison_neural_{name}.csv")
        comp = comp[comp["role"] != "settings"].reset_index(drop=True)
        assert np.allclose(tune.registered_holm(comp, cfg), comp["nw_p_holm"], equal_nan=True)
