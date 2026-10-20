import numpy as np
import pandas as pd
import pytest

from fed_news.coral import run_networks as rn

P = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 1e-4, "lookback": 2, "batch": 16, "layers": 1}


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    ranks = np.digitize(X[:, 0], [-0.5, 0.5])
    return X, ranks


def _panel(n=96, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    X = rng.normal(size=(n, 3))
    change = np.where(X[:, 0] > 0.5, 0.25, np.where(X[:, 0] < -0.5, -0.25, 0.0))
    level = 5 + np.cumsum(change)
    return pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "y_change": change,
                         "s_rate_lag": level - change, "y_level": level,
                         "y_class": np.digitize(change, [-0.125, 0.125])}, index=idx)


def test_cv_score_coral_never_reads_rows_at_or_after_the_point():
    X, ranks = _data(120)
    s1 = rn.cv_score_coral(X, ranks, 100, "lstm", P, "cpu", max_lookback=2, max_epochs=5)
    X2, r2 = X.copy(), ranks.copy()
    X2[100:] = 99.0
    r2[100:] = 2
    assert rn.cv_score_coral(X2, r2, 100, "lstm", P, "cpu", max_lookback=2, max_epochs=5) == s1
    assert np.isfinite(s1[0]) and s1[1] >= 1


def test_too_few_sequences_is_infeasible():
    X, ranks = _data(30)
    score, _ = rn.cv_score_coral(X, ranks, 20, "lstm", {**P, "lookback": 12}, "cpu", max_lookback=12, max_epochs=2)
    assert score == np.inf


def test_forecast_unchanged_when_later_rows_are_deleted():
    df = _panel()
    settings = {int(np.flatnonzero(df.index >= "2001-01-01")[0]): {**P, "epochs": 3}}
    full = rn.run_tuned_coral(df, ["a", "b", "c"], "gru", settings, 0, "cpu", "2001-01-01")
    k = 10
    cut = df.iloc[:int(np.flatnonzero(df.index >= "2001-01-01")[0]) + k + 1]
    short = rn.run_tuned_coral(cut, ["a", "b", "c"], "gru", settings, 0, "cpu", "2001-01-01")
    cols = ["lp1", "lp2", "y_pred", "class_pred"]
    assert np.allclose(full.iloc[k][cols].to_numpy(float), short.iloc[k][cols].to_numpy(float))
    assert len(full) == int((df.index >= "2001-01-01").sum())
    assert {"p_cut", "p_hold", "p_hike", "size_cut", "size_hike", "biases_ordered"} <= set(full.columns)


def test_seed_average_is_over_levels_and_classes_follow_the_paper_rule():
    base = {"date": pd.to_datetime(["2001-01-01"]), "y_true": [5.0], "lag": [5.0], "class_true": [1],
            "size_cut": [-0.3], "size_hike": [0.25]}
    a = pd.DataFrame({**base, "lp1": [0.9], "lp2": [0.7]})
    b = pd.DataFrame({**base, "lp1": [0.7], "lp2": [0.2]})
    ens = rn.ensemble([a, b])
    assert np.allclose(ens[["lp1", "lp2"]].to_numpy(), [[0.8, 0.45]])
    assert np.allclose(ens[["p_cut", "p_hold", "p_hike"]].to_numpy(), [[0.2, 0.35, 0.45]])
    assert ens["class_pred"].iloc[0] == 1          # one level above 0.5
    assert np.isclose(ens["y_pred"].iloc[0], 5.0 + 0.2 * -0.3 + 0.45 * 0.25)


def test_pair_stats_coral_favour_better_probabilities():
    rng = np.random.default_rng(0)
    n = 80
    cls = rng.integers(0, 3, n)
    def frame(sharp):
        p = np.full((n, 3), (1 - sharp) / 2)
        p[np.arange(n), cls] = sharp
        return pd.DataFrame({"date": pd.date_range("2001-01-01", periods=n, freq="MS"), "y_true": np.zeros(n),
                             "y_pred": rng.normal(0, 1 - sharp, n), "class_true": cls, "class_pred": cls,
                             "p_cut": p[:, 0], "p_hold": p[:, 1], "p_hike": p[:, 2]})
    s = rn.pair_stats_coral(frame(0.4), frame(0.8))
    assert s["logloss_diff_mean"] > 0 and s["nw_p"] < 0.05
    for key in ("sq_nw_p", "dm_hln_logloss_p", "mcnemar_p", "logloss_boot_ci90", "acc_diff"):
        assert key in s


def test_holm_family_is_six_and_waits_for_missing_arms():
    rows = [{"arch": a, "model": m, "against": "S+M", "role": "primary", "subset": "scheduled", "nw_p": 0.01}
            for a in rn.ARCHS for m, _ in rn.PRIMARY]
    comp = pd.DataFrame(rows)
    adj = rn.registered_holm_coral(comp)
    assert len(rn.ARCHS) * len(rn.PRIMARY) == 6 and np.allclose(adj, 0.06)
    assert np.isnan(rn.registered_holm_coral(comp.iloc[:5])).all()


def test_arms_and_primary_pairs_follow_the_design():
    assert list(rn.ARMS) == ["S+M", "S+M+C", "S+M+H", "S+M+Co", "S+M+Ho", "placebo-Co", "placebo-Ho"]
    assert rn.PRIMARY == [("S+M+Co", "S+M"), ("S+M+Ho", "S+M")] and rn.N_TRIALS == 20
    assert rn.ARMS["placebo-Co"][1] == rn.ARMS["S+M+Co"][1]


def test_cli_refuses_to_train_without_cuda():
    with pytest.raises(SystemExit, match="CUDA"):
        rn.main(["tune", "--device", "cpu"])


@pytest.mark.gpu
def test_coral_networks_train_and_forecast_on_cuda():
    df = _panel()
    settings = {int(np.flatnonzero(df.index >= "2001-01-01")[0]): {**P, "epochs": 3, "heads": 2}}
    for arch in rn.ARCHS:
        out = rn.run_tuned_coral(df, ["a", "b", "c"], arch, settings, 0, "cuda", "2001-01-01")
        assert np.isfinite(out[["lp1", "lp2", "y_pred"]].to_numpy(float)).all()
