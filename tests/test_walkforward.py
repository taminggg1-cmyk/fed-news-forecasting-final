import subprocess

import numpy as np
import pandas as pd
import pytest

from fed_news.features import panel
from fed_news.models import walkforward as wf


def synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    df = pd.DataFrame(rng.normal(size=(n, len(panel.S_COLS))), index=idx, columns=panel.S_COLS)
    for c in panel.M_COLS + panel.N_COLS:
        df[c] = rng.normal(size=n)
    df["t_u3"] = rng.normal(size=n)
    df["s_rate_lag"] = 2.0
    change = 0.25 * np.sign(df["n_stance"].to_numpy()) * (np.abs(df["n_stance"].to_numpy()) > 0.5)
    df["y_change"], df["y_level"] = change, 2.0 + change
    df["y_class"] = np.where(change > 0.02, 2, np.where(change < -0.02, 0, 1))
    return df


def test_train_mask_t1_ends_two_rows_back_and_t2_one():
    assert np.flatnonzero(wf.train_mask(10, 6, "T1")).tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(wf.train_mask(10, 6, "T2")).tolist() == [0, 1, 2, 3, 4, 5]


def test_common_rows_drops_any_missing():
    df = synthetic()
    df.iloc[3, df.columns.get_loc("n_epu")] = np.nan
    df.iloc[5, df.columns.get_loc("t_u3")] = np.nan
    out = wf.common_rows(df)
    assert len(out) == len(df) - 2 and df.index[3] not in out.index


def test_placebo_touches_only_n_columns_and_uses_training_rows():
    X = np.arange(20.0).reshape(5, 4)
    train = np.array([True, True, True, False, False])
    out = wf.placebo_matrix(X, train, 3, [2, 3], np.random.default_rng(0))
    assert np.array_equal(out[:, :2], X[:, :2])
    assert sorted(out[:3, 2].tolist()) == sorted(X[:3, 2].tolist())
    assert out[3, 2] in X[:3, 2]
    assert np.array_equal(out[4], X[4])


def test_ladder_names_and_blocks():
    names = [s.name for s in wf.ladder("T2")]
    assert names == ["rw", "taylor", "S", "S+N", "placebo", "shuffled", "S+M", "S+M+N"]
    spec = {s.name: s for s in wf.ladder("T2")}
    assert spec["S+N"].cols == panel.S_COLS + panel.N_COLS and spec["placebo"].placebo


def test_text_spec_refused_without_prereg(monkeypatch):
    monkeypatch.setattr(wf, "prereg_ok", lambda repo=None: False)
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    with pytest.raises(PermissionError):
        wf.run_spec(synthetic(), spec, "2000-01-01")


def test_primary_t2_learns_signal_and_rw_holds():
    df = synthetic(120)
    spec = {s.name: s for s in wf.ladder("T2")}
    sn = wf.run_spec(df, spec["S+N"], "2002-01-01", require_prereg=False)
    rw = wf.run_spec(df, spec["rw"], "2002-01-01", require_prereg=False)
    assert (sn["class_pred"] == sn["class_true"]).mean() > 0.6
    assert (rw["class_pred"] == 1).all() and (rw["y_pred"] == 2.0).all()


def test_prereg_ok_reads_git_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(wf.config, "PREREG_ELSEWHERE", "")

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    assert wf.prereg_ok(tmp_path) is False
    git("tag", "prereg")
    assert wf.prereg_ok(tmp_path) is True


def test_news_guard_is_satisfied_by_the_original_repository_tag(tmp_path):
    assert "fed-news-forecasting" in wf.config.PREREG_ELSEWHERE
    assert wf.prereg_ok(tmp_path) is True


def _one_row(df, spec, i):
    d = df.index[i].strftime("%Y-%m-%d")
    return wf.run_spec(df, spec, d, d, require_prereg=False)


def _perturb(df, i, target_rows):
    d = df.copy()
    for r in target_rows:
        d.iloc[r, d.columns.get_loc("y_level")] = 9.0
        d.iloc[r, d.columns.get_loc("y_change")] = 7.0
        d.iloc[r, d.columns.get_loc("y_class")] = 2
    fut = d.index[i + 1:]
    d.loc[fut, :] = d.loc[fut, :].to_numpy() * -3.0 + 5.0
    d["y_class"] = d["y_class"].astype(int)
    return d


def test_t1_forecast_ignores_rows_after_training_window():
    df = synthetic(100)
    spec = {s.name: s for s in wf.ladder("T1")}["S+N"]
    i = 80
    base = _one_row(df, spec, i)
    alt = _one_row(_perturb(df, i, [i - 1, i]), spec, i)
    assert np.isfinite(base["y_pred"].iloc[0])
    assert base["y_pred"].iloc[0] == alt["y_pred"].iloc[0]


def test_t2_forecast_ignores_current_and_future_targets():
    df = synthetic(100)
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    i = 80
    base = _one_row(df, spec, i)
    alt = _one_row(_perturb(df, i, [i]), spec, i)
    assert base["y_pred"].iloc[0] == alt["y_pred"].iloc[0]
    assert base["class_pred"].iloc[0] == alt["class_pred"].iloc[0]


def test_taylor_branch_runs():
    spec = {s.name: s for s in wf.ladder("T1")}["taylor"]
    out = wf.run_spec(synthetic(100), spec, "2001-01-01", require_prereg=False)
    assert len(out) > 0 and np.isfinite(out["y_pred"]).all()


def test_placebo_runs_end_to_end():
    spec = {s.name: s for s in wf.ladder("T2")}["placebo"]
    out = wf.run_spec(synthetic(100), spec, "2002-01-01", require_prereg=False)
    assert len(out) > 0 and np.isfinite(out["y_pred"]).all()


@pytest.mark.parametrize("name", ["S+N", "placebo", "shuffled", "S+M+N"])
def test_all_text_specs_refused_without_prereg(monkeypatch, name):
    monkeypatch.setattr(wf, "prereg_ok", lambda repo=None: False)
    spec = {s.name: s for s in wf.ladder("T2")}[name]
    with pytest.raises(PermissionError):
        wf.run_spec(synthetic(), spec, "2000-01-01")


def test_run_spec_rejects_missing_values():
    df = synthetic()
    df.iloc[10, df.columns.get_loc("n_epu")] = np.nan
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    with pytest.raises(ValueError):
        wf.run_spec(df, spec, "2000-01-01", require_prereg=False)


def test_run_spec_rejects_unsorted_index():
    df = synthetic().iloc[::-1]
    spec = {s.name: s for s in wf.ladder("T2")}["S"]
    with pytest.raises(ValueError):
        wf.run_spec(df, spec, "2000-01-01", require_prereg=False)



def test_run_spec_rejects_inf_feature():
    df = synthetic()
    df.iloc[10, df.columns.get_loc("n_epu")] = np.inf
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    with pytest.raises(ValueError, match="non-finite"):
        wf.run_spec(df, spec, "2000-01-01", require_prereg=False)


def test_shuffled_labels_control_destroys_signal():
    df = synthetic(120)
    spec = {s.name: s for s in wf.ladder("T2")}
    sn = wf.run_spec(df, spec["S+N"], "2002-01-01", require_prereg=False)
    sh = wf.run_spec(df, spec["shuffled"], "2002-01-01", require_prereg=False)
    acc = lambda p: (p["class_pred"] == p["class_true"]).mean()
    assert np.isfinite(sh["y_pred"]).all()
    assert acc(sh) < acc(sn) - 0.15


def test_run_spec_can_save_class_probabilities_without_changing_forecasts():
    spec = {s.name: s for s in wf.ladder("T2")}["S+M"]
    df = synthetic(100)
    plain = wf.run_spec(df, spec, "2002-01-01", require_prereg=False)
    out = wf.run_spec(df, spec, "2002-01-01", require_prereg=False, proba=True)
    p = out[["p_cut", "p_hold", "p_hike"]].to_numpy()
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert np.array_equal(out["class_pred"], p.argmax(1))
    assert out[plain.columns].equals(plain)
