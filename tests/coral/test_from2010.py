import json

import numpy as np
import pandas as pd
import pytest

from fed_news.coral import from2010 as f10
from fed_news.coral import run_logit as rl


def _preds(dates, rng, noise=0.5):
    n = len(dates)
    y = np.linspace(5, 6, n)
    cls = np.arange(n) % 3
    p = rng.dirichlet([2, 2, 2], n)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + rng.normal(0, noise, n), "lag": y,
                         "class_true": cls, "class_pred": p.argmax(1), "p_cut": p[:, 0], "p_hold": p[:, 1],
                         "p_hike": p[:, 2]})


def test_subset_keeps_only_meetings_from_the_start_date():
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(["2009-11-04", "2009-12-16", "2010-01-27", "2010-03-16"])
    out = f10.subset({"S+M": _preds(dates, rng), "S+M+Ho": _preds(dates, rng)}, "2010-01-01")
    for p in out.values():
        assert list(p["date"]) == list(dates[2:]) and list(p.index) == [0, 1]


def test_subset_refuses_versions_with_different_meetings():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2009-06-01", periods=12, freq="MS")
    with pytest.raises(ValueError, match="same meetings"):
        f10.subset({"S+M": _preds(dates, rng), "S+M+Ho": _preds(dates[1:], rng)}, "2009-01-01")


def test_subset_refuses_an_empty_result():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2009-06-01", periods=12, freq="MS")
    with pytest.raises(ValueError, match="no test meetings"):
        f10.subset({"S+M": _preds(dates, rng)}, "2030-01-01")


def _fake_memory(dates):
    n = len(dates)
    return pd.DataFrame({"recalled": np.arange(n) % 2 == 0, "dated": np.ones(n, bool), "has_exam": np.ones(n, bool)},
                        index=pd.DatetimeIndex(dates))


def test_logit_from_start_scores_only_the_later_meetings(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2005-01-01", periods=120, freq="MS")
    preds = {name: _preds(dates, rng, 0.3 if "+" in name else 0.5) for name in rl.specs()}
    for name, p in preds.items():
        p.to_csv(tmp_path / f"preds_T2_{name}.csv", index=False)
    monkeypatch.setattr(rl, "OUT", tmp_path)
    monkeypatch.setattr(rl, "_memory_indicators", _fake_memory)
    monkeypatch.setattr(f10, "OUT_LOGIT", tmp_path / "from2010")
    res = f10.run_logit("2010-01-01")
    later = dates >= "2010-01-01"
    assert res["n_test_meetings"] == later.sum() and res["first"] == "2010-01-01"
    assert res["n_rate_moves"] == int((preds["S+M"]["class_true"][later] != 1).sum())
    sub = {k: p[later].reset_index(drop=True) for k, p in preds.items()}
    assert np.isclose(res["primary_clark_west_p"]["S+M+Ho"], rl.compare_row(sub["S+M"], sub["S+M+Ho"])["clark_west_p"])
    assert set(res["primary_holm_p"]) == {"S+M+Co", "S+M+Ho"} and "decision" in res["S+M+Co"]
    table = pd.read_csv(tmp_path / "from2010" / "comparison.csv")
    assert len(table) == 7 and (table["n"] == later.sum()).all()
    assert json.loads((tmp_path / "from2010" / "verdict.json").read_text())["start"] == "2010-01-01"


def test_networks_from_start_use_the_later_scheduled_meetings(tmp_path, monkeypatch):
    rn = pytest.importorskip("fed_news.coral.run_networks")
    rng = np.random.default_rng(2)
    dates = pd.date_range("2006-01-01", periods=96, freq="MS")
    for arch in rn.ARCHS:
        for arm in rn.ARMS:
            _preds(dates, rng, 0.3 if arm != "S+M" else 0.5).to_csv(tmp_path / f"preds_{arch}_{arm}.csv",
                                                                     index=False)
    unsched = pd.Series((np.arange(96) % 10 == 0).astype(int), index=dates)
    monkeypatch.setattr(rn, "OUT", tmp_path)
    monkeypatch.setattr(rn, "BOOT_DRAWS", 50)
    monkeypatch.setattr(rn, "panels", lambda coral=True: {"C": pd.DataFrame({"unscheduled": unsched})})
    monkeypatch.setattr(f10, "OUT_NETWORKS", tmp_path / "from2010")
    comp = f10.run_networks("2010-01-01")
    later = dates >= "2010-01-01"
    sched = comp[comp["subset"] == "scheduled"]
    assert (sched["n"] == int((later & (unsched.to_numpy() == 0)).sum())).all()
    assert (comp.loc[comp["subset"] == "all meetings", "n"] == later.sum()).all()
    prim = sched[sched["role"] == "primary"]
    assert len(prim) == 6 and prim["nw_p_holm"].notna().all()
    summary = json.loads((tmp_path / "from2010" / "summary.json").read_text())
    assert summary["start"] == "2010-01-01" and "lstm S+M+Ho" in summary["results"]
