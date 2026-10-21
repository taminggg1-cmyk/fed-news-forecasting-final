import numpy as np
import pandas as pd

from fed_news.studies.coral import run_networks as rn
from fed_news.studies.doctype import run_networks as tn
from fed_news.studies.doctype.group_tone import HS
from fed_news.studies.sigtests import tests as st
from fed_news.studies.simple_model.run_simple_model import log_loss


def test_arms_are_the_split_tone_and_its_placebo_on_the_coral_columns():
    assert tn.ARMS == {"S+M+Hs": ("Hs", rn.SM + HS), "placebo-Hs": ("Hsp", rn.SM + HS)}
    assert ("S+M+Hs", "S+M+Hd") in tn.PAIRS and ("S+M+Hs", "placebo-Hs") in tn.PAIRS


def test_the_placebo_panel_shifts_all_six_columns_by_half_the_panel():
    df = pd.DataFrame(np.arange(60.0).reshape(10, 6), columns=HS).assign(s_rate_lag=1.0)
    out = tn.placebo(df)
    assert np.array_equal(out[HS].to_numpy(), np.roll(df[HS].to_numpy(), 5, axis=0))
    assert (out["s_rate_lag"] == 1.0).all()


def _comp(p_by_arch, period="from2010", subset="scheduled"):
    rows = []
    for arch, (p_prim, p_real_vs_pl, p_pl_vs_sm) in p_by_arch.items():
        rows += [{"arch": arch, "model": "S+M+Hs", "against": "S+M", "period": period, "subset": subset,
                  "nw_p": p_prim},
                 {"arch": arch, "model": "S+M+Hs", "against": "placebo-Hs", "period": period, "subset": subset,
                  "nw_p": p_real_vs_pl},
                 {"arch": arch, "model": "placebo-Hs", "against": "S+M", "period": period, "subset": subset,
                  "nw_p": p_pl_vs_sm}]
    return pd.DataFrame(rows)


def test_holm_runs_over_the_three_primary_tests_from_2010_on_scheduled_meetings_only():
    comp = pd.concat([_comp({"lstm": (0.01, 0.2, 0.5), "gru": (0.02, 0.2, 0.5), "tft": (0.3, 0.2, 0.5)}),
                      _comp({"lstm": (0.001, 0.2, 0.5), "gru": (0.9, 0.2, 0.5), "tft": (0.9, 0.2, 0.5)},
                            period="full"),
                      _comp({"lstm": (0.001, 0.2, 0.5), "gru": (0.9, 0.2, 0.5), "tft": (0.9, 0.2, 0.5)},
                            subset="all meetings")], ignore_index=True)
    adj = tn.primary_holm(comp)
    prim = comp[(comp["model"] == "S+M+Hs") & (comp["against"] == "S+M") & (comp["period"] == "from2010")
                & (comp["subset"] == "scheduled")]
    assert np.allclose(adj[prim.index], [0.03, 0.04, 0.3])
    assert adj.drop(prim.index).isna().all()


def test_placebo_reading_needs_real_beating_placebo_and_placebo_not_beating_no_news():
    comp = _comp({"lstm": (0.01, 0.03, 0.5), "gru": (0.01, 0.2, 0.5), "tft": (0.01, 0.03, 0.04)})
    assert tn.reading(comp) == {"lstm": True, "gru": False, "tft": False}


def _preds(n, rng):
    p = rng.dirichlet([2, 2, 2], n)
    return pd.DataFrame({"class_true": np.arange(n) % 3, "p_cut": p[:, 0], "p_hold": p[:, 1], "p_hike": p[:, 2]})


def test_small_sample_checks_use_the_log_loss_and_rps_differences():
    rng = np.random.default_rng(0)
    base, new = _preds(60, rng), _preds(60, rng)
    out = tn.small_sample(base, new)
    d_ll = log_loss(base) - log_loss(new)
    d_rps = st.rps(base) - st.rps(new)
    assert np.isclose(out["fixed_b_p"], st.fixed_b(d_ll)["p_one_sided"])
    assert np.isclose(out["fixed_m_p"], st.fixed_m(d_ll)["p_one_sided"])
    assert np.isclose(out["rps_nw_p"], st.newey_west(d_rps)["p_one_sided"])
    assert np.isclose(out["rps_fixed_b_p"], st.fixed_b(d_rps)["p_one_sided"])
