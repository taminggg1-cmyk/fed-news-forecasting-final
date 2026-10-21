import numpy as np
import pandas as pd

from fed_news.coral import run_networks as rn
from fed_news.distill import run_networks as dn


def test_arms_are_the_distilled_tone_and_its_placebo_on_the_coral_columns():
    assert dn.ARMS == {"S+M+Hd": ("Hd", rn.SM + rn.D), "placebo-Hd": ("Hdp", rn.SM + rn.D)}


def _comp(p_by_arch, period="from2010", subset="scheduled"):
    rows = []
    for arch, (p_prim, p_real_vs_pl, p_pl_vs_sm) in p_by_arch.items():
        rows += [{"arch": arch, "model": "S+M+Hd", "against": "S+M", "period": period, "subset": subset,
                  "nw_p": p_prim},
                 {"arch": arch, "model": "S+M+Hd", "against": "placebo-Hd", "period": period, "subset": subset,
                  "nw_p": p_real_vs_pl},
                 {"arch": arch, "model": "placebo-Hd", "against": "S+M", "period": period, "subset": subset,
                  "nw_p": p_pl_vs_sm}]
    return pd.DataFrame(rows)


def test_holm_runs_over_the_three_primary_tests_from_2010_on_scheduled_meetings_only():
    comp = pd.concat([_comp({"lstm": (0.01, 0.2, 0.5), "gru": (0.02, 0.2, 0.5), "tft": (0.3, 0.2, 0.5)}),
                      _comp({"lstm": (0.001, 0.2, 0.5), "gru": (0.9, 0.2, 0.5), "tft": (0.9, 0.2, 0.5)},
                            period="full")], ignore_index=True)
    adj = dn.primary_holm(comp)
    prim = comp[(comp["model"] == "S+M+Hd") & (comp["against"] == "S+M") & (comp["period"] == "from2010")]
    assert np.allclose(adj[prim.index], [0.03, 0.04, 0.3])
    assert adj.drop(prim.index).isna().all()


def test_placebo_reading_needs_real_beating_placebo_and_placebo_not_beating_no_news():
    comp = _comp({"lstm": (0.01, 0.03, 0.5), "gru": (0.01, 0.2, 0.5), "tft": (0.01, 0.03, 0.04)})
    assert dn.reading(comp) == {"lstm": True, "gru": False, "tft": False}
