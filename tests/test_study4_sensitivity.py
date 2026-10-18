import numpy as np
import pandas as pd

from fed_news.studies.study4_haiku_logit import study4_sensitivity as s4
from fed_news.features.panel import M_COLS, S_COLS


def test_realtime_columns_drop_only_the_fallback_series():
    cols = s4.realtime(S_COLS + M_COLS)
    assert not set(s4.FALLBACK_COLS) & set(cols)
    assert len(cols) == len(S_COLS) + len(M_COLS) - 4 and "s_cpi_yoy" in cols


def test_ablation_specs_are_cumulative_then_single():
    names = list(s4.ablation_specs())
    assert names[:6] == ["S+M", "S+M+stance", "S+M+stance+beige", "S+M+stance+beige+news",
                         "S+M+stance+beige+news+gpr", "S+M+N"]
    assert set(names[6:]) == {"S+M+beige", "S+M+news", "S+M+gpr", "S+M+epu"}
    assert s4.ablation_specs()["S+M+N"].cols[-1] == "n_epu"


def test_class_counts_report_correct_and_net_change():
    base = pd.DataFrame({"class_true": [0, 1, 1, 2, 1], "class_pred": [1, 1, 0, 2, 1]})
    new = pd.DataFrame({"class_true": [0, 1, 1, 2, 1], "class_pred": [0, 1, 1, 2, 2]})
    t = s4.class_counts(base, new).set_index("group")
    assert t.loc["cut", "base_correct"] == 0 and t.loc["cut", "new_correct"] == 1 and t.loc["cut", "net"] == 1
    assert t.loc["hold", "n"] == 3 and t.loc["hold", "net"] == 0
    assert t.loc["turning point", "n"] == 3  # classes 1->2, 2->1 and 0->1 differ from the previous meeting


def test_hac_p_is_small_for_a_clear_positive_mean_and_large_for_zero_mean():
    rng = np.random.default_rng(0)
    assert s4.hac_p(1 + rng.normal(0, 1, 200), 4) < 0.01
    assert s4.hac_p(rng.normal(0, 1, 200) - 0.5, 4) > 0.5


def test_block_bootstrap_is_reproducible_and_centred_on_the_mean():
    d = np.random.default_rng(1).normal(0.1, 1, 300)
    a, b = s4.block_bootstrap(d, 8, 2000, 0), s4.block_bootstrap(d, 8, 2000, 0)
    assert a == b and a["ci90"][0] < d.mean() < a["ci90"][1]
