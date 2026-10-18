import numpy as np
import pytest

from fed_news.evaluation import stats_tests as st


def test_dm_hln_hand_computed():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    new = np.array([0.5, 1.5, 2.0, 4.0])
    # d = [0.5, 0.5, 1.0, 0.0], mean 0.5, var (population) 0.125, DM = 0.5/sqrt(0.125/4) = 2.828
    # HLN factor sqrt((n - 1)/n) = sqrt(0.75); stat = 2.449; one-sided t(3) p = 0.0459
    stat, p = st.dm_hln(base, new)
    assert stat == pytest.approx(2.4495, abs=1e-3)
    assert p == pytest.approx(0.0459, abs=1e-3)


def test_dm_hln_identical_losses_is_p_one_half_or_more():
    x = np.ones(10)
    assert st.dm_hln(x, x)[1] >= 0.5


def test_mcnemar_exact_one_sided():
    base = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    new = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    fixes, breaks, p = st.mcnemar_greater(base.astype(bool), new.astype(bool))
    assert (fixes, breaks) == (6, 1)
    assert p == pytest.approx(8 / 128)


def test_holm():
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
