import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fed_news.evaluation.clark_west import clark_west
from fed_news.evaluation.significance import nw_summary
from fed_news.sigtests import tests as st


def _preds(p, cls):
    p = np.asarray(p, dtype=float)
    return pd.DataFrame({"p_cut": p[:, 0], "p_hold": p[:, 1], "p_hike": p[:, 2], "class_true": cls})


def test_ranked_probability_score_uses_cumulative_probabilities():
    out = st.rps(_preds([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [1.0, 0.0, 0.0]], [0, 2, 2]))
    # cut: (0.2-1)^2 + (0.7-1)^2; hike: 0.2^2 + 0.7^2; certain cut but hike: 1 + 1
    assert np.allclose(out, [0.64 + 0.09, 0.04 + 0.49, 2.0])


def test_rps_punishes_a_far_miss_more_than_a_near_miss():
    near, far = st.rps(_preds([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [2, 2]))
    assert far > near


def test_fixed_m_uses_the_daniell_periodogram_and_a_t_distribution_with_2m_degrees_of_freedom():
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 1, 125)
    out = st.fixed_m(d)
    T, m = len(d), int(np.floor(125 ** (1 / 3)))
    lam = 2 * np.pi * np.arange(1, m + 1) / T
    w = np.array([np.sum(d * np.exp(-1j * l * np.arange(1, T + 1))) for l in lam])
    lrv = np.mean(np.abs(w) ** 2 / T)
    t = np.sqrt(T) * d.mean() / np.sqrt(lrv)
    assert out["m"] == m and out["df"] == 2 * m
    assert np.isclose(out["stat"], t) and np.isclose(out["p_one_sided"], 1 - stats.t.cdf(t, 2 * m))


def test_fixed_b_statistic_is_the_bartlett_hac_t_with_bandwidth_root_t():
    rng = np.random.default_rng(1)
    d = rng.normal(0.05, 1, 206)
    out = st.fixed_b(d, n_sim=2000)
    M = int(np.floor(np.sqrt(206)))
    u = d - d.mean()
    lrv = u @ u / 206 + 2 * sum((1 - j / M) * (u[j:] @ u[:-j]) / 206 for j in range(1, M))
    assert out["bandwidth"] == M
    assert np.isclose(out["stat"], d.mean() / np.sqrt(lrv / 206))


def test_fixed_b_critical_value_matches_kiefer_vogelsang():
    # Kiefer and Vogelsang (2005), Bartlett kernel, one-sided 5%: 1.6449 + 2.1859 b + 0.3142 b^2 - 0.3427 b^3
    T = 206
    b = np.floor(np.sqrt(T)) / T
    kv = 1.6449 + 2.1859 * b + 0.3142 * b ** 2 - 0.3427 * b ** 3
    assert abs(st.fixed_b_critical(T, 0.05, n_sim=40000) - kv) < 0.06


def test_fixed_b_p_values_are_one_sided():
    rng = np.random.default_rng(2)
    assert st.fixed_b(rng.normal(0.5, 1, 150), n_sim=2000)["p_one_sided"] < 0.01
    assert st.fixed_b(rng.normal(-0.5, 1, 150), n_sim=2000)["p_one_sided"] > 0.99


def test_giacomini_white_with_only_a_constant_is_the_squared_newey_west_t():
    rng = np.random.default_rng(3)
    d = rng.normal(0.1, 1, 200)
    gw = st.giacomini_white(d, lags=8, lagged=False)
    t = nw_summary(d, lags=8)["mean"] / nw_summary(d, lags=8)["se"]
    assert np.isclose(gw["stat"], t ** 2) and gw["df"] == 1
    assert np.isclose(gw["p"], 1 - stats.chi2.cdf(t ** 2, 1))


def test_giacomini_white_conditional_uses_a_constant_and_the_lagged_difference():
    rng = np.random.default_rng(4)
    d = rng.normal(0, 1, 200)
    gw = st.giacomini_white(d, lags=8)
    assert gw["df"] == 2 and gw["n"] == 199 and 0 <= gw["p"] <= 1


def test_clark_west_newey_west_with_no_lags_is_close_to_the_plain_test():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 300)
    small, large = y + rng.normal(0, 1, 300), y + rng.normal(0, 0.8, 300)
    plain_t, _ = clark_west(y, small, large)
    assert np.isclose(st.clark_west_nw(y, small, large, lags=0)["stat"], plain_t, rtol=0.01)


def test_in_sample_likelihood_ratio_finds_a_real_predictor_and_not_noise():
    rng = np.random.default_rng(6)
    n = 400
    base, x, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    latent = 0.8 * base + 1.2 * x + rng.logistic(size=n)
    y = np.digitize(latent, [-1.0, 1.0])
    df = pd.DataFrame({"y_class": y, "b": base, "x": x, "z": noise}, index=pd.date_range("1990", periods=n, freq="MS"))
    real = st.in_sample(df, ["b"], ["x"])
    fake = st.in_sample(df, ["b"], ["z"])
    assert real["lr_p"] < 1e-6 and real["hac_wald_p"] < 1e-4 and real["first_tone_z_p_one_sided"] < 1e-4
    assert fake["lr_p"] > 0.01
    assert np.isclose(real["lr_stat"], 2 * (real["loglike_with"] - real["loglike_without"]))


def test_spa_and_stepm_find_a_clearly_better_model_and_mcs_drops_a_clearly_worse_one():
    rng = np.random.default_rng(7)
    n = 300
    d = np.column_stack([rng.normal(0.4, 1, n), rng.normal(0.0, 1, n)])  # loss differences, base minus news
    out = st.joint(d, ["good", "none"], reps=500)
    assert out["spa_p_consistent"] < 0.05 and out["stepm_superior"] == ["good"]
    losses = pd.DataFrame({"a": rng.normal(1.0, 0.3, n), "b": rng.normal(2.0, 0.3, n)})
    assert st.mcs(losses, reps=500)["included"] == ["a"]


def test_in_sample_reports_not_identified_when_the_classes_separate():
    rng = np.random.default_rng(8)
    n = 120
    base, x = rng.normal(size=n), rng.normal(size=n)
    y = np.digitize(5 * base, [-2.0, 2.0])  # the base column alone sorts the classes perfectly
    df = pd.DataFrame({"y_class": y, "b": base, "x": x}, index=pd.date_range("1990", periods=n, freq="MS"))
    out = st.in_sample(df, ["b"], ["x"])
    assert out["identified"] is False and np.isnan(out["lr_p"]) and np.isnan(out["hac_wald_p"])


def test_rotation_test_compares_the_real_likelihood_gain_with_circularly_shifted_tones():
    rng = np.random.default_rng(9)
    n = 300
    base = rng.normal(size=n)
    x = np.convolve(rng.normal(size=n + 4), np.ones(5) / 5, mode="valid")  # a persistent tone series
    y = np.digitize(0.8 * base + 3.0 * x + rng.logistic(size=n), [-1.0, 1.0])
    df = pd.DataFrame({"y_class": y, "b": base, "x": x, "z": np.roll(x, n // 2)},
                      index=pd.date_range("1990", periods=n, freq="MS"))
    real = st.rotation_lr(df, ["b"], ["x"], n_shifts=19)
    fake = st.rotation_lr(df, ["b"], ["z"], n_shifts=19)
    assert real["n_shifts"] == 19 and real["p_rotation"] == 0.05  # (1 + 0) / (19 + 1): no shift beats the real tone
    assert fake["p_rotation"] > 0.05
    assert np.isclose(real["lr_stat"], st.in_sample(df, ["b"], ["x"])["lr_stat"], rtol=1e-4)
