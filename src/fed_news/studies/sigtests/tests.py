"""EVALUATION: forecast-comparison tests used in the literature, beyond the registered ones.

Used by sigtests/run_battery.py.

Details: docs/studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md, Part A. Loss differences d are base minus news, so
positive values favour the news version and one-sided p-values test "news is better".
- rps: ranked probability score (Epstein 1969), the proper score for ordered classes.
- fixed_b, fixed_m: Diebold-Mariano with fixed-smoothing asymptotics (Coroneo and Iacone 2020): Bartlett kernel with
  bandwidth floor(T^(1/2)) and a simulated fixed-b null (Kiefer and Vogelsang 2005); Daniell weighted periodogram
  with m = floor(T^(1/3)) and a t distribution with 2m degrees of freedom.
- giacomini_white: conditional predictive ability (Giacomini and White 2006), instruments (1, d_{t-1}).
- clark_west_nw: Clark-West with a Newey-West variance.
- in_sample: unpenalised ordered logit, likelihood ratio and Newey-West sandwich Wald tests of the tone columns
  (Inoue and Kilian 2004 on in-sample tests).
- joint, mcs: Hansen (2005) SPA, Romano and Wolf (2005) StepM and the model confidence set of Hansen, Lunde and
  Nason (2011), from the arch package (stationary bootstrap, mean block length 8).
"""
import numpy as np
import pandas as pd
from scipy import stats

from fed_news.evaluation.significance import nw_summary

PROBS = ["p_cut", "p_hold", "p_hike"]
BLOCK = 8
MAX_COEF = 20.0  # a standardised coefficient above this in an ordered logit signals separation
_FIXED_B_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def rps(preds: pd.DataFrame) -> np.ndarray:
    """Ranked probability score of each forecast: sum over the two cumulative cut-offs of (F_k - O_k)^2."""
    p = preds[PROBS].to_numpy(float)
    y = preds["class_true"].to_numpy(int)
    F = np.cumsum(p, axis=1)[:, :2]
    O = (y[:, None] < np.arange(1, 3)[None, :]).astype(float)
    return ((F - O) ** 2).sum(1)


def _bartlett_lrv(u: np.ndarray, M: int) -> np.ndarray:
    """Bartlett long-run variance of demeaned rows u (..., T), weights 1 - j/M for lags j < M."""
    T = u.shape[-1]
    s = (u * u).sum(-1) / T
    for j in range(1, M):
        s = s + 2 * (1 - j / M) * (u[..., j:] * u[..., :-j]).sum(-1) / T
    return s


def _fixed_b_null(T: int, n_sim: int, seed: int) -> np.ndarray:
    """The fixed-b null distribution: the same Bartlett t statistic on i.i.d. normal series of length T."""
    key = (T, n_sim, seed)
    if key not in _FIXED_B_CACHE:
        rng, M, out = np.random.default_rng(seed), int(np.floor(np.sqrt(T))), []
        for k in range(0, n_sim, 5000):
            x = rng.standard_normal((min(5000, n_sim - k), T))
            u = x - x.mean(1, keepdims=True)
            out.append(x.mean(1) / np.sqrt(_bartlett_lrv(u, M) / T))
        _FIXED_B_CACHE[key] = np.sort(np.concatenate(out))
    return _FIXED_B_CACHE[key]


def fixed_b(d, n_sim: int = 100_000, seed: int = 0) -> dict:
    """Diebold-Mariano t with a Bartlett kernel, bandwidth floor(T^(1/2)), one-sided p from the fixed-b null."""
    d = np.asarray(d, dtype=float)
    T, M = len(d), int(np.floor(np.sqrt(len(d))))
    t = float(d.mean() / np.sqrt(_bartlett_lrv(d - d.mean(), M) / T))
    null = _fixed_b_null(T, n_sim, seed)
    p = float(1 - np.searchsorted(null, t, side="left") / len(null))
    return {"n": T, "bandwidth": M, "stat": t, "p_one_sided": p}


def fixed_b_critical(T: int, alpha: float = 0.05, n_sim: int = 100_000, seed: int = 0) -> float:
    """The simulated one-sided fixed-b critical value for series of length T."""
    return float(np.quantile(_fixed_b_null(T, n_sim, seed), 1 - alpha))


def fixed_m(d) -> dict:
    """Diebold-Mariano t with the Daniell weighted periodogram over m = floor(T^(1/3)) Fourier frequencies; one-sided
    p from a t distribution with 2m degrees of freedom."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    m = int(np.floor(T ** (1 / 3)))
    w = np.fft.fft(d)[1:m + 1]  # frequencies 2 pi j / T, j = 1..m (the mean drops out)
    lrv = float(np.mean(np.abs(w) ** 2) / T)
    t = float(np.sqrt(T) * d.mean() / np.sqrt(lrv))
    return {"n": T, "m": m, "df": 2 * m, "stat": t, "p_one_sided": float(1 - stats.t.cdf(t, 2 * m))}


def newey_west(d, lags: int = BLOCK) -> dict:
    """The registered statistic: Newey-West t with `lags` lags, one-sided normal p."""
    r = nw_summary(d, lags=lags)
    return {"n": r["n"], "stat": r["mean"] / r["se"], "p_one_sided": r["p_one_sided"]}


def giacomini_white(d, lags: int = BLOCK, lagged: bool = True) -> dict:
    """Giacomini-White Wald test of E[h_t d_t] = 0 with h_t = (1, d_{t-1}) (or 1 only), Newey-West covariance."""
    d = np.asarray(d, dtype=float)
    Z = np.column_stack([d[1:], d[1:] * d[:-1]]) if lagged else d[:, None]
    T = len(Z)
    zbar = Z.mean(0)
    u = Z - zbar
    S = u.T @ u / T
    for j in range(1, lags + 1):
        G = u[j:].T @ u[:-j] / T
        S = S + (1 - j / (lags + 1)) * (G + G.T)
    W = float(T * zbar @ np.linalg.solve(S, zbar))
    return {"n": T, "df": Z.shape[1], "stat": W, "p": float(1 - stats.chi2.cdf(W, Z.shape[1])),
            "mean_d": float(d.mean())}


def clark_west_nw(y, small, large, lags: int = BLOCK) -> dict:
    """Clark-West adjusted squared-error difference with a Newey-West standard error, one-sided normal p."""
    y, small, large = (np.asarray(a, dtype=float) for a in (y, small, large))
    f = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
    return newey_west(f, lags)


def _nw_cov(scores: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of the rows of `scores` (not demeaned: scores sum to zero)."""
    S = scores.T @ scores
    for j in range(1, lags + 1):
        G = scores[j:].T @ scores[:-j]
        S = S + (1 - j / (lags + 1)) * (G + G.T)
    return S


def in_sample(df: pd.DataFrame, base_cols: list[str], tone_cols: list[str], lags: int = BLOCK) -> dict:
    """Unpenalised ordered logit of y_class on standardised columns, with and without the tone columns: likelihood
    ratio test, Newey-West sandwich Wald test of the tone coefficients, and one-sided z on the first tone column."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    y = df["y_class"].to_numpy(int)

    def fit(cols):
        X = df[cols].to_numpy(float)
        X = (X - X.mean(0)) / X.std(0)
        model = OrderedModel(y, X, distr="logit")
        res = model.fit(method="bfgs", disp=False, maxiter=5000, gtol=1e-8)
        return model, res

    _, r0 = fit(base_cols)
    m1, r1 = fit(base_cols + tone_cols)
    k = len(tone_cols)
    big = float(max(np.abs(r.params[:-2]).max() for r in (r0, r1)))
    ok = big < MAX_COEF and all(r.mle_retvals.get("converged", False) for r in (r0, r1))
    if not ok:  # quasi-separation: few rate moves, the likelihood keeps rising as coefficients grow
        nan = float("nan")
        return {"n": int(len(y)), "identified": False, "max_abs_coef": big, "loglike_without": float(r0.llf),
                "loglike_with": float(r1.llf), "lr_stat": nan, "lr_df": k, "lr_p": nan, "hac_wald_stat": nan,
                "hac_wald_p": nan, "first_tone_coef": nan, "first_tone_z": nan, "first_tone_z_p_one_sided": nan}
    lr = float(2 * (r1.llf - r0.llf))
    H = m1.hessian(r1.params)
    Hinv = np.linalg.inv(H)
    cov = Hinv @ _nw_cov(m1.score_obs(r1.params), lags) @ Hinv
    idx = np.arange(len(base_cols), len(base_cols) + k)
    b = r1.params[idx]
    wald = float(b @ np.linalg.solve(cov[np.ix_(idx, idx)], b))
    z = float(b[0] / np.sqrt(cov[idx[0], idx[0]]))
    return {"n": int(len(y)), "identified": True, "max_abs_coef": big, "loglike_without": float(r0.llf),
            "loglike_with": float(r1.llf),
            "lr_stat": lr,
            "lr_df": k, "lr_p": float(1 - stats.chi2.cdf(lr, k)), "hac_wald_stat": wald,
            "hac_wald_p": float(1 - stats.chi2.cdf(wald, k)), "first_tone_coef": float(b[0]),
            "first_tone_z": z, "first_tone_z_p_one_sided": float(1 - stats.norm.cdf(z))}


def rotation_lr(df: pd.DataFrame, base_cols: list[str], tone_cols: list[str], n_shifts: int = 99,
                min_shift: int = BLOCK) -> dict:
    """In-sample likelihood-ratio gain of the tone columns against the same gain with the tone series circularly
    shifted (each shift keeps the tone's persistence but breaks its link to the meetings): p = (1 + number of shifts
    with a gain at least as large) / (n_shifts + 1). Valid without the chi-square approximation."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    y = df["y_class"].to_numpy(int)
    T = len(df)

    def llf(cols, frame):
        X = frame[cols].to_numpy(float)
        X = (X - X.mean(0)) / X.std(0)
        return OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=5000, gtol=1e-8).llf

    ll0 = llf(base_cols, df)
    real = 2 * (llf(base_cols + tone_cols, df) - ll0)
    shifts = np.unique(np.round(np.linspace(min_shift, T - min_shift, n_shifts)).astype(int))
    null = np.array([2 * (llf(base_cols + tone_cols, df.assign(**{c: np.roll(df[c].to_numpy(), k)
                                                                   for c in tone_cols})) - ll0) for k in shifts])
    return {"n": T, "lr_stat": float(real), "n_shifts": int(len(shifts)),
            "p_rotation": float((1 + (null >= real).sum()) / (len(null) + 1)),
            "null_median": float(np.median(null)), "null_95": float(np.quantile(null, 0.95))}


def joint(d: np.ndarray, names: list[str], reps: int = 10_000, seed: int = 0) -> dict:
    """SPA p-values and StepM superior models for several comparisons at once. d (T, k): loss differences, base
    minus news; each comparison may have its own base, so the benchmark is zero and the models' losses are -d."""
    from arch.bootstrap import SPA, StepM
    d = np.asarray(d, dtype=float)
    bench, models = np.zeros(len(d)), pd.DataFrame(-d, columns=names)
    spa = SPA(bench, models, block_size=BLOCK, reps=reps, seed=seed)
    spa.compute()
    stepm = StepM(bench, models, size=0.05, block_size=BLOCK, reps=reps, seed=seed)
    stepm.compute()
    pv = spa.pvalues
    return {"spa_p_lower": float(pv["lower"]), "spa_p_consistent": float(pv["consistent"]),
            "spa_p_upper": float(pv["upper"]), "stepm_superior": [str(m) for m in stepm.superior_models]}


def mcs(losses: pd.DataFrame, size: float = 0.10, reps: int = 10_000, seed: int = 0) -> dict:
    """Model confidence set at 1 - size: the members and each model's MCS p-value."""
    from arch.bootstrap import MCS
    m = MCS(losses, size=size, reps=reps, block_size=BLOCK, method="R", seed=seed)
    m.compute()
    return {"included": [str(c) for c in losses.columns if c in set(m.included)],
            "pvalues": {str(k): float(v) for k, v in m.pvalues["Pvalue"].items()}}
