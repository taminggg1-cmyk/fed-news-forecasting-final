"""Primary models (spec section 7): penalised ordered logit (T2), ridge on the change (T1),
expanding-window penalty choice, move sizes, the class band, and the inertial Taylor rule."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from fed_news import config


class OrderedLogit:
    """Cumulative logit for cut < hold < hike. Thresholds t1 = a, t2 = a + exp(b).
    Loss = mean negative log-likelihood + lam/2 * ||beta||^2 (slopes only)."""

    def __init__(self, lam: float):
        self.lam = lam

    @staticmethod
    def _probs(X, beta, t1, t2):
        eta = X @ beta
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        return np.column_stack([c1, c2 - c1, 1.0 - c2])

    def _loss(self, w, X, y):
        p = X.shape[1]
        beta, a, b = w[:p], w[p], w[p + 1]
        t1, t2 = a, a + np.exp(b)
        eta = X @ beta
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        g1, g2 = c1 * (1 - c1), c2 * (1 - c2)
        prob = np.clip(np.select([y == 0, y == 1], [c1, c2 - c1], 1.0 - c2), 1e-12, None)
        d_eta = np.select([y == 0, y == 1], [-g1, g1 - g2], g2) / prob
        d_t1 = np.select([y == 0, y == 1], [g1, -g1], 0.0) / prob
        d_t2 = np.select([y == 0, y == 1], [0.0, g2], -g2) / prob
        n = len(y)
        loss = -np.mean(np.log(prob)) + 0.5 * self.lam * beta @ beta
        grad = np.empty_like(w)
        grad[:p] = -(X.T @ d_eta) / n + self.lam * beta
        grad[p] = -np.mean(d_t1 + d_t2)
        grad[p + 1] = -np.mean(d_t2) * np.exp(b)
        return loss, grad

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if not np.isfinite(X).all():
            raise ValueError("OrderedLogit.fit: non-finite values in X")
        p = X.shape[1]
        frac = np.clip(np.cumsum(np.bincount(y, minlength=3) / len(y))[:2], 1e-3, 1 - 1e-3)
        gap = max(logit(frac[1]) - logit(frac[0]), 1e-3)
        w0 = np.concatenate([np.zeros(p), [logit(frac[0]), np.log(gap)]])
        res = minimize(self._loss, w0, args=(X, y), jac=True, method="L-BFGS-B")
        self.beta_ = res.x[:p]
        self.t1_, self.t2_ = res.x[p], res.x[p] + np.exp(res.x[p + 1])
        if not (np.isfinite(self.beta_).all() and np.isfinite([self.t1_, self.t2_]).all()):
            raise ValueError("OrderedLogit fit produced non-finite parameters (check inputs for inf/NaN)")
        return self

    def predict_proba(self, X):
        proba = self._probs(X, self.beta_, self.t1_, self.t2_)
        if not np.isfinite(proba).all():
            raise ValueError("OrderedLogit produced non-finite probabilities")
        return proba


def expanding_folds(n: int, k: int = config.CV_FOLDS):
    edges = np.linspace(0, n, k + 2).astype(int)
    return [(np.arange(0, edges[i]), np.arange(edges[i], edges[i + 1])) for i in range(1, k + 1)]


def ridge_score(model, X, y) -> float:
    return float(np.mean((model.predict(X) - y) ** 2))


def logloss_score(model, X, y) -> float:
    proba = np.clip(model.predict_proba(X), 1e-12, None)
    return float(-np.mean(np.log(proba[np.arange(len(y)), np.asarray(y, dtype=int)])))


def choose_penalty(make, score, X, y, grid) -> float:
    best, best_score = grid[0], np.inf
    for value in grid:
        s = np.mean([score(make(value).fit(X[tr], y[tr]), X[va], y[va])
                     for tr, va in expanding_folds(len(y))])
        if not np.isfinite(s):
            raise ValueError(f"non-finite CV score for penalty {value}")
        if s < best_score:
            best, best_score = value, s
    return best


def move_sizes(y_class: np.ndarray, y_change: np.ndarray) -> np.ndarray:
    sizes = np.array([-0.25, 0.0, 0.25])
    for k in (0, 2):
        if (y_class == k).any():
            sizes[k] = float(np.mean(y_change[y_class == k]))
    return sizes


def classes_from_change(change: np.ndarray, band: float = config.CLASS_BAND) -> np.ndarray:
    change = np.asarray(change, dtype=float)
    return np.where(change > band, 2, np.where(change < -band, 0, 1))


def taylor_predict(X_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(X_train)), X_train])
    coef = np.linalg.lstsq(design, y_train, rcond=None)[0]
    return float(coef[0] + x_test @ coef[1:])
