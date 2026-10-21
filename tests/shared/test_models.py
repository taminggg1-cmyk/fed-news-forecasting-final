import numpy as np
import pytest
from scipy.optimize import check_grad
from statsmodels.miscmodels.ordinal_model import OrderedModel

from fed_news.models import linear as models


def simulate(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    eta = X @ np.array([1.0, -0.5]) + rng.logistic(size=n)
    y = np.digitize(eta, [-1.0, 1.0])
    return X, y


def test_ordered_logit_matches_statsmodels_without_penalty():
    X, y = simulate()
    ours = models.OrderedLogit(0.0).fit(X, y)
    ref = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    assert np.allclose(ours.beta_, ref.params[:2], atol=5e-3)
    assert np.allclose(ours.predict_proba(X).sum(1), 1.0)


def test_ordered_logit_gradient_is_correct():
    X, y = simulate(100)
    model = models.OrderedLogit(0.3)
    w = np.array([0.2, -0.1, -0.5, 0.3])
    assert check_grad(lambda v: model._loss(v, X, y)[0], lambda v: model._loss(v, X, y)[1], w) < 1e-5


def test_penalty_shrinks_coefficients():
    X, y = simulate()
    weak = models.OrderedLogit(0.0).fit(X, y).beta_
    strong = models.OrderedLogit(3.0).fit(X, y).beta_
    assert np.abs(strong).sum() < np.abs(weak).sum()


def test_expanding_folds_are_chronological():
    folds = models.expanding_folds(60, 5)
    assert len(folds) == 5
    for train, val in folds:
        assert train.max() < val.min() and train.min() == 0


def test_choose_penalty_prefers_strong_penalty_on_noise():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 20)), rng.integers(0, 3, 200)
    lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, X, y, (0.01, 3.0))
    assert lam == 3.0


def test_move_sizes_and_class_band():
    sizes = models.move_sizes(np.array([0, 0, 1, 2]), np.array([-0.25, -0.75, 0.0, 0.25]))
    assert sizes.tolist() == pytest.approx([-0.5, 0.0, 0.25])
    assert models.move_sizes(np.array([1, 2]), np.array([0.0, 0.25]))[0] == pytest.approx(-0.25)
    assert models.classes_from_change(np.array([-0.2, 0.1, 0.13])).tolist() == [0, 1, 2]


def test_taylor_predict_recovers_linear_rule():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3))
    y = 0.5 + X @ np.array([0.9, 0.3, -0.2])
    assert models.taylor_predict(X, y, np.array([1.0, 1.0, 1.0])) == pytest.approx(1.5)


def test_ordered_logit_raises_on_non_finite_data():
    X, y = simulate(100)
    X[3, 0] = np.inf
    with pytest.raises(ValueError):
        models.OrderedLogit(0.1).fit(X, y)


def test_ordered_logit_predict_raises_on_non_finite_proba():
    X, y = simulate(100)
    fitted = models.OrderedLogit(0.1).fit(X, y)
    Xbad = X[:2].copy()
    Xbad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fitted.predict_proba(Xbad)


def test_choose_penalty_raises_on_nan_score():
    X, y = simulate(100)
    with pytest.raises(ValueError):
        models.choose_penalty(models.OrderedLogit, lambda m, X, y: float("nan"), X, y, (0.1, 1.0))
