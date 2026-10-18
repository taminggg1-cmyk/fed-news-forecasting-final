import numpy as np
import pytest
import torch

from fed_news.models import neural


def test_make_sequences_windows_end_at_each_row():
    Z = np.arange(20, dtype=float).reshape(10, 2)
    seq = neural.make_sequences(Z, np.array([3, 9]), 4)
    assert seq.shape == (2, 4, 2)
    assert seq[0, -1].tolist() == [6.0, 7.0] and seq[0, 0].tolist() == [0.0, 1.0]
    assert seq[1, -1].tolist() == [18.0, 19.0]


def test_make_sequences_refuses_rows_without_enough_history():
    with pytest.raises(ValueError, match="history"):
        neural.make_sequences(np.zeros((10, 2)), np.array([2]), 4)


def test_build_model_shapes_and_seed_determinism():
    x = torch.zeros((5, 12, 3))
    for arch in ("lstm", "gru", "tft"):
        a = neural.build_model(arch, 3, 16, 0.0, seed=7)
        b = neural.build_model(arch, 3, 16, 0.0, seed=7)
        a.eval(); b.eval()
        assert a(x).shape == (5,)
        assert torch.equal(a(x), b(x))


def test_fit_fixed_reduces_loss_on_learnable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6, 2)).astype(np.float32)
    y = X[:, -1, 0] * 2.0
    model = neural.build_model("lstm", 2, 16, 0.0, seed=0)
    before = float(np.mean((neural.predict(model, X, "cpu") - y) ** 2))
    neural.fit_fixed(model, X, y, lr=1e-2, epochs=150, device="cpu")
    after = float(np.mean((neural.predict(model, X, "cpu") - y) ** 2))
    assert after < 0.5 * before


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[:, -1, 0] * self.w


def test_fit_early_stopping_stops_after_patience_without_improvement():
    X = np.ones((20, 3, 1), dtype=np.float32)
    y = np.ones(20, dtype=np.float32)
    model = Counter()
    neural.fit_early_stopping(model, X, y, lr=0.0, max_epochs=50, patience=3, val_frac=0.2, device="cpu")
    assert model.calls == 1 + 2 * 3   # untrained validation pass, then 3 epochs (train + val) without improvement


def test_fit_early_stopping_never_returns_worse_than_start_on_validation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4, 2)).astype(np.float32)
    y = X[:, -1, 1].astype(np.float32)
    model = neural.build_model("tft", 2, 8, 0.0, seed=0)
    val_x, val_y = X[40:], y[40:]
    start = float(np.mean((neural.predict(model, val_x, "cpu") - val_y) ** 2))
    neural.fit_early_stopping(model, X, y, lr=1e-2, max_epochs=40, patience=5, val_frac=0.2, device="cpu")
    end = float(np.mean((neural.predict(model, val_x, "cpu") - val_y) ** 2))
    assert end <= start + 1e-9


def test_best_epoch_is_zero_when_nothing_trains():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4, 2)).astype(np.float32)
    y = X[:, -1, 1].astype(np.float32)
    model = neural.build_model("tft", 2, 8, 0.0, seed=0)
    out = neural.fit_early_stopping(model, X, y, lr=0.0, max_epochs=10, patience=3, val_frac=0.2, device="cpu")
    assert out is model and model.best_epoch_ == 0


def test_best_epoch_is_a_trained_epoch_on_learnable_signal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4, 2)).astype(np.float32)
    y = X[:, -1, 1].astype(np.float32)
    model = neural.build_model("tft", 2, 8, 0.0, seed=0)
    neural.fit_early_stopping(model, X, y, lr=1e-2, max_epochs=30, patience=5, val_frac=0.2, device="cpu")
    assert 1 <= model.best_epoch_ <= 30
