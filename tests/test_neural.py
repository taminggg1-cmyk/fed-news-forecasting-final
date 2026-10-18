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
