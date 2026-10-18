"""MODEL: the LSTM and GRU networks, the input sequences they read, and their training loops.

Used by studies/networks/run_networks.py; the TFT is in tft.py.

Details: Neural forecasters for Plan 2 (spec section 7, secondary architectures).

The paper's LSTM/GRU and the meeting LSTM train mini-batch Adam on MSE for a fixed number of
epochs; the TFT (vendored in tft.py) trains with early stopping on the chronologically last part
of its training rows and keeps the best epoch's weights.
"""
import numpy as np
import torch
import torch.nn as nn

from fed_news.models.tft import CompactTFT


class RNNRegressor(nn.Module):
    """Recurrent layers (LSTM or GRU; one by default) and a linear head on the last time step."""

    def __init__(self, n_features: int, hidden: int, cell: str, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)  # dropout acts between stacked layers only
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        """Forecast from the last time step of each sequence."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(arch: str, n_features: int, hidden: int, dropout: float, seed: int, layers: int = 1,
                heads: int = 4) -> nn.Module:
    """A seeded LSTM, GRU or TFT."""
    torch.manual_seed(seed)
    if arch in ("lstm", "gru"):
        return RNNRegressor(n_features, hidden, arch, layers, dropout)
    if arch == "tft":
        return CompactTFT(n_features, hidden=hidden, n_heads=heads, dropout=dropout)
    raise ValueError(f"unknown arch {arch}")


def make_sequences(Z: np.ndarray, rows, lookback: int) -> np.ndarray:
    """For each row, the `lookback` rows ending at it, as (rows, lookback, features)."""
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return np.empty((0, lookback, Z.shape[1]), dtype=np.float32)
    if rows.min() < lookback - 1:
        raise ValueError(f"row {rows.min()} has less than {lookback} rows of history")
    return np.stack([Z[j - lookback + 1: j + 1] for j in rows]).astype(np.float32)


def _tensor(a, device: str) -> torch.Tensor:
    """Float32 tensor on `device`."""
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def _epoch(model, opt, xt, yt, batch_size: int, rng, device: str) -> None:
    """One pass over the training rows in shuffled mini-batches."""
    model.train()
    order = torch.as_tensor(rng.permutation(len(yt)), device=device)
    for s in range(0, len(yt), batch_size):
        idx = order[s: s + batch_size]
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        opt.step()


def fit_fixed(model, X, y, lr: float, epochs: int, device: str, batch_size: int = 32, seed: int = 0,
              weight_decay: float = 0.0):
    """Mini-batch Adam on MSE (batch 32 is the Keras default; shuffled once per epoch)."""
    model.to(device).train()
    xt, yt = _tensor(X, device), _tensor(y, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        _epoch(model, opt, xt, yt, batch_size, rng, device)
    return model


def fit_validate(model, X, y, X_val, y_val, lr: float, weight_decay: float, batch_size: int, max_epochs: int,
                 patience: int, device: str, seed: int = 0) -> tuple[float, int]:
    """Study 4 tuning: mini-batch Adam on the fold's training rows; return the best validation MSE and its epoch."""
    model.to(device)
    xt, yt, xv, yv = (_tensor(a, device) for a in (X, y, X_val, y_val))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best, best_epoch, wait = np.inf, 1, 0
    for epoch in range(1, max_epochs + 1):
        _epoch(model, opt, xt, yt, batch_size, rng, device)
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(xv), yv))
        if not np.isfinite(v):
            raise ValueError("non-finite validation loss")
        if v < best - 1e-12:
            best, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, best_epoch


def predict(model, X, device: str) -> np.ndarray:
    """Forecasts for X (no gradient)."""
    model.to(device).eval()
    with torch.no_grad():
        return model(_tensor(X, device)).float().cpu().numpy()


# Copied unchanged from fed_news/studies/plan2_neural_original/neural_walk.py (fed-news-forecasting ad2183c),
# the Plan 2 network walk-forward.
def standardise(X, train) -> np.ndarray:
    """Scale X by the mean and standard deviation of the training rows."""
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)
