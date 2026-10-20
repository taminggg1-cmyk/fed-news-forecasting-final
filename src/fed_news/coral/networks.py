"""MODEL: the LSTM, GRU and TFT with a CORAL output layer, and their training loops.

Used by coral/run_networks.py.

Details: CORAL study, section 7. The networks are the existing ones (models/neural.py, models/tft.py) with their final
linear layer replaced by a CORAL head, so they output two level logits instead of one rate change. Training uses
the CORAL loss on the decision ranks (cut 0, hold 1, hike 2); tuning and early stopping use the validation log loss
of the three class probabilities. The optimiser, mini-batching and shuffling match models/neural.py.
"""
import numpy as np
import torch

from fed_news.coral.core import CoralHead, class_probs, coral_loss, level_probs, levels
from fed_news.models.neural import build_model


def build_coral_model(arch: str, n_features: int, hidden: int, dropout: float, seed: int, layers: int = 1,
                      heads: int = 4) -> torch.nn.Module:
    """A seeded LSTM, GRU or TFT whose head is a CORAL layer (the existing forwards then return (rows, 2))."""
    model = build_model(arch, n_features, hidden, dropout, seed, layers=layers, heads=heads)
    torch.manual_seed(seed + 1)
    model.head = CoralHead(hidden)
    return model


def _tensor(a, device: str) -> torch.Tensor:
    """Float32 tensor on `device`."""
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def _epoch(model, opt, xt, lv, batch_size: int, rng, device: str) -> None:
    """One pass over the training rows in shuffled mini-batches, minimising the CORAL loss."""
    model.train()
    order = torch.as_tensor(rng.permutation(len(lv)), device=device)
    for s in range(0, len(lv), batch_size):
        idx = order[s: s + batch_size]
        opt.zero_grad()
        loss = coral_loss(model(xt[idx]), lv[idx])
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        opt.step()


def val_log_loss(lp: np.ndarray, ranks) -> float:
    """Mean log loss of the true class under the class probabilities implied by the level probabilities."""
    p = np.clip(class_probs(lp), 1e-12, None)
    r = np.asarray(ranks, dtype=int)
    return float(-np.mean(np.log(p[np.arange(len(r)), r])))


def predict_levels(model, X, device: str) -> np.ndarray:
    """Level probabilities P(rank >= 1), P(rank >= 2) for X (no gradient)."""
    model.to(device).eval()
    with torch.no_grad():
        return level_probs(model(_tensor(X, device)))


def fit_fixed_coral(model, X, ranks, lr: float, epochs: int, device: str, batch_size: int = 32, seed: int = 0,
                    weight_decay: float = 0.0):
    """Mini-batch Adam on the CORAL loss for a fixed number of epochs (the final fit at each refit)."""
    model.to(device).train()
    xt, lv = _tensor(X, device), levels(ranks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        _epoch(model, opt, xt, lv, batch_size, rng, device)
    return model


def fit_validate_coral(model, X, ranks, X_val, ranks_val, lr: float, weight_decay: float, batch_size: int,
                       max_epochs: int, patience: int, device: str, seed: int = 0) -> tuple[float, int]:
    """Tuning: train on the fold's rows, stop early on the validation log loss; return the best value and epoch."""
    model.to(device)
    xt, lv, xv = _tensor(X, device), levels(ranks).to(device), _tensor(X_val, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best, best_epoch, wait = np.inf, 1, 0
    for epoch in range(1, max_epochs + 1):
        _epoch(model, opt, xt, lv, batch_size, rng, device)
        model.eval()
        with torch.no_grad():
            v = val_log_loss(level_probs(model(xv)), ranks_val)
        if not np.isfinite(v):
            raise ValueError("non-finite validation loss")
        if v < best - 1e-12:
            best, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, best_epoch
