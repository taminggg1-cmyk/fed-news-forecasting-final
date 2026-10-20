"""MODEL: the CORAL output layer, its loss, and the conversions between levels, classes and tone.

Used by every other module in fed_news/coral/.

Details: CORAL study, section 2 (Cao, Mirjalili and Raschka 2020, arXiv:1901.07884). K = 3 ordered classes become
K - 1 = 2 binary levels y_k = 1{rank >= k}: rank 0 = (0, 0), 1 = (1, 0), 2 = (1, 1). One weight vector without a
bias is shared by both levels; each level has its own bias, initialised to (1.0, 0.5) as in the authors' package
coral-pytorch. The loss is the sum of the two binary cross-entropies (equal task weights), averaged over rows.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 3
# Stored stance codes (download/stance_labels.LABELS: dovish 0, hawkish 1, neutral 2) -> ordinal ranks
# (dovish 0 < neutral 1 < hawkish 2).
TONE_RANK = {0: 0, 2: 1, 1: 2}


class CoralHead(nn.Module):
    """One shared score w.h(x) (no bias) plus a separate bias per level: logits of shape (rows, K - 1)."""

    def __init__(self, size_in: int, n_classes: int = K):
        super().__init__()
        self.weight = nn.Linear(size_in, 1, bias=False)
        self.bias = nn.Parameter(torch.arange(n_classes - 1, 0, -1).float() / (n_classes - 1))

    def forward(self, h):
        """Level logits for each row of h."""
        return self.weight(h) + self.bias


def init_biases(head: CoralHead, ranks) -> None:
    """Design amendment 2: start the biases at the logits of the cumulative class shares of the training rows,
    b_k = logit(share of rank >= k), shares clipped to [0.001, 0.999] (the intercept-only ordered fit, the start
    values of models.linear.OrderedLogit). With (1.0, 0.5) the gap b_1 - b_2 = 0.5 caps P(middle class) at 0.124
    until the biases move, which small learning rates barely allow."""
    r = np.asarray(ranks, dtype=int)
    n_levels = head.bias.numel()
    shares = np.clip([(r >= k).mean() for k in range(1, n_levels + 1)], 1e-3, 1 - 1e-3)
    with torch.no_grad():
        head.bias.copy_(torch.as_tensor(np.log(shares / (1 - shares)), dtype=head.bias.dtype))


def param_groups(model: nn.Module, lr: float, weight_decay: float, bias_lr: float | None = None) -> list[dict]:
    """Optimizer groups: the CORAL biases (thresholds) without weight decay, at `bias_lr` (default `lr`); every
    other parameter at `lr` with `weight_decay`."""
    bias = model.head.bias
    rest = [p for p in model.parameters() if p is not bias]
    return [{"params": rest, "lr": lr, "weight_decay": weight_decay},
            {"params": [bias], "lr": lr if bias_lr is None else bias_lr, "weight_decay": 0.0}]


def levels(ranks, n_classes: int = K) -> torch.Tensor:
    """Binary level targets 1{rank >= k}, k = 1..K-1, as a float tensor of shape (rows, K - 1)."""
    r = torch.as_tensor(np.asarray(ranks), dtype=torch.long).reshape(-1, 1)
    return (r >= torch.arange(1, n_classes).reshape(1, -1)).float()


def coral_loss(logits, lv, reduction: str = "mean"):
    """CORAL loss: minus the summed log-likelihood of the binary levels, per row, then averaged (or summed)."""
    term = F.logsigmoid(logits) * lv + (F.logsigmoid(logits) - logits) * (1 - lv)
    per_row = -term.sum(dim=1)
    return per_row.mean() if reduction == "mean" else per_row.sum() if reduction == "sum" else per_row


def level_probs(logits) -> np.ndarray:
    """P(rank >= k) for each level, as a numpy array."""
    return torch.sigmoid(torch.as_tensor(logits).detach().float()).cpu().numpy()


def class_probs(lp: np.ndarray) -> np.ndarray:
    """Class probabilities from level probabilities: (1 - P1, P1 - P2, P2); clipped at 0 and renormalised, so a
    model whose levels cross (biases out of order) still gives valid probabilities."""
    lp = np.asarray(lp, dtype=float)
    p = np.column_stack([1.0 - lp[:, 0], lp[:, 0] - lp[:, 1], lp[:, 1]])
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def predict_rank(lp: np.ndarray) -> np.ndarray:
    """The paper's rule: the predicted rank is the number of levels with probability above 0.5."""
    return (np.asarray(lp) > 0.5).sum(1).astype(int)


def biases_ordered(head: CoralHead) -> bool:
    """True when b_1 >= b_2 >= ..., which makes P(rank >= k) fall with k for every input (rank consistency)."""
    b = head.bias.detach().cpu().numpy()
    return bool(np.all(np.diff(b) <= 0))


def tone_from_levels(lp: np.ndarray) -> np.ndarray:
    """Tone = P(hawkish) - P(dovish) = P2 - (1 - P1), the same scale as the earlier ChronoBERT tone."""
    lp = np.asarray(lp, dtype=float)
    return lp[:, 1] - (1.0 - lp[:, 0])
