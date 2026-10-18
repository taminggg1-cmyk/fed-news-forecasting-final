# Vendored unchanged from gold_gpt2/src/tft.py (Plan 2, spec section 7).
"""A compact Temporal Fusion Transformer, built to take the same tensors as the LSTM.

Why this rather than `pytorch_forecasting.TemporalFusionTransformer`: the reference
implementation wants its own `TimeSeriesDataSet`, does its own normalisation and encoding,
and trains through Lightning. Routing one model through that and the other through a plain
tensor loop would mean the two architectures no longer see identical inputs, and the whole
point of the comparison is that they do. So the TFT's distinctive machinery is implemented
here against the same `(batch, time, features)` tensor the LSTM receives.

Components follow Lim, Arik, Loeff & Pfister (2021), "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting":

  GLU   gated linear unit, the gate on every skip path
  GRN   gated residual network: two dense layers, a GLU, an added skip, layer norm
  VSN   variable selection network: softmax weights over input variables, learned per
        timestep from all of them jointly, with each variable also passed through its own
        GRN. This is what gives a TFT per-feature attribution.
  interpretable multi-head attention: value weights shared across heads and head outputs
        averaged rather than concatenated, so the attention weights remain readable.

Left out deliberately, because this problem does not have them: static covariate encoders
(there are no static variables for a single series) and the separate known-future input
path (the only known-future variables here are calendar dummies, already in the feature
matrix). The output is a point forecast under MSE rather than TFT's usual quantile head,
so that the loss is identical to the LSTM's and the comparison isolates architecture.

This is a TFT-style architecture, and is described as such rather than as the canonical
model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.fc = nn.Linear(size, size * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network. Skips itself entirely when the gate closes."""

    def __init__(self, input_size: int, hidden: int, output_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(hidden)
        self.norm = nn.LayerNorm(output_size)
        self.project = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )
        self.out = nn.Linear(hidden, output_size) if hidden != output_size else None

    def forward(self, x):
        skip = self.project(x) if self.project is not None else x
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.drop(h)
        h = self.glu(h)
        if self.out is not None:
            h = self.out(h)
        return self.norm(skip + h)


class BatchedGRN(nn.Module):
    """n independent GRNs on n scalar inputs, computed in one batched pass.

    Same maths as [GRN(1, hidden, hidden, dropout) for _ in range(n)] - each variable keeps its
    own weights - but as einsums over a variable axis instead of a Python loop. The loop issued
    hundreds of tiny kernels per step and left the GPU ~80% idle (~17 h per walk-forward run).
    """

    def __init__(self, n: int, hidden: int, dropout: float):
        super().__init__()
        self.w_skip = nn.Parameter(torch.empty(n, hidden)); self.b_skip = nn.Parameter(torch.empty(n, hidden))
        self.w1 = nn.Parameter(torch.empty(n, hidden)); self.b1 = nn.Parameter(torch.empty(n, hidden))
        self.w2 = nn.Parameter(torch.empty(n, hidden, hidden)); self.b2 = nn.Parameter(torch.empty(n, hidden))
        self.wg = nn.Parameter(torch.empty(n, hidden, 2 * hidden)); self.bg = nn.Parameter(torch.empty(n, 2 * hidden))
        self.ln_w = nn.Parameter(torch.ones(n, hidden)); self.ln_b = nn.Parameter(torch.zeros(n, hidden))
        self.drop = nn.Dropout(dropout)
        # Initialise each variable's slice exactly as its own nn.Linear would be.
        with torch.no_grad():
            for i in range(n):
                g = GRN(1, hidden, hidden, dropout)
                self.load_from(i, g)

    def load_from(self, i, g):
        self.w_skip[i] = g.project.weight[:, 0]; self.b_skip[i] = g.project.bias
        self.w1[i] = g.fc1.weight[:, 0]; self.b1[i] = g.fc1.bias
        self.w2[i] = g.fc2.weight.T; self.b2[i] = g.fc2.bias
        self.wg[i] = g.glu.fc.weight.T; self.bg[i] = g.glu.fc.bias
        self.ln_w[i] = g.norm.weight; self.ln_b[i] = g.norm.bias

    def forward(self, x):
        # x: (batch, time, n) -> (batch, time, n, hidden)
        x = x.unsqueeze(-1)
        skip = x * self.w_skip + self.b_skip
        h = F.elu(x * self.w1 + self.b1)
        h = torch.einsum("btvh,vhk->btvk", h, self.w2) + self.b2
        h = self.drop(h)
        a, b = (torch.einsum("btvh,vhk->btvk", h, self.wg) + self.bg).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        return F.layer_norm(skip + h, (h.shape[-1],)) * self.ln_w + self.ln_b


class VariableSelection(nn.Module):
    """Learns a softmax weight per input variable, shared across timesteps."""

    def __init__(self, n_vars: int, hidden: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.hidden = hidden
        # Each scalar variable gets its own GRN, so a variable can be transformed before
        # it is weighted rather than only scaled.
        self.var_grns = BatchedGRN(n_vars, hidden, dropout)
        self.weight_grn = GRN(n_vars, hidden, n_vars, dropout)

    def forward(self, x):
        # x: (batch, time, n_vars)
        weights = torch.softmax(self.weight_grn(x), dim=-1).unsqueeze(-1)
        transformed = self.var_grns(x)  # (batch, time, n_vars, hidden)
        combined = (transformed * weights).sum(dim=-2)
        return combined, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Heads share one value projection and their outputs are averaged, not concatenated."""

    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = hidden // n_heads
        self.q = nn.Linear(hidden, self.d_head * n_heads)
        self.k = nn.Linear(hidden, self.d_head * n_heads)
        self.v = nn.Linear(hidden, self.d_head)      # shared across heads
        self.out = nn.Linear(self.d_head, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, _ = x.shape
        q = self.q(x).view(batch, time, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(batch, time, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).unsqueeze(1)                    # (batch, 1, time, d_head)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        # Causal mask: a timestep may not attend to anything after it.
        mask = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        heads = attn @ v                              # (batch, heads, time, d_head)
        averaged = heads.mean(dim=1)                  # interpretable: average the heads
        return self.out(averaged), attn.mean(dim=1)


class CompactTFT(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.vsn = VariableSelection(n_features, hidden, dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.gate_after_lstm = GLU(hidden)
        self.norm_after_lstm = nn.LayerNorm(hidden)
        self.attention = InterpretableMultiHeadAttention(hidden, n_heads, dropout)
        self.gate_after_attn = GLU(hidden)
        self.norm_after_attn = nn.LayerNorm(hidden)
        self.position_wise = GRN(hidden, hidden, hidden, dropout)
        self.gate_final = GLU(hidden)
        self.norm_final = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        self.last_weights = None

    def forward(self, x):
        selected, var_weights = self.vsn(x)
        self.last_weights = var_weights.detach().mean(dim=(0, 1))

        lstm_out, _ = self.lstm(selected)
        h = self.norm_after_lstm(selected + self.gate_after_lstm(lstm_out))

        attended, _ = self.attention(h)
        h2 = self.norm_after_attn(h + self.gate_after_attn(attended))

        h3 = self.position_wise(h2)
        h3 = self.norm_final(h2 + self.gate_final(h3))

        return self.head(h3[:, -1, :]).squeeze(-1)

    def variable_importance(self) -> torch.Tensor:
        """Mean VSN weight per input variable from the last forward pass."""
        return self.last_weights
