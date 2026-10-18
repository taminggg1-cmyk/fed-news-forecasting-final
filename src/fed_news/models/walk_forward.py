"""MODEL: walk-forward forecasting for the simple models.

For each test row, fit on earlier rows only, then forecast that row. Used by studies/simple_model/.

Details: Walk-forward engine (spec sections 7-8).

Each test row is forecast from a model fit only on earlier rows: T1 on rows up to t-2 (the
paper's protocol), T2 on all earlier meetings. Features are standardised on the training rows
and penalties chosen by expanding-window CV inside them. Any spec that uses an N column on a
test window is refused unless the git tag `prereg` exists.
"""
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config
from fed_news.models import linear as models
from fed_news.features.panels import M_COLS, N_COLS, S_COLS, TAYLOR_COLS


@dataclass
class Spec:
    """One model version: name, task, kind (rw, taylor, primary), columns, and placebo / shuffled flags."""
    name: str
    task: str
    kind: str
    cols: list[str] = field(default_factory=list)
    placebo: bool = False
    shuffle_labels: bool = False


def ladder(task: str) -> list[Spec]:
    """The model versions of the original design (used by the tests of run_spec)."""
    return [
        Spec("rw", task, "rw"),
        Spec("taylor", task, "taylor", TAYLOR_COLS),
        Spec("S", task, "primary", S_COLS),
        Spec("S+N", task, "primary", S_COLS + N_COLS),
        Spec("placebo", task, "primary", S_COLS + N_COLS, placebo=True),
        Spec("shuffled", task, "primary", S_COLS + N_COLS, shuffle_labels=True),
        Spec("S+M", task, "primary", S_COLS + M_COLS),
        Spec("S+M+N", task, "primary", S_COLS + M_COLS + N_COLS),
    ]


def common_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no missing input or target."""
    needed = list(dict.fromkeys(S_COLS + M_COLS + N_COLS + TAYLOR_COLS + ["y_level", "y_change", "y_class"]))
    return df[df[needed].notna().all(axis=1)]


def train_mask(n: int, i: int, task: str) -> np.ndarray:
    """Rows a forecast for row i may train on: up to i-2 for months, all earlier rows for meetings."""
    idx = np.arange(n)
    return idx <= i - 2 if task == "T1" else idx < i


def placebo_matrix(X, train, i, ncols, rng) -> np.ndarray:
    """Inputs with the news columns shuffled across training rows (placebo control)."""
    X = X.copy()
    rows = np.flatnonzero(train)
    X[np.ix_(rows, ncols)] = X[np.ix_(rng.permutation(rows), ncols)]
    X[i, ncols] = X[rng.choice(rows), ncols]
    return X


def prereg_ok(repo=config.REPO) -> bool:
    """True when the news guard is satisfied (see config.PREREG_ELSEWHERE)."""
    if config.PREREG_ELSEWHERE:  # the tag is in the original repository (config.py)
        return True
    out = subprocess.run(["git", "-C", str(repo), "tag", "-l", config.PREREG_TAG],
                         capture_output=True, text=True)
    return out.stdout.strip() == config.PREREG_TAG


def _standardise(X, train):
    """Scale X by the training rows' mean and standard deviation."""
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def run_spec(df: pd.DataFrame, spec: Spec, test_start: str, test_end: str | None = None,
             seed: int = 0, require_prereg: bool = True, proba: bool = False) -> pd.DataFrame:
    # proba=True (Study 4) also saves the ordered logit's class probabilities; forecasts are unchanged.
    """Walk-forward forecasts of one model version: fit on earlier rows, forecast each test row."""
    if require_prereg and any(c in N_COLS for c in spec.cols) and not prereg_ok():
        raise PermissionError(f"{spec.name} uses news columns; tag '{config.PREREG_TAG}' first")
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        raise ValueError("df index must be strictly increasing (sorted, no duplicates)")
    needed = list(dict.fromkeys(spec.cols + ["y_level", "y_change", "y_class", "s_rate_lag"]))
    if df[needed].isna().any().any():
        raise ValueError(f"{spec.name}: NaN in required columns; pass common_rows(df)")
    if not np.isfinite(df[needed].to_numpy(float)).all():
        raise ValueError(f"{spec.name}: non-finite (inf) value in required columns")
    dates = df.index
    test =np.flatnonzero((dates >= pd.Timestamp(test_start))
                          & (dates <= pd.Timestamp(test_end) if test_end else True))
    y_change, y_class = df["y_change"].to_numpy(float), df["y_class"].to_numpy(int)
    lag, y_level = df["s_rate_lag"].to_numpy(float), df["y_level"].to_numpy(float)
    X = df[spec.cols].to_numpy(float) if spec.cols else None
    ncols = [spec.cols.index(c) for c in spec.cols if c in N_COLS]
    rng = np.random.default_rng(seed)
    out = []
    for i in test:
        train = train_mask(len(df), i, spec.task)
        cls, probs = None, (np.nan, np.nan, np.nan)
        if spec.kind == "rw":
            change = 0.0
        elif spec.kind == "taylor":
            change = models.taylor_predict(X[train], y_level[train], X[i]) - lag[i]
        else:
            Xi = placebo_matrix(X, train, i, ncols, rng) if spec.placebo else X
            Z = _standardise(Xi, train)
            yc, ycl = y_change, y_class
            if spec.shuffle_labels:  # spec section 11 control: break the feature-target link in training
                yc, ycl = y_change.copy(), y_class.copy()
                rows = np.flatnonzero(train)
                perm = rng.permutation(rows)
                yc[rows], ycl[rows] = y_change[perm], y_class[perm]
            if spec.task == "T1":
                alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score,
                                              Z[train], yc[train], config.RIDGE_GRID)
                change = float(Ridge(alpha=alpha).fit(Z[train], yc[train]).predict(Z[i:i + 1])[0])
            else:
                lam = models.choose_penalty(models.OrderedLogit, models.logloss_score,
                                            Z[train], ycl[train], config.ORDLOGIT_GRID)
                p = models.OrderedLogit(lam).fit(Z[train], ycl[train]).predict_proba(Z[i:i + 1])[0]
                change = float(p @ models.move_sizes(ycl[train], yc[train]))
                cls, probs = int(p.argmax()), tuple(float(v) for v in p)
        if cls is None:
            cls = int(models.classes_from_change(np.array([change]))[0])
        row = {"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
               "class_true": int(y_class[i]), "class_pred": cls}
        if proba:
            row.update(zip(("p_cut", "p_hold", "p_hike"), probs))
        out.append(row)
    return pd.DataFrame(out)
