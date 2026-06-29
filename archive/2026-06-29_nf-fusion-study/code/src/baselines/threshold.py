"""Single-threshold classifier utilities shared by the Phase-3 baselines.

Both baselines reduce a pair (A, B) to a scalar *correspondence statistic* where a
larger value indicates a stronger A<->B match (label 1). Classification is then a
single learned threshold: predict 1 iff stat >= threshold. The threshold is fit on
a TRAIN split (maximize train accuracy) and frozen for evaluation, so no label
information leaks from the evaluation split.

The fit is exhaustive over all distinct decision boundaries, so it is the global
optimum for the (fixed-direction) thresholded rule -- deterministic and tie-broken
toward the boundary nearest the data centre for stability.
"""
from __future__ import annotations

import numpy as np


def accuracy_at(stats: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Accuracy of the rule ``predict 1 iff stat >= threshold``."""
    stats = np.asarray(stats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (stats >= threshold).astype(np.int64)
    return float(np.mean(preds == labels))


def fit_threshold(stats: np.ndarray, labels: np.ndarray) -> float:
    """Pick the threshold maximizing train accuracy for ``stat >= threshold -> 1``.

    Candidate boundaries are every distinct statistic value plus one above the max
    (the "predict all 0" boundary). Among optima, the threshold closest to the
    median statistic is chosen for a stable, well-centred decision boundary.
    """
    stats = np.asarray(stats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if stats.size == 0:
        raise ValueError("fit_threshold: empty statistic array")

    uniq = np.unique(stats)
    # Threshold just above the maximum -> predicts all-zero; covers that partition.
    candidates = np.concatenate([uniq, [uniq[-1] + 1.0]])

    best_acc = -1.0
    best_thr = candidates[0]
    centre = float(np.median(stats))
    for thr in candidates:
        acc = accuracy_at(stats, labels, float(thr))
        if acc > best_acc or (acc == best_acc and abs(thr - centre) < abs(best_thr - centre)):
            best_acc = acc
            best_thr = float(thr)
    return best_thr


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation in float64; returns 0.0 if either input is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
