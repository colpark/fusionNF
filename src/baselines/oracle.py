"""Oracle baseline -- the correspondence upper bound (Phase 3).

The oracle is *not* a realistic method: it is allowed to see the ground-truth
instantaneous-frequency trajectories ``f_A`` and ``f_B`` that the generator used.
Its only job is to establish the ceiling for the task -- if the matching signal is
present at all, the oracle should recover it almost perfectly.

Mechanism
---------
For a labelled pair, ``f_A`` lives on grid ``t_A`` and ``f_B`` on grid ``t_B``
(different rate, possibly jittered). We resample ``f_B`` onto ``t_A`` with linear
interpolation and take the Pearson correlation between ``f_A`` and the resampled
``f_B``. For label-1 pairs the two trajectories are identical up to resampling, so
the correlation is ~1.0; for label-0 pairs the trajectories are independent draws,
giving a spread of correlations well below 1.0. A single threshold (fit on train)
then separates the classes.
"""
from __future__ import annotations

import numpy as np

from ..config import DataConfig
from ..data.dataset import SignalPairDataset
from ..data.generator import Sample
from .threshold import accuracy_at, fit_threshold, pearson


def correspondence_stat(sample: Sample) -> float:
    """Oracle correspondence statistic for a single pair.

    Resamples ``f_B`` onto the ``t_A`` grid via :func:`numpy.interp` and returns the
    Pearson correlation with ``f_A``. ~1.0 for true matches, lower for mismatches.
    """
    t_A = np.asarray(sample.t_A, dtype=np.float64)
    f_A = np.asarray(sample.f_A, dtype=np.float64)
    t_B = np.asarray(sample.t_B, dtype=np.float64)
    f_B = np.asarray(sample.f_B, dtype=np.float64)
    # numpy.interp requires increasing xp; t_B is monotonic non-decreasing by
    # construction. Points of t_A outside t_B's range clamp to the endpoints.
    f_B_on_A = np.interp(t_A, t_B, f_B)
    return pearson(f_A, f_B_on_A)


def _stats_and_labels(data_cfg: DataConfig, base_seed: int, n: int, split: str):
    """Compute the oracle statistic and label for ``n`` samples of a split."""
    ds = SignalPairDataset(data_cfg, base_seed, split, n)
    stats = np.empty(n, dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = ds.raw(i)
        stats[i] = correspondence_stat(s)
        labels[i] = int(s.label)
    return stats, labels


def oracle_accuracy(data_cfg: DataConfig, base_seed: int, n: int,
                    split: str = "test") -> dict:
    """Fit the oracle threshold on train and evaluate on ``split``.

    The threshold that maximizes accuracy on ``n`` TRAIN samples is frozen, then
    applied to ``n`` samples of the requested ``split``. Returns
    ``{"accuracy", "threshold", "n"}``. Expected to be ~1.0 on easy and hard.
    """
    train_stats, train_labels = _stats_and_labels(data_cfg, base_seed, n, "train")
    threshold = fit_threshold(train_stats, train_labels)

    if split == "train":
        eval_stats, eval_labels = train_stats, train_labels
    else:
        eval_stats, eval_labels = _stats_and_labels(data_cfg, base_seed, n, split)

    accuracy = accuracy_at(eval_stats, eval_labels, threshold)
    return {"accuracy": accuracy, "threshold": threshold, "n": n}
