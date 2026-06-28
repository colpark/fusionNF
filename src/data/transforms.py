"""Normalization fit on TRAIN ONLY (operating rule 5).

Signals are standardized per modality using mean/std estimated from the train
split. Coordinates are mapped to [0,1] by the configured duration (no fitting
needed). Trajectory targets f(t) are left in Hz; the probe standardizes them
itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import DataConfig
from .dataset import SignalPairDataset


@dataclass
class NormStats:
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float
    duration: float


def fit_norm(data_cfg: DataConfig, base_seed: int, n_train: int,
             max_samples: int = 256) -> NormStats:
    ds = SignalPairDataset(data_cfg, base_seed, "train", n_train)
    n = min(n_train, max_samples)
    a_vals, b_vals = [], []
    for i in range(n):
        s = ds.raw(i)
        a_vals.append(s.A)
        b_vals.append(s.B)
    a = np.concatenate(a_vals)
    b = np.concatenate(b_vals)
    return NormStats(
        a_mean=float(a.mean()), a_std=float(a.std() + 1e-8),
        b_mean=float(b.mean()), b_std=float(b.std() + 1e-8),
        duration=float(data_cfg.duration),
    )


def apply_norm(batch: dict, stats: NormStats) -> dict:
    """Return a new batch dict with standardized signals and [0,1] coordinates."""
    out = dict(batch)
    out["A"] = (batch["A"] - stats.a_mean) / stats.a_std
    out["B"] = (batch["B"] - stats.b_mean) / stats.b_std
    out["t_A"] = batch["t_A"] / stats.duration
    out["t_B"] = batch["t_B"] / stats.duration
    return out
