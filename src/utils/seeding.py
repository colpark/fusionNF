"""Global determinism control (operating rule 3).

set_seed() seeds Python, NumPy, and PyTorch (CPU + CUDA). enable_determinism()
additionally requests deterministic algorithms so two runs with the same config
match within tolerance. We do NOT hard-fail on nondeterministic ops (some have no
deterministic kernel on MPS/CPU); instead we set warn_only and rely on the
determinism test to catch real drift.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: MPS/CPU lack deterministic kernels for some ops; surface a warning
    # rather than crash, and let tests/test_determinism.py verify actual stability.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seeded_generator(seed: int) -> torch.Generator:
    """A torch.Generator for dataloader / sampling that doesn't touch global RNG."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def np_rng(seed: int) -> np.random.Generator:
    """A NumPy Generator for the data generator -- per-sample seeding lives here."""
    return np.random.default_rng(seed)
