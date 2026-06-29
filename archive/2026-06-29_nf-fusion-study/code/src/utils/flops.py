"""Parameter and FLOP accounting (operating rule 4).

The central tension in this study lives in the split between representation/
field-fitting cost and fusion cost, so we must be able to measure each
separately. `flop_scope` wraps any callable and returns its FLOPs via torch's
FlopCounterMode (available in torch 2.x); count_params counts trainable params.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def measure_flops(fn: Callable[[], object]) -> int:
    """Run fn() under FLOP counting and return total FLOPs.

    Returns -1 if the FLOP counter is unavailable in this torch build (callers
    log it as 'unknown' rather than failing the run).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return -1
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return int(counter.get_total_flops())


@contextmanager
def flop_scope():
    """Context manager yielding a getter for FLOPs accumulated within the block.

    Usage:
        with flop_scope() as get:
            model(x)
        flops = get()
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        yield lambda: -1
        return
    counter = FlopCounterMode(display=False)
    with counter:
        yield lambda: int(counter.get_total_flops())
