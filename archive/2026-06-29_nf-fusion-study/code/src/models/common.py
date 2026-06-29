"""Shared interface for the four fusion families (Phase 4).

The brief requires logging FLOPs separately for (a) representation / field-fitting
and (b) fusion, because the central tension lives in that split. To make that
measurable uniformly, every family implements a two-stage contract:

    encoded = model.encode(A, t_A, B, t_B)   # representation stage
    logit   = model.fuse(encoded)            # fusion stage
    forward(...) == fuse(encode(...))

`encode` returns a dict that must contain a flat representation tensor under key
"z" (shape (Batch, D)) for the frequency probe (Phase 5). The accounting harness
measures encode-FLOPs vs fuse-FLOPs by wrapping each call in a FLOP scope.

All inputs are already normalized (signals standardized on train; coords in [0,1]):
    A:   (Batch, N_A)    t_A: (Batch, N_A)
    B:   (Batch, N_B)    t_B: (Batch, N_B)
Output logit: (Batch,) raw logits for BCEWithLogits.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class FusionModel(Protocol):
    def encode(self, A, t_A, B, t_B) -> dict: ...
    def fuse(self, encoded: dict) -> torch.Tensor: ...
    def forward(self, A, t_A, B, t_B) -> torch.Tensor: ...
    def representation(self, A, t_A, B, t_B) -> torch.Tensor: ...


class BaseFusion(nn.Module):
    """Mixin providing forward()/representation() from encode()/fuse()."""

    def forward(self, A, t_A, B, t_B) -> torch.Tensor:
        return self.fuse(self.encode(A, t_A, B, t_B))

    @torch.no_grad()
    def representation(self, A, t_A, B, t_B) -> torch.Tensor:
        return self.encode(A, t_A, B, t_B)["z"].detach()

    # subclasses implement:
    def encode(self, A, t_A, B, t_B) -> dict:  # pragma: no cover
        raise NotImplementedError

    def fuse(self, encoded: dict) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def mlp(sizes: list[int], act=nn.GELU, last_act=False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)
