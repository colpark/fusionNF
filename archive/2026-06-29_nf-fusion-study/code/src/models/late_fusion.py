"""Late-fusion family (Phase 4).

Each modality is encoded *independently* by a 1-D CNN into a fixed-width latent,
the two latents are concatenated, and a small MLP performs the cross-modal decision
at the very end -- hence "late" fusion. This is the cheap, modular reference point:
representation is per-modality and fusion touches only two pooled vectors, so its
fusion-FLOPs are O(latent_dim), independent of sequence length.

``SignalCNN`` is defined here and reused by the amortized neural-field encoder so
the two cheap encoders share an identical convolutional backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseFusion, mlp


class SignalCNN(nn.Module):
    """Strided 1-D CNN encoder: ``(Batch, N) -> (Batch, out_dim)``.

    The signal is treated as a single-channel sequence ``(Batch, 1, N)``. Three
    strided convolutions downsample by ~8x while widening to ``hidden`` channels,
    then a final conv maps to ``out_dim``. A global pooling reduces the variable
    length ``L`` to a fixed vector, so the encoder is agnostic to the (config-fixed
    but modality-dependent) input length.

    pooling : ``"mean"`` | ``"attention"`` | ``"last"``.
    """

    def __init__(self, hidden: int, out_dim: int, pooling: str = "mean"):
        super().__init__()
        if pooling not in {"mean", "attention", "last"}:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, out_dim, kernel_size=3, stride=2, padding=1),
        )
        if pooling == "attention":
            # Learned attention pooling: a score per position, softmax-weighted sum.
            self.attn = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))             # (B, out_dim, L)
        h = h.transpose(1, 2)                     # (B, L, out_dim)
        if self.pooling == "mean":
            return h.mean(dim=1)
        if self.pooling == "last":
            return h[:, -1, :]
        # attention pooling
        scores = self.attn(h)                     # (B, L, 1)
        weights = torch.softmax(scores, dim=1)
        return (weights * h).sum(dim=1)           # (B, out_dim)


class LateFusion(BaseFusion):
    """Per-modality CNN encoders + concat + MLP fusion head.

    encode -> {"z": concat(z_A, z_B), "z_a": z_A, "z_b": z_B}
    fuse   -> small MLP over the concatenated latent -> logit ``(Batch,)``.
    """

    def __init__(self, hidden: int = 64, latent_dim: int = 32,
                 depth: int = 2, pooling: str = "mean"):
        super().__init__()
        self.enc_a = SignalCNN(hidden, latent_dim, pooling=pooling)
        self.enc_b = SignalCNN(hidden, latent_dim, pooling=pooling)
        sizes = [2 * latent_dim] + [hidden] * max(1, depth) + [1]
        self.head = mlp(sizes)

    def encode(self, A, t_A, B, t_B) -> dict:
        z_a = self.enc_a(A)
        z_b = self.enc_b(B)
        return {"z": torch.cat([z_a, z_b], dim=-1), "z_a": z_a, "z_b": z_b}

    def fuse(self, encoded: dict) -> torch.Tensor:
        return self.head(encoded["z"]).squeeze(-1)
