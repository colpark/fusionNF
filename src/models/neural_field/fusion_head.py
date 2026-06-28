"""Fusion head over the two per-modality latents (Phase 4, shared by NF variants).

Both neural-field families reduce a pair to two latents ``z_A, z_B`` and then must
decide correspondence. This head is the cross-modal combiner and is deliberately
kept identical across the auto-decoded and amortized variants so that any accuracy
gap between them is attributable to the *latent quality* (how ``z`` is produced),
not to the classifier on top.

Two combiners are provided:

* ``"mlp"``   -- concatenate ``[z_A, z_B, z_A * z_B, |z_A - z_B|]`` and run a small
                MLP. The product / absolute-difference terms give the head explicit
                access to agreement features, which is exactly the correspondence
                signal.
* ``"xattn"`` -- a single multi-head cross-attention block treating the two latents
                as a length-2 token sequence, then pool and project to a logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..common import mlp


class FusionHead(nn.Module):
    """Map ``(z_A, z_B)`` -> scalar logit ``(Batch,)``.

    Parameters
    ----------
    latent_dim : width of each per-modality latent.
    hidden     : hidden width of the combiner.
    mode       : ``"mlp"`` (default) or ``"xattn"``.
    n_heads    : attention heads when ``mode == "xattn"``.
    """

    def __init__(self, latent_dim: int, hidden: int = 64,
                 mode: str = "mlp", n_heads: int = 4):
        super().__init__()
        if mode not in {"mlp", "xattn"}:
            raise ValueError(f"unknown fusion-head mode {mode!r}")
        self.mode = mode
        self.latent_dim = latent_dim

        if mode == "mlp":
            # [z_A, z_B, z_A*z_B, |z_A - z_B|] -> 4 * latent_dim
            self.net = mlp([4 * latent_dim, hidden, hidden, 1])
        else:
            # Project latents to a common model width, attend across the 2 tokens.
            d = hidden
            self.proj = nn.Linear(latent_dim, d)
            self.type_emb = nn.Parameter(torch.zeros(2, d))
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.ff = mlp([d, hidden, 1])

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            feats = torch.cat([z_a, z_b, z_a * z_b, (z_a - z_b).abs()], dim=-1)
            return self.net(feats).squeeze(-1)

        # cross-attention combiner
        tokens = torch.stack([self.proj(z_a), self.proj(z_b)], dim=1)  # (B, 2, d)
        tokens = tokens + self.type_emb.unsqueeze(0)
        attended, _ = self.attn(tokens, tokens, tokens)               # (B, 2, d)
        pooled = self.norm(attended).mean(dim=1)                      # (B, d)
        return self.ff(pooled).squeeze(-1)
