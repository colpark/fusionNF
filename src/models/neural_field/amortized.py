"""Amortized neural-field family (Phase 4) -- the CHEAP end of amortization.

A feed-forward 1-D CNN predicts the per-signal latent ``z`` in a *single* forward
pass (no inner optimization). The latent conditions the shared ``FieldDecoder`` for
reconstruction, and the same two latents feed the shared ``FusionHead`` for the
correspondence decision.

This is the inexpensive counterpart to the auto-decoded field: it amortizes the
field-fitting cost into encoder weights learned across the dataset, trading the
faithful per-signal optimization for a single forward pass. Comparing the two NF
variants isolates the cost/quality effect of amortization while holding the
decoder and fusion head fixed.

Training signal:
* classification gradient flows through the encoder (``z`` is differentiable),
* the reconstruction auxiliary loss trains BOTH the encoder and the decoder.
"""
from __future__ import annotations

import torch

from ..common import BaseFusion
from ..late_fusion import SignalCNN
from .field_decoder import FieldDecoder
from .fusion_head import FusionHead


class AmortizedField(BaseFusion):
    def __init__(self, hidden: int = 64, latent_dim: int = 32, depth: int = 2,
                 n_fourier: int = 64, f_max: float = 64.0,
                 fusion_mode: str = "mlp"):
        super().__init__()
        # Two encoders (modalities have different statistics / lengths).
        self.enc_a = SignalCNN(hidden, latent_dim, pooling="mean")
        self.enc_b = SignalCNN(hidden, latent_dim, pooling="mean")
        self.decoder = FieldDecoder(latent_dim, hidden, depth,
                                    n_fourier=n_fourier, f_max=f_max)
        self.head = FusionHead(latent_dim, hidden=hidden, mode=fusion_mode)

    def encode(self, A, t_A, B, t_B) -> dict:
        z_a = self.enc_a(A)
        z_b = self.enc_b(B)
        return {
            "z": torch.cat([z_a, z_b], dim=-1),
            "z_a": z_a, "z_b": z_b,
            "A": A, "t_A": t_A, "B": B, "t_B": t_B,
        }

    def fuse(self, encoded: dict) -> torch.Tensor:
        return self.head(encoded["z_a"], encoded["z_b"])

    def recon_loss(self, encoded: dict) -> torch.Tensor:
        """Reconstruction MSE for both modalities (trains encoder + decoder)."""
        pred_a = self.decoder(encoded["t_A"], encoded["z_a"])
        pred_b = self.decoder(encoded["t_B"], encoded["z_b"])
        return torch.mean((pred_a - encoded["A"]) ** 2) + \
            torch.mean((pred_b - encoded["B"]) ** 2)
