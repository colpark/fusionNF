"""Auto-decoded neural-field family (Phase 4) -- the FAITHFUL / EXPENSIVE end.

Each signal's latent ``z`` is fit by an *inner-loop* gradient optimization that
minimizes reconstruction MSE against the signal's own samples, using the current
shared ``FieldDecoder`` (auto-decoding / "functa" style; Park et al. 2019, Dupont
et al. 2022). The decoder parameters are trained in the OUTER loop by the normal
training loop via the reconstruction auxiliary loss.

Gradient design (kept simple and correct)
-----------------------------------------
* Inner loop: only ``z`` is optimized. We compute ``d MSE / d z`` with
  ``torch.autograd.grad`` and step ``z`` in place under ``no_grad``, so NO graph is
  retained across inner steps (a first-order scheme -- no meta-gradient through the
  unroll). A small manual Adam makes the few inner steps effective.
* The fitted ``z`` is returned DETACHED, so the classification gradient trains only
  the fusion head (it reads fixed, reconstruction-fit latents).
* ``recon_loss`` recomputes the field at the fitted (detached) latent WITH decoder
  parameters requiring grad, so the outer loop trains the decoder to make
  inner-loop-fit latents reconstruct well.

``encode`` wraps the inner loop in ``torch.enable_grad()`` so it also works when the
caller is under ``torch.no_grad()`` (e.g. ``BaseFusion.representation`` for the
frequency probe).
"""
from __future__ import annotations

import torch

from ..common import BaseFusion
from .field_decoder import FieldDecoder
from .fusion_head import FusionHead


class AutoDecodedField(BaseFusion):
    def __init__(self, hidden: int = 64, latent_dim: int = 32, depth: int = 2,
                 n_fourier: int = 64, f_max: float = 64.0,
                 inner_steps: int = 24, inner_lr: float = 0.05,
                 fusion_mode: str = "mlp"):
        super().__init__()
        self.latent_dim = latent_dim
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.decoder = FieldDecoder(latent_dim, hidden, depth,
                                    n_fourier=n_fourier, f_max=f_max)
        self.head = FusionHead(latent_dim, hidden=hidden, mode=fusion_mode)
        # Learned shared initialization for the per-signal latent (meta-init).
        self.z_init = torch.nn.Parameter(torch.zeros(latent_dim))

    def _fit_latent(self, signal: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Inner-loop fit of ``z`` to reconstruct ``signal`` at ``coords``.

        Returns the fitted latent ``(B, latent_dim)`` detached from the graph.
        """
        B = signal.shape[0]
        with torch.enable_grad():
            z = self.z_init.detach().unsqueeze(0).expand(B, -1).clone()
            z.requires_grad_(True)
            # Manual Adam state (first-order; no graph across steps).
            m = torch.zeros_like(z)
            v = torch.zeros_like(z)
            b1, b2, eps = 0.9, 0.999, 1e-8
            for step in range(1, self.inner_steps + 1):
                pred = self.decoder(coords, z)
                loss = torch.mean((pred - signal) ** 2)
                (g,) = torch.autograd.grad(loss, z)
                with torch.no_grad():
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mhat = m / (1 - b1 ** step)
                    vhat = v / (1 - b2 ** step)
                    z.sub_(self.inner_lr * mhat / (vhat.sqrt() + eps))
        return z.detach()

    def encode(self, A, t_A, B, t_B) -> dict:
        z_a = self._fit_latent(A, t_A)
        z_b = self._fit_latent(B, t_B)
        return {
            "z": torch.cat([z_a, z_b], dim=-1),
            "z_a": z_a, "z_b": z_b,
            "A": A, "t_A": t_A, "B": B, "t_B": t_B,
        }

    def fuse(self, encoded: dict) -> torch.Tensor:
        # Latents are detached; classification trains the fusion head only.
        return self.head(encoded["z_a"], encoded["z_b"])

    def recon_loss(self, encoded: dict) -> torch.Tensor:
        """Reconstruction MSE at the fitted latents (trains the decoder)."""
        pred_a = self.decoder(encoded["t_A"], encoded["z_a"])
        pred_b = self.decoder(encoded["t_B"], encoded["z_b"])
        return torch.mean((pred_a - encoded["A"]) ** 2) + \
            torch.mean((pred_b - encoded["B"]) ** 2)
