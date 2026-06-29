"""FiLM-modulated implicit field decoder f_theta(t, z) -> scalar (Phase 4).

The decoder maps a 1-D coordinate ``t`` (already normalized to [0,1]) and a latent
``z`` to a scalar field value. It is the shared reconstruction unit for BOTH neural
-field families (auto-decoding and amortized), so they differ ONLY in how ``z`` is
produced, never in the decoder.

Why Fourier features (and NOT a plain ReLU MLP)
-----------------------------------------------
A plain coordinate MLP has a strong *spectral bias*: it fits low frequencies first
and struggles with high frequencies. In this harness the matching information lives
in a specific frequency band, so a spectrally-biased field would silently erase the
matching component and confound the whole study. We therefore lift the coordinate
into a Fourier-feature basis ``[sin(2*pi f t), cos(2*pi f t)]`` (Tancik et al.,
2020) with a spread of frequencies, removing the bias by construction. The
frequencies are learnable so the field can adapt its spectral support to the data.

FiLM conditioning
-----------------
The latent modulates the field via per-layer Feature-wise Linear Modulation
(Perez et al., 2018): each hidden block applies ``act((1 + scale) * W h + shift)``
where ``(scale, shift)`` are produced from ``z``. The ``1 +`` keeps the modulation
near identity at initialization for stable optimization (important for the
auto-decoder inner loop).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _as_coords(coords: torch.Tensor) -> torch.Tensor:
    """Accept ``(Batch, M)`` or ``(Batch, M, 1)`` and return ``(Batch, M)``."""
    if coords.dim() == 3:
        if coords.shape[-1] != 1:
            raise ValueError(f"expected last dim 1, got shape {tuple(coords.shape)}")
        coords = coords.squeeze(-1)
    if coords.dim() != 2:
        raise ValueError(f"coords must be (B,M) or (B,M,1), got {tuple(coords.shape)}")
    return coords


class FourierFeatures(nn.Module):
    """Lift a scalar coordinate into ``[sin(2*pi f t), cos(2*pi f t)]`` features.

    Frequencies are initialized geometrically from ``f_min`` to ``f_max`` cycles
    over the unit interval and are learnable, so the field can shift its spectral
    support toward the bands present in the data.
    """

    def __init__(self, n_fourier: int, f_min: float = 0.5, f_max: float = 64.0):
        super().__init__()
        if n_fourier < 1:
            raise ValueError("n_fourier must be >= 1")
        # Geometric spacing covers many octaves with few features.
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_fourier)
        self.freqs = nn.Parameter(freqs)
        self.out_dim = 2 * n_fourier

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = _as_coords(coords)                       # (B, M)
        ang = 2.0 * math.pi * coords.unsqueeze(-1) * self.freqs  # (B, M, F)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, M, 2F)


class FiLMBlock(nn.Module):
    """One residual-free hidden block with FiLM conditioning from ``z``."""

    def __init__(self, hidden: int, latent_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.film = nn.Linear(latent_dim, 2 * hidden)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(z).chunk(2, dim=-1)      # (B, hidden) each
        scale = scale.unsqueeze(1)                        # (B, 1, hidden)
        shift = shift.unsqueeze(1)
        return self.act((1.0 + scale) * self.lin(h) + shift)


class FieldDecoder(nn.Module):
    """FiLM-modulated Fourier-feature field ``f_theta(t, z) -> scalar``.

    Parameters
    ----------
    latent_dim : width of the conditioning latent ``z``.
    hidden     : width of the field MLP.
    depth      : number of FiLM hidden blocks (>= 1).
    n_fourier  : number of Fourier frequencies (feature dim = 2 * n_fourier).
    f_max      : highest Fourier frequency (cycles over the unit coordinate range).
    """

    def __init__(self, latent_dim: int, hidden: int, depth: int,
                 n_fourier: int = 64, f_min: float = 0.5, f_max: float = 64.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.latent_dim = latent_dim
        self.features = FourierFeatures(n_fourier, f_min=f_min, f_max=f_max)
        self.in_proj = nn.Linear(self.features.out_dim, hidden)
        self.in_act = nn.GELU()
        self.blocks = nn.ModuleList(FiLMBlock(hidden, latent_dim) for _ in range(depth))
        self.out = nn.Linear(hidden, 1)

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """coords ``(B, M)`` or ``(B, M, 1)``, z ``(B, latent_dim)`` -> ``(B, M)``."""
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must be (B,{self.latent_dim}), got {tuple(z.shape)}")
        h = self.in_act(self.in_proj(self.features(coords)))   # (B, M, hidden)
        for block in self.blocks:
            h = block(h, z)
        return self.out(h).squeeze(-1)                          # (B, M)
