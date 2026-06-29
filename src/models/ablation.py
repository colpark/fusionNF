"""Architecture-only ablations to separate, in LAINR, the *neural-field interface*
from the *multi-band Fourier spectral basis* as the cause of the SNR-robustness gap.

All variants reuse the shared FusionHead (mode=mlp, features [z_A,z_B,z_A*z_B,|z_A-z_B|])
and expose a flat `z` for the probe, so accuracy differences reflect representation
quality, not the classifier. Existing families (late/early/nf_lainr/nf_omnifield) are
untouched; these are additive.

Variants:
  - LateMBFF       : late-fusion CNN encoder + the SAME multi-band Fourier reconstruction
                     decoder/aux as LAINR (no field: no coordinate cross-attention,
                     no locality bias, no continuous query). Spectral basis WITHOUT the field.
  - LAINRReLU      : LAINR exactly (token encoder, coordinate cross-attention, locality
                     bias, recon aux) but the multi-band Fourier decoder is replaced by a
                     plain ReLU coordinate-MLP of matched depth/width. Field WITHOUT the
                     multi-band basis.
  - LAINRSingleSigma: LAINR with a single wide Gaussian Fourier-feature decoder (one sigma,
                     matched total feature count) instead of the geometric coarse-to-fine
                     bands. Tests multi-scale structuring vs "having Fourier features".
  - (n_bands in {1,2,8}) reuse the existing LAINRField with n_bands set in the registry.

The multi-band construction here is copied verbatim from
``neural_field/lainr._LocalityAwareDecoder`` so the spectral basis is identical; the
locality-aware cross-attention modulation is likewise an exact copy, so the only thing
that changes between LAINR and LAINRReLU/LAINRSingleSigma is the decoder *basis*.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseFusion
from .late_fusion import SignalCNN
from .neural_field.lainr import _TokenEncoder
from .neural_field.fusion_head import FusionHead


# --------------------------------------------------------------------------- #
# Shared building blocks (copies of LAINR internals; LAINR itself is unchanged).
# --------------------------------------------------------------------------- #
def _band_freqs(n_bands: int, feats_per_band: int, f_max: float) -> torch.Tensor:
    """Geometric coarse-to-fine bands, identical to LAINR._LocalityAwareDecoder."""
    edges = torch.logspace(math.log10(1.0), math.log10(f_max), n_bands + 1)
    freqs = [torch.logspace(math.log10(edges[b].item()), math.log10(edges[b + 1].item()),
                            feats_per_band) for b in range(n_bands)]
    return torch.stack(freqs)                                  # (n_bands, feats_per_band)


class _LocalityModulation(nn.Module):
    """Exact copy of LAINR's locality-aware cross-attention modulation."""

    def __init__(self, hidden: int):
        super().__init__()
        self.scale = hidden ** -0.5
        self.q_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.log_locality = nn.Parameter(torch.tensor(2.0))

    def forward(self, coords2, tokens, tau):                   # (B,M),(B,T,H),(B,T)->(B,M,H)
        q = self.q_proj(coords2.unsqueeze(-1))
        k = self.k_proj(tokens); v = self.v_proj(tokens)
        logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        dist2 = (coords2.unsqueeze(-1) - tau.unsqueeze(1)) ** 2
        logits = logits - F.softplus(self.log_locality) * dist2
        return torch.bmm(torch.softmax(logits, dim=-1), v)


class MultiBandReconDecoder(nn.Module):
    """Multi-band coarse-to-fine FiLM decoder (LAINR's basis), driven by a modulation m.

    m may be (B,M,H) [per-coord, as in LAINR] or (B,1,H) [a single global latent, as in
    LateMBFF]; broadcasting handles both. Used WITHOUT cross-attention in LateMBFF.
    """

    def __init__(self, hidden: int, n_bands: int = 4, feats_per_band: int = 16,
                 f_max: float = 64.0):
        super().__init__()
        self.hidden = hidden; self.n_bands = n_bands
        self.register_buffer("band_freqs", _band_freqs(n_bands, feats_per_band, f_max))
        self.band_in = nn.ModuleList(nn.Linear(2 * feats_per_band, hidden) for _ in range(n_bands))
        self.band_film = nn.ModuleList(nn.Linear(hidden, 2 * hidden) for _ in range(n_bands))
        self.act = nn.GELU(); self.out = nn.Linear(hidden, 1)

    def forward(self, coords, m):                              # coords (B,M); m (B,M|1,H)
        c2 = coords.squeeze(-1) if coords.dim() == 3 else coords
        if m.dim() == 2:
            m = m.unsqueeze(1)
        h = torch.zeros(c2.shape[0], c2.shape[1], self.hidden, device=c2.device, dtype=m.dtype)
        for b in range(self.n_bands):
            ang = 2.0 * math.pi * c2.unsqueeze(-1) * self.band_freqs[b]
            feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            hb = self.band_in[b](feat)
            scale, shift = self.band_film[b](m).chunk(2, dim=-1)
            h = h + self.act((1.0 + scale) * hb + shift)
        return self.out(h).squeeze(-1)


class _ReLUDecoder(nn.Module):
    """LAINR field interface (locality cross-attention) + plain ReLU coordinate-MLP basis.

    Capacity matched to the multi-band decoder: n_blocks = n_bands FiLM-ReLU blocks at
    width `hidden`. Input is the RAW coordinate (no Fourier features) -> spectral bias.
    """

    def __init__(self, hidden: int, n_blocks: int = 4):
        super().__init__()
        self.mod = _LocalityModulation(hidden)
        self.in_proj = nn.Linear(1, hidden)
        self.lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_blocks))
        self.film = nn.ModuleList(nn.Linear(hidden, 2 * hidden) for _ in range(n_blocks))
        self.act = nn.ReLU(); self.out = nn.Linear(hidden, 1)

    def forward(self, coords, tokens, tau):
        c2 = coords.squeeze(-1) if coords.dim() == 3 else coords
        m = self.mod(c2, tokens, tau)
        h = self.act(self.in_proj(c2.unsqueeze(-1)))
        for lin, film in zip(self.lin, self.film):
            scale, shift = film(m).chunk(2, dim=-1)
            h = self.act((1.0 + scale) * lin(h) + shift)
        return self.out(h).squeeze(-1)


class _SingleSigmaDecoder(nn.Module):
    """LAINR field interface + a SINGLE wide Gaussian Fourier-feature basis (one sigma),
    matched total feature count (n_freq = n_bands*feats_per_band). Tests multi-scale
    structuring vs merely having Fourier features."""

    def __init__(self, hidden: int, n_freq: int = 64, f_max: float = 64.0, seed: int = 0):
        super().__init__()
        self.mod = _LocalityModulation(hidden)
        g = torch.Generator().manual_seed(seed)
        # single Gaussian scale; std ~ f_max/3 so support ~[0, f_max] (documented choice)
        self.register_buffer("B", torch.randn(n_freq, generator=g) * (f_max / 3.0))
        self.in_proj = nn.Linear(2 * n_freq, hidden)
        self.film = nn.Linear(hidden, 2 * hidden)
        self.act = nn.GELU(); self.out = nn.Linear(hidden, 1)

    def forward(self, coords, tokens, tau):
        c2 = coords.squeeze(-1) if coords.dim() == 3 else coords
        m = self.mod(c2, tokens, tau)
        ang = 2.0 * math.pi * c2.unsqueeze(-1) * self.B
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        hb = self.in_proj(feat)
        scale, shift = self.film(m).chunk(2, dim=-1)
        h = self.act((1.0 + scale) * hb + shift)
        return self.out(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Variant 1: spectral basis WITHOUT the field.
# --------------------------------------------------------------------------- #
class LateMBFF(BaseFusion):
    def __init__(self, hidden=64, latent_dim=32, depth=2, n_bands=4, feats_per_band=16,
                 f_max=64.0, pooling="mean"):
        super().__init__()
        self.enc_a = SignalCNN(hidden, latent_dim, pooling=pooling)
        self.enc_b = SignalCNN(hidden, latent_dim, pooling=pooling)
        self.lift_a = nn.Linear(latent_dim, hidden)
        self.lift_b = nn.Linear(latent_dim, hidden)
        self.dec_a = MultiBandReconDecoder(hidden, n_bands, feats_per_band, f_max)
        self.dec_b = MultiBandReconDecoder(hidden, n_bands, feats_per_band, f_max)
        self.head = FusionHead(latent_dim, hidden=hidden, mode="mlp")

    def encode(self, A, t_A, B, t_B) -> dict:
        z_a = self.enc_a(A); z_b = self.enc_b(B)
        return {"z": torch.cat([z_a, z_b], -1), "z_a": z_a, "z_b": z_b,
                "A": A, "t_A": t_A, "B": B, "t_B": t_B}

    def fuse(self, e) -> torch.Tensor:
        return self.head(e["z_a"], e["z_b"])

    def recon_loss(self, e) -> torch.Tensor:
        pa = self.dec_a(e["t_A"], self.lift_a(e["z_a"]))      # global latent -> (B,1,H)
        pb = self.dec_b(e["t_B"], self.lift_b(e["z_b"]))
        return torch.mean((pa - e["A"]) ** 2) + torch.mean((pb - e["B"]) ** 2)


# --------------------------------------------------------------------------- #
# Variants 2 & 3: field WITH alternative bases.
# --------------------------------------------------------------------------- #
class _LAINRWithDecoder(BaseFusion):
    """LAINR token encoder + locality field interface, with a pluggable decoder basis."""

    def __init__(self, make_dec, hidden=64, latent_dim=32, depth=2, n_tokens=32):
        super().__init__()
        self.enc_a = _TokenEncoder(hidden, n_tokens, n_layers=max(1, depth))
        self.enc_b = _TokenEncoder(hidden, n_tokens, n_layers=max(1, depth))
        self.dec_a = make_dec(); self.dec_b = make_dec()
        self.proj_a = nn.Linear(hidden, latent_dim)
        self.proj_b = nn.Linear(hidden, latent_dim)
        self.head = FusionHead(latent_dim, hidden=hidden, mode="mlp")

    def encode(self, A, t_A, B, t_B) -> dict:
        tok_a, tau_a = self.enc_a(A, t_A); tok_b, tau_b = self.enc_b(B, t_B)
        z_a = self.proj_a(tok_a.mean(1)); z_b = self.proj_b(tok_b.mean(1))
        return {"z": torch.cat([z_a, z_b], -1), "z_a": z_a, "z_b": z_b,
                "tok_a": tok_a, "tau_a": tau_a, "tok_b": tok_b, "tau_b": tau_b,
                "A": A, "t_A": t_A, "B": B, "t_B": t_B}

    def fuse(self, e) -> torch.Tensor:
        return self.head(e["z_a"], e["z_b"])

    def recon_loss(self, e) -> torch.Tensor:
        pa = self.dec_a(e["t_A"], e["tok_a"], e["tau_a"])
        pb = self.dec_b(e["t_B"], e["tok_b"], e["tau_b"])
        return torch.mean((pa - e["A"]) ** 2) + torch.mean((pb - e["B"]) ** 2)


def LAINRReLU(hidden=64, latent_dim=32, depth=2, n_bands=4, **_):
    return _LAINRWithDecoder(lambda: _ReLUDecoder(hidden, n_blocks=n_bands),
                             hidden=hidden, latent_dim=latent_dim, depth=depth)


def LAINRSingleSigma(hidden=64, latent_dim=32, depth=2, n_bands=4, feats_per_band=16,
                     f_max=64.0, **_):
    n_freq = n_bands * feats_per_band
    return _LAINRWithDecoder(lambda: _SingleSigmaDecoder(hidden, n_freq=n_freq, f_max=f_max),
                             hidden=hidden, latent_dim=latent_dim, depth=depth)
