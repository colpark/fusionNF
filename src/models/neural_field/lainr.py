"""LAINR-style neural-field fusion arm (Phase 4).

Faithful 1-D adaptation of *Locality-Aware Generalizable Implicit Neural
Representation* (Lee et al., NeurIPS 2023, arXiv:2310.05624). The original targets
images/PDE fields on 2-D/3-D grids; we keep its two defining mechanisms and apply
them to 1-D temporal signals:

1. **Transformer encoder -> latent tokens.** A strided-conv stem turns the signal
   into a short sequence of tokens, each carrying *local* information, then a couple
   of self-attention layers contextualize them. Each token i has a center time tau_i.

2. **Locality-aware INR decoder.** For a query coordinate t, a cross-attention
   *selectively aggregates* the latent tokens into a modulation vector -- but with a
   learnable Gaussian **locality bias** -|t - tau_i|^2 added to the attention logits,
   so spatially-local tokens dominate (the "locality-aware" part). The modulation
   then drives a **multi-band, coarse-to-fine** Fourier-feature INR: the modulation
   is split across frequency bands (low -> high) and progressively composed, which is
   how LAINR captures high-frequency detail (the "spectral locality" part). This is
   exactly the property we need: the field is forced to preserve the high-frequency
   matching band rather than smear it (no ReLU-MLP spectral bias).

Amortization: the encoder is feed-forward (one pass), so this is an *amortized*
field -- the cheap end the brief cares about, but a real INR architecture rather
than a plain CNN->latent strawman.

Fusion: per-modality fields produce pooled latents z_A, z_B which a shared
``FusionHead`` combines. So LAINR is a *per-modality field + separate fusion head* --
the natural mid-point between late fusion (pooled encoders) and early fusion
(token-level cross-attention).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaseFusion
from .fusion_head import FusionHead


class _TokenEncoder(nn.Module):
    """Conv stem -> ~n_tokens tokens -> self-attention. Returns (tokens, tau)."""

    def __init__(self, hidden: int, n_tokens: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2), nn.GELU(),
        )
        self.pos = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.stem(x.unsqueeze(1))                       # (B, H, L)
        h = F.adaptive_avg_pool1d(h, self.n_tokens)         # (B, H, T)
        h = h.transpose(1, 2)                               # (B, T, H)
        tau = F.adaptive_avg_pool1d(t.unsqueeze(1), self.n_tokens).transpose(1, 2)  # (B,T,1)
        h = h + self.pos(tau)
        h = self.encoder(h)                                 # (B, T, H)
        return h, tau.squeeze(-1)                           # tokens, token-center times


class _LocalityAwareDecoder(nn.Module):
    """Per-coordinate cross-attention (with locality bias) -> multi-band coarse-to-fine INR."""

    def __init__(self, hidden: int, n_bands: int = 4, feats_per_band: int = 16,
                 f_max: float = 64.0):
        super().__init__()
        self.hidden = hidden
        self.n_bands = n_bands
        self.scale = hidden ** -0.5
        # cross-attention projections (single-head, query=coord embed, key/val=tokens)
        self.q_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        # learnable locality bandwidth (softplus keeps it positive)
        self.log_locality = nn.Parameter(torch.tensor(2.0))

        # per-band Fourier frequencies, geometrically increasing across bands
        band_edges = torch.logspace(math.log10(1.0), math.log10(f_max), n_bands + 1)
        freqs = []
        for b in range(n_bands):
            freqs.append(torch.logspace(math.log10(band_edges[b].item()),
                                        math.log10(band_edges[b + 1].item()),
                                        feats_per_band))
        self.register_buffer("band_freqs", torch.stack(freqs))   # (n_bands, feats_per_band)
        self.band_in = nn.ModuleList(
            nn.Linear(2 * feats_per_band, hidden) for _ in range(n_bands))
        self.band_film = nn.ModuleList(
            nn.Linear(hidden, 2 * hidden) for _ in range(n_bands))
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, 1)

    def modulation(self, coords: torch.Tensor, tokens: torch.Tensor,
                   tau: torch.Tensor) -> torch.Tensor:
        """Locality-aware cross-attention: (B,M) coords, (B,T,H) tokens -> (B,M,H)."""
        q = self.q_proj(coords.unsqueeze(-1))               # (B, M, H)
        k = self.k_proj(tokens)                             # (B, T, H)
        v = self.v_proj(tokens)                             # (B, T, H)
        logits = torch.bmm(q, k.transpose(1, 2)) * self.scale          # (B, M, T)
        # locality bias: penalize tokens far (in time) from the query coordinate
        dist2 = (coords.unsqueeze(-1) - tau.unsqueeze(1)) ** 2         # (B, M, T)
        logits = logits - F.softplus(self.log_locality) * dist2
        attn = torch.softmax(logits, dim=-1)
        return torch.bmm(attn, v)                            # (B, M, H)

    def forward(self, coords: torch.Tensor, tokens: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        coords2 = coords.squeeze(-1) if coords.dim() == 3 else coords  # (B, M)
        m = self.modulation(coords2, tokens, tau)           # (B, M, H)
        h = torch.zeros(coords2.shape[0], coords2.shape[1], self.hidden,
                        device=coords2.device, dtype=tokens.dtype)
        # coarse -> fine: accumulate band contributions, each FiLM-modulated by m
        for b in range(self.n_bands):
            ang = 2.0 * math.pi * coords2.unsqueeze(-1) * self.band_freqs[b]  # (B,M,Fb)
            feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)        # (B,M,2Fb)
            hb = self.band_in[b](feat)                                        # (B,M,H)
            scale, shift = self.band_film[b](m).chunk(2, dim=-1)
            h = h + self.act((1.0 + scale) * hb + shift)
        return self.out(h).squeeze(-1)                       # (B, M)


class LAINRField(BaseFusion):
    """LAINR per-modality field + shared fusion head."""

    def __init__(self, hidden: int = 64, latent_dim: int = 32, depth: int = 2,
                 n_tokens: int = 32, n_bands: int = 4, f_max: float = 64.0,
                 fusion_mode: str = "mlp"):
        super().__init__()
        self.enc_a = _TokenEncoder(hidden, n_tokens, n_layers=max(1, depth))
        self.enc_b = _TokenEncoder(hidden, n_tokens, n_layers=max(1, depth))
        self.dec_a = _LocalityAwareDecoder(hidden, n_bands=n_bands, f_max=f_max)
        self.dec_b = _LocalityAwareDecoder(hidden, n_bands=n_bands, f_max=f_max)
        self.proj_a = nn.Linear(hidden, latent_dim)
        self.proj_b = nn.Linear(hidden, latent_dim)
        self.head = FusionHead(latent_dim, hidden=hidden, mode=fusion_mode)

    def encode(self, A, t_A, B, t_B) -> dict:
        tok_a, tau_a = self.enc_a(A, t_A)
        tok_b, tau_b = self.enc_b(B, t_B)
        z_a = self.proj_a(tok_a.mean(dim=1))                # pooled per-modality latent
        z_b = self.proj_b(tok_b.mean(dim=1))
        return {"z": torch.cat([z_a, z_b], dim=-1), "z_a": z_a, "z_b": z_b,
                "tok_a": tok_a, "tau_a": tau_a, "tok_b": tok_b, "tau_b": tau_b,
                "A": A, "t_A": t_A, "B": B, "t_B": t_B}

    def fuse(self, encoded: dict) -> torch.Tensor:
        return self.head(encoded["z_a"], encoded["z_b"])

    def recon_loss(self, encoded: dict) -> torch.Tensor:
        pred_a = self.dec_a(encoded["t_A"], encoded["tok_a"], encoded["tau_a"])
        pred_b = self.dec_b(encoded["t_B"], encoded["tok_b"], encoded["tau_b"])
        return torch.mean((pred_a - encoded["A"]) ** 2) + \
            torch.mean((pred_b - encoded["B"]) ** 2)

    @torch.no_grad()
    def reconstruct(self, A, t_A, B, t_B):
        enc = self.encode(A, t_A, B, t_B)
        return (self.dec_a(t_A, enc["tok_a"], enc["tau_a"]),
                self.dec_b(t_B, enc["tok_b"], enc["tau_b"]))
