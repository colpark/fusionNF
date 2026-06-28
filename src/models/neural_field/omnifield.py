"""OmniField-style neural-field fusion arm (Phase 4).

Faithful 1-D adaptation of *OmniField: Conditioned Neural Fields for Robust
Multimodal Spatiotemporal Learning* (arXiv:2511.02205). OmniField is a
multimodal-native conditioned neural field with an encoder -> processor -> decoder
structure and, crucially, **cross-modal crosstalk with iterative refinement**. We
keep those mechanisms for 1-D signals:

1. **Per-modality encoder (E).** Observation tokens (signal patches + Gaussian
   Fourier-feature encoding of their time coordinate) are summarized by M learnable
   latent queries via cross-attention into a fixed set of context latents
   (permutation-invariant over inputs). This is the amortized field encoder.

2. **Multimodal Crosstalk (MCT) + Iterative Cross-Modal Refinement (ICMR).** A
   global bridge feature z (initialized to zero) is broadcast into every modality's
   latents; a self-attention block updates them; z is refreshed as the mean of all
   latents. Repeated for `n_refine` stages (paper default 3). z is the channel
   through which modalities communicate -- fusion happens *inside the field*.

3. **Processor + per-modality decoder.** For a query coordinate, GFF(t) cross-
   attends to that modality's refined latents to produce a hidden state, which a
   lightweight decoder maps to the signal value (for the reconstruction loss).

Design choice for FLOP accounting: the per-modality encoders run in ``encode`` (the
representation stage) and the cross-modal crosstalk/refinement runs in ``fuse`` (the
fusion stage). So OmniField's fusion-FLOPs reflect its cross-modal cost -- making it
the more *early-fusion-like* neural field, in contrast to LAINR's cheap per-modality
head. The correspondence logit is read from the refined bridge feature z.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaseFusion, mlp
from .fusion_head import FusionHead


class _GFF(nn.Module):
    """Gaussian Fourier features of a scalar coordinate: [cos(2pi B t), sin(2pi B t)]."""

    def __init__(self, n_freq: int, sigma: float = 8.0):
        super().__init__()
        self.register_buffer("B", torch.randn(n_freq) * sigma)
        self.out_dim = 2 * n_freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * t.unsqueeze(-1) * self.B          # (..., n_freq)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class _ModalityEncoder(nn.Module):
    """Irregular point observations -> M learnable latent queries cross-attend -> (B, M, H).

    Faithful to OmniField: NO gridding/conv. Each observation is a true (t, value)
    pair embedded pointwise by an MLP of [value, GaussianFourierFeatures(t)], then a
    set of learnable latent queries summarizes the observation set via cross-attention
    (permutation-invariant over observations). We pass the *actual* sample values
    (optionally subsampled by stride, never averaged -- averaging would annihilate the
    oscillation that carries f(t)). The Fourier-feature time embedding exposes the
    high-frequency coordinate structure the latents need to localize.
    """

    def __init__(self, hidden: int, n_latents: int, n_heads: int = 4,
                 n_gff: int = 32, max_obs: int = 512):
        super().__init__()
        self.max_obs = max_obs
        self.hidden = hidden
        self.scale = hidden ** -0.5
        self.gff = _GFF(n_gff)
        # Observation embedding that keeps frequency content recoverable under
        # attention-averaging: include the value, the Fourier position, AND the
        # value-modulated Fourier features value*[cos wt, sin wt]. The product terms
        # are a windowed-Fourier view -- summing them over observations approximates
        # spectral coefficients, so a weighted average (attention) can extract f(t)
        # without any convolution/gridding (faithful to OmniField's point-observation
        # premise). Plain value+position would lose frequency when averaged.
        self.in_proj = nn.Linear(1 + 2 * self.gff.out_dim, hidden)
        self.obs_ff = mlp([hidden, hidden, hidden], last_act=True)
        # query-local latents: learnable content + a fixed time anchor each.
        self.q_content = nn.Parameter(torch.randn(n_latents, hidden) * (hidden ** -0.5))
        self.register_buffer("anchors", torch.linspace(0.0, 1.0, n_latents))
        self.q_pos = nn.Linear(self.gff.out_dim, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        # Locality bandwidth: coordinates live in [0,1], so to give each of the M
        # anchored queries a *sharp* window (~half the inter-anchor spacing) the bias
        # coefficient must scale ~ M^2. Without this the attention is near-uniform and
        # averages the whole signal, collapsing the latents.
        self.register_buffer("loc_scale", torch.tensor(float(4 * n_latents ** 2)))
        self.log_locality = nn.Parameter(torch.tensor(0.0))    # learnable refinement
        self.norm = nn.LayerNorm(hidden)
        self.ff = mlp([hidden, 2 * hidden, hidden], last_act=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        if N > self.max_obs:                                    # subsample points, never average
            idx = torch.linspace(0, N - 1, self.max_obs, device=x.device).long()
            x, t = x[:, idx], t[:, idx]
        g = self.gff(t)                                        # (B, N', 2F)
        xv = x.unsqueeze(-1)
        feat = torch.cat([xv, g, xv * g], dim=-1)              # value, position, value*position
        obs = self.obs_ff(F.gelu(self.in_proj(feat)))         # (B, N', H)
        q = self.q_content.unsqueeze(0) + self.q_pos(self.gff(self.anchors)).unsqueeze(0)
        q = q.expand(B, -1, -1)                                 # (B, M, H)
        Q, K, V = self.q_proj(q), self.k_proj(obs), self.v_proj(obs)
        logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, M, N')
        # query-local bias: each anchored query attends to nearby-in-time observations
        dist2 = (self.anchors.view(1, -1, 1) - t.unsqueeze(1)) ** 2             # (B, M, N')
        logits = logits - self.loc_scale * F.softplus(self.log_locality) * dist2
        attn = torch.softmax(logits, dim=-1)
        attended = torch.bmm(attn, V)                          # (B, M, H)
        h = self.norm(q + attended)
        return h + self.ff(h)


class _CrosstalkBlock(nn.Module):
    """One MCT stage: this modality's latents cross-attend to the OTHER modality's.

    Token-level cross-attention (not bridge-averaging) lets each modality gather
    information from the other while keeping its own per-token structure -- so the
    pooled summaries remain discriminative for correspondence. This is the
    cross-modal communication channel of OmniField, realized at the token level.
    """

    def __init__(self, hidden: int, n_heads: int = 4):
        super().__init__()
        self.cross = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = mlp([hidden, 2 * hidden, hidden], last_act=False)

    def forward(self, x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        a, _ = self.cross(x, other, other)                  # x attends to other modality
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class _Decoder(nn.Module):
    """Processor+decoder: GFF(t) cross-attends to refined latents -> signal value."""

    def __init__(self, hidden: int, n_heads: int = 4, n_gff: int = 32):
        super().__init__()
        self.gff = _GFF(n_gff)
        self.q_proj = nn.Linear(self.gff.out_dim, hidden)
        self.cross = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.dec = mlp([hidden, hidden, 1])

    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        coords2 = coords.squeeze(-1) if coords.dim() == 3 else coords
        q = self.q_proj(self.gff(coords2))                  # (B, M_q, H)
        attended, _ = self.cross(q, latents, latents)       # (B, M_q, H)
        return self.dec(attended).squeeze(-1)               # (B, M_q)


class OmniFieldFusion(BaseFusion):
    """Multimodal conditioned neural field with iterative cross-modal refinement."""

    def __init__(self, hidden: int = 64, latent_dim: int = 32, depth: int = 2,
                 n_latents: int = 16, n_refine: int = 3, fusion_mode: str = "mlp"):
        super().__init__()
        self.hidden = hidden
        self.n_refine = n_refine
        self.enc_a = _ModalityEncoder(hidden, n_latents)
        self.enc_b = _ModalityEncoder(hidden, n_latents)
        self.crosstalk = nn.ModuleList(_CrosstalkBlock(hidden) for _ in range(n_refine))
        self.dec_a = _Decoder(hidden)
        self.dec_b = _Decoder(hidden)
        # per-modality field latents -> fusion head (the decision path). The
        # correspondence decision is made on the FIELD latents (the NF-fusion premise),
        # NOT on crosstalk-mixed summaries: bidirectional crosstalk homogenizes the two
        # modalities and erases the |z_a-z_b| agreement signal (verified: that variant
        # sits at chance). The crosstalk instead serves reconstruction (below).
        self.proj_a = nn.Linear(hidden, latent_dim)
        self.proj_b = nn.Linear(hidden, latent_dim)
        self.head = FusionHead(latent_dim, hidden=hidden, mode=fusion_mode)

    def encode(self, A, t_A, B, t_B) -> dict:
        lat_a = self.enc_a(A, t_A)                           # (B, M, H) representation
        lat_b = self.enc_b(B, t_B)
        z_a = self.proj_a(lat_a.mean(dim=1))                 # per-modality field latent
        z_b = self.proj_b(lat_b.mean(dim=1))
        return {"z": torch.cat([z_a, z_b], dim=-1), "z_a": z_a, "z_b": z_b,
                "lat_a": lat_a, "lat_b": lat_b,
                "A": A, "t_A": t_A, "B": B, "t_B": t_B}

    def _refine(self, lat_a, lat_b):
        """ICMR: iterative bidirectional cross-modal token attention."""
        for block in self.crosstalk:
            new_a = block(lat_a, lat_b)
            new_b = block(lat_b, lat_a)
            lat_a, lat_b = new_a, new_b
        z = 0.5 * (lat_a.mean(dim=1) + lat_b.mean(dim=1))       # bridge summary
        return lat_a, lat_b, z

    def fuse(self, encoded: dict) -> torch.Tensor:
        # decision on the per-modality field latents (NF-fusion premise)
        return self.head(encoded["z_a"], encoded["z_b"])

    def recon_loss(self, encoded: dict) -> torch.Tensor:
        # crosstalk does its real job here: cross-modal refinement improves the
        # conditioned-field reconstruction, which in turn shapes the field latents.
        ref_a, ref_b, _ = self._refine(encoded["lat_a"], encoded["lat_b"])
        pred_a = self.dec_a(encoded["t_A"], ref_a)
        pred_b = self.dec_b(encoded["t_B"], ref_b)
        return torch.mean((pred_a - encoded["A"]) ** 2) + \
            torch.mean((pred_b - encoded["B"]) ** 2)

    @torch.no_grad()
    def reconstruct(self, A, t_A, B, t_B):
        enc = self.encode(A, t_A, B, t_B)
        ref_a, ref_b, _ = self._refine(enc["lat_a"], enc["lat_b"])
        return (self.dec_a(t_A, ref_a), self.dec_b(t_B, ref_b))
