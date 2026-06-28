"""Early-fusion family (Phase 4) -- the accuracy ceiling and cost reference.

Both modalities are tokenized (a strided Conv1d patch embedding), tagged with a
coordinate-based positional encoding and a modality-type embedding, then a small
Transformer performs *cross*-attention over the concatenated A+B token set so the
two modalities interact from the very first layer -- hence "early" fusion. A
learned CLS token summarizes the joint sequence for the logit.

Cost note: the fusion stage is the Transformer over the combined sequence, whose
cost scales with the total token count (and quadratically in self-attention), so
``fuse``-FLOPs grow with sequence length -- the intended expensive cross-modal
reference against which the cheaper families are measured.

Two-stage split:
* ``encode`` does per-modality tokenization + embedding (convs, positional and
  type embeddings). ``z`` is the mean of the embedded tokens (a pooled summary for
  the frequency probe).
* ``fuse`` runs the cross-attention Transformer + CLS head over the combined
  tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseFusion, mlp


def _patch_size(n: int, target_tokens: int) -> int:
    """Patch/stride giving ~``target_tokens`` tokens for a length-``n`` signal."""
    return max(1, round(n / max(1, target_tokens)))


class _Tokenizer(nn.Module):
    """Conv1d patch embedding + coordinate positional encoding for one modality."""

    def __init__(self, d_model: int, patch: int):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
        # Positional encoding from the (already [0,1]) coordinate of each patch.
        self.pos = mlp([1, d_model, d_model])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x.unsqueeze(1)).transpose(1, 2)      # (B, T, d)
        T = tok.shape[1]
        # Downsample coordinates to one value per token, robust to off-by-one.
        coord = F.adaptive_avg_pool1d(t.unsqueeze(1), T).transpose(1, 2)  # (B, T, 1)
        return tok + self.pos(coord)                          # (B, T, d)


class EarlyFusion(BaseFusion):
    """Tokenize -> cross-attention Transformer over combined tokens -> logit."""

    def __init__(self, n_a: int, n_b: int, hidden: int = 64, depth: int = 2,
                 n_heads: int = 4, target_tokens: int = 32, latent_dim: int = 32):
        super().__init__()
        d = hidden
        self.d_model = d
        self.tok_a = _Tokenizer(d, _patch_size(n_a, target_tokens))
        self.tok_b = _Tokenizer(d, _patch_size(n_b, target_tokens))
        # Modality-type embeddings distinguish A-tokens from B-tokens.
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=max(1, depth), enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)
        # Project the pooled summary to latent_dim so ``z`` width is comparable
        # across families (used only by the probe, not by fuse).
        self.z_proj = nn.Linear(d, latent_dim)

    def encode(self, A, t_A, B, t_B) -> dict:
        tok_a = self.tok_a(A, t_A) + self.type_emb[0]
        tok_b = self.tok_b(B, t_B) + self.type_emb[1]
        combined = torch.cat([tok_a, tok_b], dim=1)           # (B, T_a+T_b, d)
        z = self.z_proj(combined.mean(dim=1))                 # (B, latent_dim)
        return {"z": z, "tokens": combined}

    def fuse(self, encoded: dict) -> torch.Tensor:
        tokens = encoded["tokens"]
        B = tokens.shape[0]
        cls = self.cls.expand(B, -1, -1)                      # (B, 1, d)
        seq = torch.cat([cls, tokens], dim=1)                 # (B, 1+T, d)
        out = self.transformer(seq)
        return self.head(out[:, 0, :]).squeeze(-1)            # CLS -> logit
