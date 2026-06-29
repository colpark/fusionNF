"""Smoke + determinism tests for the field-vs-spectral ablation arms."""
from __future__ import annotations

import torch

from src.config import load_experiment, ModelConfig
from src.models.registry import build_model, signal_lengths
from src.models.train import seeded_build

NEW_ARMS = ["late_mbff", "lainr_relu", "nf_lainr_single",
            "nf_lainr_nb1", "nf_lainr_nb2", "nf_lainr_nb8"]


def _batch(dc, n=4):
    na, nb = signal_lengths(dc)
    g = torch.Generator().manual_seed(123)
    return (torch.randn(n, na, generator=g), torch.rand(n, na, generator=g),
            torch.randn(n, nb, generator=g), torch.rand(n, nb, generator=g))


def test_arms_forward_and_recon():
    dc = load_experiment("configs/tiny.yaml").data
    A, tA, B, tB = _batch(dc)
    for fam in NEW_ARMS:
        m = build_model(fam, ModelConfig(hidden=16, depth=2, latent_dim=16), dc)
        enc = m.encode(A, tA, B, tB)
        logit = m.fuse(enc)
        assert logit.shape == (4,)
        assert enc["z"].dim() == 2 and enc["z"].shape[0] == 4   # flat z for the probe
        assert hasattr(m, "recon_loss")
        rl = m.recon_loss(enc)
        assert torch.isfinite(rl)


def test_arms_deterministic_build():
    dc = load_experiment("configs/tiny.yaml").data
    A, tA, B, tB = _batch(dc)
    for fam in NEW_ARMS:
        mc = ModelConfig(hidden=16, depth=2, latent_dim=16)
        m1 = seeded_build(fam, mc, dc, seed=0)
        m2 = seeded_build(fam, mc, dc, seed=0)
        with torch.no_grad():
            l1 = m1(A, tA, B, tB); l2 = m2(A, tA, B, tB)
        assert torch.allclose(l1, l2, atol=1e-6)               # same seed -> identical


def test_arms_backward_runs():
    dc = load_experiment("configs/tiny.yaml").data
    A, tA, B, tB = _batch(dc)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce = torch.nn.BCEWithLogitsLoss()
    for fam in NEW_ARMS:
        m = seeded_build(fam, ModelConfig(hidden=16, depth=2, latent_dim=16), dc, seed=0)
        enc = m.encode(A, tA, B, tB)
        loss = bce(m.fuse(enc), y) + 0.3 * m.recon_loss(enc)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
