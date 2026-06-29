"""Model factory (Phase 4): map a family name + configs to a ``BaseFusion``.

``build_model`` is the single construction entry point used by the training loop
and the experiment harness, so family-specific construction details (sequence
lengths for the early-fusion tokenizer, Fourier bandwidth for the field decoders)
live here rather than leaking into the training code.
"""
from __future__ import annotations

import math

from ..config import DataConfig, ModelConfig
from .common import BaseFusion
from .late_fusion import LateFusion
from .early_fusion import EarlyFusion
from .neural_field.lainr import LAINRField
from .neural_field.omnifield import OmniFieldFusion
from .ablation import LateMBFF, LAINRReLU, LAINRSingleSigma

# Two amortized neural-field arms (per user decision): LAINR (per-modality field +
# fusion head) and OmniField (cross-modal crosstalk field). The auto-decoded
# (functa/MAML) arm was dropped, so P4's amortization-spectrum prediction is
# reframed as a LAINR-vs-OmniField comparison (see reports/findings).
FAMILIES = ("late", "early", "nf_lainr", "nf_omnifield")


def signal_lengths(data_cfg: DataConfig) -> tuple[int, int]:
    """Sample counts ``(N_A, N_B)`` produced by the generator for this config."""
    n_a = int(math.floor(data_cfg.duration * data_cfg.modality_a.rate))
    n_b = int(math.floor(data_cfg.duration * data_cfg.modality_b.rate))
    return n_a, n_b


def field_f_max(data_cfg: DataConfig) -> float:
    """Top Fourier frequency (cycles over [0,1]) for the field decoders.

    Sized to comfortably cover the instantaneous-frequency trajectory band
    (``f_max`` Hz over ``duration`` s, with margin); deliberately not stretched to
    the AM carrier so the tiny field stays small -- the recon test reports which
    bands are (un)recoverable.
    """
    traj_cycles = data_cfg.trajectory.f_max * data_cfg.duration
    return float(min(128.0, max(32.0, math.ceil(traj_cycles * 1.5))))


def build_model(family: str, model_cfg: ModelConfig,
                data_cfg: DataConfig) -> BaseFusion:
    hidden = model_cfg.hidden
    depth = model_cfg.depth
    latent = model_cfg.latent_dim

    if family == "late":
        pooling = getattr(model_cfg, "pooling", "mean")
        return LateFusion(hidden=hidden, latent_dim=latent, depth=depth,
                          pooling=pooling)

    if family == "early":
        n_a, n_b = signal_lengths(data_cfg)
        return EarlyFusion(n_a=n_a, n_b=n_b, hidden=hidden, depth=depth,
                           latent_dim=latent)

    if family == "nf_lainr":
        return LAINRField(hidden=hidden, latent_dim=latent, depth=depth,
                          f_max=field_f_max(data_cfg))

    if family == "nf_omnifield":
        return OmniFieldFusion(hidden=hidden, latent_dim=latent, depth=depth)

    # ---- field-vs-spectral ablation arms (additive; selected via --families) ----
    if family == "late_mbff":            # spectral basis WITHOUT the field
        return LateMBFF(hidden=hidden, latent_dim=latent, depth=depth,
                        f_max=field_f_max(data_cfg))
    if family == "lainr_relu":           # field WITHOUT the multi-band basis
        return LAINRReLU(hidden=hidden, latent_dim=latent, depth=depth,
                         f_max=field_f_max(data_cfg))
    if family == "nf_lainr_single":      # field + single wide Gaussian FF (no multi-scale)
        return LAINRSingleSigma(hidden=hidden, latent_dim=latent, depth=depth,
                                f_max=field_f_max(data_cfg))
    if family.startswith("nf_lainr_nb"): # LAINR with n_bands in {1,2,8} (nb4 == nf_lainr)
        k = int(family.replace("nf_lainr_nb", ""))
        return LAINRField(hidden=hidden, latent_dim=latent, depth=depth,
                          n_bands=k, f_max=field_f_max(data_cfg))

    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


# Full ablation family set (for --families and parameter matching across the whole set).
ABLATION_FAMILIES = ("late", "early", "nf_lainr", "nf_omnifield",
                     "late_mbff", "lainr_relu", "nf_lainr_single",
                     "nf_lainr_nb1", "nf_lainr_nb2", "nf_lainr_nb8")
