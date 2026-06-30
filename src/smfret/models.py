"""smFRET arms: cheap comparator, binned-grid CNNs, and coordinate neural fields.

The contrast that tests the rubric:
  - comparator : logistic on the acceptor autocorrelation (the Bayes-cheap timescale read).
  - late_cnn / early_cnn : CNN on BINNED intensities -> grid resampling (the C4 disadvantage).
  - nf_relu  : DeepSets over RAW photon times with a ReLU coordinate-MLP embedding (control:
               spectral bias attenuates the fast structure -> should behave like the grid arms).
  - nf_fourier : DeepSets over RAW photon times with a MULTI-BAND FOURIER embedding (lossless
               ingestion of irregular arrival times; the candidate).
nf_fourier vs nf_relu isolates the Fourier basis (C2) from the "neural field" brand.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class Comparator(nn.Module):
    def __init__(self, k, hidden=32):
        super().__init__(); self.net = mlp([k, hidden, 1])

    def forward(self, b):
        return self.net(b["ac"]).squeeze(-1)


class _BinnedCNN(nn.Module):
    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, 2, 3), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, 2, 1), nn.GELU())

    def feat(self, x):
        return self.c(x).mean(-1)


class LateCNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.d = _BinnedCNN(1, hidden); self.a = _BinnedCNN(1, hidden)
        self.head = mlp([2 * hidden, hidden, 1])

    def forward(self, b):
        z = torch.cat([self.d.feat(b["bd"][:, None]), self.a.feat(b["ba"][:, None])], -1)
        return self.head(z).squeeze(-1)


class EarlyCNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.c = _BinnedCNN(2, hidden); self.head = mlp([hidden, hidden, 1])

    def forward(self, b):
        x = torch.stack([b["bd"], b["ba"]], 1)
        return self.head(self.c.feat(x)).squeeze(-1)


class SpectralField(nn.Module):
    """Neural field: LOSSLESS spectral ingestion of raw photon arrival times (no binning).

    The discriminative signal (switching timescale) is a SECOND-ORDER statistic, so we form
    the empirical periodogram of each channel and the cross-spectrum between channels at
    multi-band frequencies, directly from the irregular arrival times:
        C_ch(f) = (1/sqrt(N)) * sum_i exp(i 2pi f t_i)
        donor power |C_d|^2 ; acceptor power |C_a|^2 ; cross Re(C_d * conj(C_a))
    The cross-spectrum carries the donor/acceptor anti-correlation. This is the continuous
    analog of a spectral read-out, ingesting irregular times losslessly (the C4 advantage).
    """
    def __init__(self, freqs, hidden=64):
        super().__init__()
        self.register_buffer("w", torch.tensor(2 * np.pi * freqs, dtype=torch.float32))
        self.head = mlp([3 * len(freqs), hidden, hidden, 1])

    def _sum(self, t, m):                                   # complex sum / sqrt(N): (B,F)
        a = t[..., None] * self.w
        cos = (torch.cos(a) * m[..., None]).sum(1)
        sin = (torch.sin(a) * m[..., None]).sum(1)
        s = m.sum(1, keepdim=True).clamp(min=1).sqrt()
        return cos / s, sin / s

    def forward(self, b):
        cd, sd = self._sum(b["td"], b["md"]); ca, sa = self._sum(b["ta"], b["ma"])
        feat = torch.cat([cd * cd + sd * sd,            # donor power spectrum
                          ca * ca + sa * sa,            # acceptor power spectrum
                          cd * ca + sd * sa], -1)       # cross-spectrum (anti-correlation)
        return self.head(feat).squeeze(-1)


class _ReLUSet(nn.Module):
    """Control: naive coordinate field -- ReLU MLP over raw times, masked mean-pool. A plain
    coordinate-MLP has spectral bias and (mean-pooling) captures only first-order structure,
    so it should NOT recover the switching timescale -- isolating the spectral basis as the
    mechanism, not 'neural field' as a brand."""
    def __init__(self, d=64, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = mlp([2 * d, hidden, 1])

    def _pool(self, t, m):
        h = self.phi(t[..., None]) * m[..., None]
        return h.sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    def forward(self, b):
        return self.head(torch.cat([self._pool(b["td"], b["md"]), self._pool(b["ta"], b["ma"])], -1)).squeeze(-1)


FAMILIES = ("comparator", "late_cnn", "early_cnn", "nf_relu", "nf_fourier")


def build_model(family, cfg, freqs):
    if family == "comparator":
        return Comparator(cfg.ac_lags)
    if family == "late_cnn":
        return LateCNN()
    if family == "early_cnn":
        return EarlyCNN()
    if family == "nf_relu":
        return _ReLUSet()
    if family == "nf_fourier":
        return SpectralField(freqs)
    raise ValueError(family)
