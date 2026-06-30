"""smFRET photon-by-photon simulator + feature builder.

A single molecule switches between a low-FRET and a high-FRET conformational state as a
2-state continuous-time Markov chain with relaxation time tau = 1/(k_AB+k_BA). Photons
arrive as a Poisson process at rate `photon_rate` (the BRIGHTNESS / SNR knob); each photon
is routed to the ACCEPTOR with probability E(state) and to the DONOR otherwise. The two
channels are therefore anti-correlated on the switching timescale -- the buried, high-
frequency, cross-channel factor.

TASK: binary classification of the kinetic regime -- FAST switching (short tau) vs SLOW
(long tau). The FRET levels (e_low,e_high) are FIXED, so the mean intensities are identical
across classes; the ONLY discriminative signal is the temporal/cross-correlation structure.
The Bayes-cheap comparator is the donor/acceptor (anti)correlation timescale.

This matches the four-condition rubric: C1 cheap comparator (correlation timescale),
C2 high-frequency buried signal (fast transitions vs shot noise), C3 intermediate SNR
(tunable via photon_rate), C4 irregular sampling (photon ARRIVAL TIMES; binning is lossy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SimConfig:
    duration: float = 2.0          # seconds per trace
    photon_rate: float = 300.0     # photons/s  (THE SNR KNOB)
    e_low: float = 0.3             # FRET efficiency, low state
    e_high: float = 0.7            # FRET efficiency, high state
    # Two SUB-BIN switching timescales (faster than the bin width below): the regime where
    # binning aliases the dynamics and photon-by-photon analysis is required. The classes
    # differ only in their (sub-bin) relaxation time -> the discriminative band is above the
    # binned-grid Nyquist, so grid encoders should fail and lossless raw-time reps should win.
    tau_fast: tuple = (0.002, 0.004)  # relaxation time (s), FAST class (~2-4 ms)
    tau_slow: tuple = (0.006, 0.012)  # relaxation time (s), SLOW class (~6-12 ms)
    n_bins: int = 64               # binned-grid resolution (31 ms bins @2s -> Nyquist 16 Hz)
    max_phot: int = 1024           # cap per channel (subsample if exceeded)
    n_freq: int = 32               # Fourier bands for the neural-field rep
    f_lo: float = 0.5              # Hz
    f_hi: float = 300.0            # Hz
    ac_lags: int = 16              # autocorrelation lags for the cheap comparator


def fourier_freqs(cfg: SimConfig) -> np.ndarray:
    return np.logspace(np.log10(cfg.f_lo), np.log10(cfg.f_hi), cfg.n_freq).astype(np.float32)


def simulate_trace(rng: np.random.Generator, cfg: SimConfig, tau: float):
    """Return (donor_times, acceptor_times, seg_t, seg_state) for one molecule."""
    k = 1.0 / (2.0 * tau)                       # symmetric switching rate (equal occupancy)
    seg_t = [0.0]; state = int(rng.integers(2)); seg_s = [state]; t = 0.0
    while t < cfg.duration:
        t += rng.exponential(1.0 / k); state = 1 - state
        seg_t.append(min(t, cfg.duration)); seg_s.append(state)
    seg_t = np.asarray(seg_t); seg_s = np.asarray(seg_s)
    n = int(rng.poisson(cfg.photon_rate * cfg.duration))
    if n == 0:
        return np.empty(0), np.empty(0), seg_t, seg_s
    pt = np.sort(rng.uniform(0.0, cfg.duration, n))
    idx = np.clip(np.searchsorted(seg_t, pt, side="right") - 1, 0, len(seg_s) - 1)
    e = np.where(seg_s[idx] == 1, cfg.e_high, cfg.e_low)
    is_acc = rng.uniform(0.0, 1.0, n) < e
    return pt[~is_acc], pt[is_acc], seg_t, seg_s


def _features(rng, cfg, d, a):
    nb, dur = cfg.n_bins, cfg.duration
    bd, _ = np.histogram(d, bins=nb, range=(0, dur))
    ba, _ = np.histogram(a, bins=nb, range=(0, dur))

    def pad(x):
        if len(x) > cfg.max_phot:
            x = np.sort(rng.choice(x, cfg.max_phot, replace=False))
        t = np.zeros(cfg.max_phot, np.float32); m = np.zeros(cfg.max_phot, bool)
        t[:len(x)] = x; m[:len(x)] = True
        return t, m

    td, md = pad(d); ta, ma = pad(a)
    x = ba.astype(float) - ba.mean()
    ac = np.correlate(x, x, "full")[nb - 1:]
    ac = ac / (ac[0] + 1e-9)
    lags = np.unique(np.clip(np.logspace(0, np.log10(nb - 1), cfg.ac_lags).astype(int), 1, nb - 1))
    acf = np.zeros(cfg.ac_lags, np.float32); acf[:len(lags)] = ac[lags]
    return (bd.astype(np.float32), ba.astype(np.float32), td, md, ta, ma, acf)


def make_split(cfg: SimConfig, n: int, base_seed: int) -> dict:
    """Generate `n` balanced traces (label 1=fast, 0=slow); return tensors for all arms."""
    rng = np.random.default_rng(base_seed)
    cols = {k: [] for k in ("bd", "ba", "td", "md", "ta", "ma", "ac", "y")}
    for i in range(n):
        lbl = i % 2                                       # balanced
        tau = rng.uniform(*(cfg.tau_fast if lbl == 1 else cfg.tau_slow))
        d, a, _, _ = simulate_trace(rng, cfg, tau)
        bd, ba, td, md, ta, ma, acf = _features(rng, cfg, d, a)
        for k, v in zip(("bd", "ba", "td", "md", "ta", "ma", "ac", "y"),
                        (bd, ba, td, md, ta, ma, acf, lbl)):
            cols[k].append(v)
    dt = {"md": torch.bool, "ma": torch.bool, "y": torch.float32}
    return {k: torch.tensor(np.array(v), dtype=dt.get(k, torch.float32)) for k, v in cols.items()}
