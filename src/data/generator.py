"""Phase 1 data generator (implements brief section 2.1 exactly).

Two modalities carry a single slowly-varying instantaneous-frequency trajectory
f(t). A pair is labeled 1 iff A and B carry the *same* f(t).

  Modality A (chirp / FM):  A(t) = c * sin(2*pi * integral_0^t f_A(s) ds) + n_A(t)
  Modality B (AM):          B(t) = (1 + m*g(f_B(t))) * sin(2*pi*f_carrier*t) + n_B(t)

n_A, n_B are multiscale 1/f^beta backgrounds summed over S octaves; their level is
set so the clean-signal-to-background SNR matches the configured snr_db.

Crucial design property (gives criterion C2 for free): each modality's own
trajectory is *always* a fresh prior draw. For label 1 we copy A's trajectory into
B; for label 0 B draws its own. So the marginal distribution of each modality is
identical across classes -- only the cross-modal coupling differs.

Determinism: generate(config, seed) is a pure function of (config, seed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

from ..config import DataConfig, TrajectoryConfig


@dataclass
class Sample:
    A: np.ndarray        # (N_A,) observed modality A
    t_A: np.ndarray      # (N_A,) sample times for A (uniform grid)
    B: np.ndarray        # (N_B,) observed modality B
    t_B: np.ndarray      # (N_B,) sample times for B (may be jittered/irregular)
    f_A: np.ndarray      # (N_A,) ground-truth trajectory underlying A
    f_B: np.ndarray      # (N_B,) ground-truth trajectory underlying B
    label: int           # 1 if f_A == f_B (correspondence), else 0
    # Optional clean/noise decomposition, for the validation suite (C3). Not used
    # by models -- only the validator requests them.
    clean_A: Optional[np.ndarray] = None
    noise_A: Optional[np.ndarray] = None
    clean_B: Optional[np.ndarray] = None
    noise_B: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Trajectory prior: f(t) = clip(f0 + sum_k a_k sin(2*pi*nu_k*t + phi_k), fmin, fmax)
# --------------------------------------------------------------------------- #
@dataclass
class TrajParams:
    f0: float
    nu: np.ndarray
    a: np.ndarray
    phi: np.ndarray
    f_min: float
    f_max: float


def sample_traj_params(rng: np.random.Generator, tc: TrajectoryConfig) -> TrajParams:
    K = tc.n_components
    return TrajParams(
        f0=tc.f0,
        nu=rng.uniform(tc.nu_min, tc.nu_max, K),
        a=rng.uniform(tc.amp_min, tc.amp_max, K),
        phi=rng.uniform(0.0, 2.0 * np.pi, K),
        f_min=tc.f_min,
        f_max=tc.f_max,
    )


def eval_traj(p: TrajParams, t: np.ndarray) -> np.ndarray:
    f = np.full_like(t, p.f0, dtype=np.float64)
    for k in range(len(p.nu)):
        f = f + p.a[k] * np.sin(2.0 * np.pi * p.nu[k] * t + p.phi[k])
    return np.clip(f, p.f_min, p.f_max)


# --------------------------------------------------------------------------- #
# Multiscale 1/f^beta background over S octaves (criterion C3).
# --------------------------------------------------------------------------- #
def octave_background(rng: np.random.Generator, n: int, rate: float,
                      n_octaves: int, beta: float) -> np.ndarray:
    """Sum of S top octave bands of white noise, weighted ~ center_freq^(-beta/2).

    Power per band scales as 1/f^beta, so the result is genuinely multiscale with
    `n_octaves` resolvable bands. Returned with unit variance.
    """
    white = rng.standard_normal(n)
    X = rfft(white)
    freqs = rfftfreq(n, d=1.0 / rate)
    nyq = rate / 2.0
    out = np.zeros_like(X)
    for s in range(n_octaves):
        f_hi = nyq / (2.0 ** s)
        f_lo = nyq / (2.0 ** (s + 1))
        band = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(band):
            continue
        center = np.sqrt(f_lo * f_hi) if f_lo > 0 else f_hi / 2.0
        weight = center ** (-beta / 2.0)
        out[band] += X[band] * weight
    sig = irfft(out, n=n)
    std = sig.std()
    if std > 0:
        sig = sig / std
    return sig.astype(np.float64)


def _apply_snr(clean: np.ndarray, noise_raw: np.ndarray, snr_db: float):
    """Scale unit-variance noise so 10*log10(var(clean)/var(noise)) == snr_db."""
    p_clean = clean.var()
    p_noise = noise_raw.var()
    if p_noise <= 0 or p_clean <= 0:
        return clean.copy(), np.zeros_like(noise_raw)
    p_target = p_clean / (10.0 ** (snr_db / 10.0))
    alpha = np.sqrt(p_target / p_noise)
    noise = alpha * noise_raw
    return clean + noise, noise


# --------------------------------------------------------------------------- #
# Modality observation models.
# --------------------------------------------------------------------------- #
def _uniform_grid(duration: float, rate: float) -> np.ndarray:
    n = int(np.floor(duration * rate))
    return np.arange(n, dtype=np.float64) / rate


def _jittered_grid(t: np.ndarray, rate: float, jitter: float,
                   rng: np.random.Generator, duration: float) -> np.ndarray:
    if jitter <= 0:
        return t
    dt = 1.0 / rate
    tj = t + jitter * dt * rng.standard_normal(t.shape[0])
    tj = np.clip(tj, 0.0, duration)
    tj = np.maximum.accumulate(tj)  # keep monotonic non-decreasing
    return tj


def _make_A(p: TrajParams, t_A: np.ndarray, dc: DataConfig,
            rng: np.random.Generator):
    f_A = eval_traj(p, t_A)
    # instantaneous phase = 2*pi * cumulative integral of f over time
    phase = np.zeros_like(t_A)
    if t_A.shape[0] > 1:
        dt = np.diff(t_A)
        incr = 0.5 * (f_A[1:] + f_A[:-1]) * dt  # trapezoid
        phase[1:] = 2.0 * np.pi * np.cumsum(incr)
    clean = dc.modality_a.signal_amp * np.sin(phase)
    noise_raw = octave_background(rng, t_A.shape[0], dc.modality_a.rate,
                                  dc.modality_a.n_octaves, dc.modality_a.noise_beta)
    obs, noise = _apply_snr(clean, noise_raw, dc.snr_db)
    return obs, clean, noise, f_A


def _make_B(p: TrajParams, t_B: np.ndarray, dc: DataConfig,
            rng: np.random.Generator):
    f_B = eval_traj(p, t_B)
    f_mid = 0.5 * (p.f_min + p.f_max)
    f_half = 0.5 * (p.f_max - p.f_min)
    g = (f_B - f_mid) / f_half          # ~[-1, 1]
    env = 1.0 + dc.modality_b.am_depth * g   # >=0 for am_depth<=1
    clean = env * np.sin(2.0 * np.pi * dc.modality_b.carrier * t_B)
    noise_raw = octave_background(rng, t_B.shape[0], dc.modality_b.rate,
                                  dc.modality_b.n_octaves, dc.modality_b.noise_beta)
    obs, noise = _apply_snr(clean, noise_raw, dc.snr_db)
    return obs, clean, noise, f_B


# --------------------------------------------------------------------------- #
# Top-level generate().
# --------------------------------------------------------------------------- #
def generate(config: DataConfig, seed: int, return_components: bool = False) -> Sample:
    rng = np.random.default_rng(seed)

    label = int(rng.random() < config.p_positive)

    params_A = sample_traj_params(rng, config.trajectory)
    if label == 1:
        params_B = params_A                       # shared trajectory
    else:
        params_B = sample_traj_params(rng, config.trajectory)  # independent draw

    t_A = _uniform_grid(config.duration, config.modality_a.rate)
    t_B = _uniform_grid(config.duration, config.modality_b.rate)
    t_B = _jittered_grid(t_B, config.modality_b.rate, config.jitter, rng, config.duration)

    A, clean_A, noise_A, f_A = _make_A(params_A, t_A, config, rng)
    B, clean_B, noise_B, f_B = _make_B(params_B, t_B, config, rng)

    s = Sample(
        A=A.astype(np.float32), t_A=t_A.astype(np.float32),
        B=B.astype(np.float32), t_B=t_B.astype(np.float32),
        f_A=f_A.astype(np.float32), f_B=f_B.astype(np.float32),
        label=label,
    )
    if return_components:
        s.clean_A, s.noise_A = clean_A.astype(np.float32), noise_A.astype(np.float32)
        s.clean_B, s.noise_B = clean_B.astype(np.float32), noise_B.astype(np.float32)
    return s
