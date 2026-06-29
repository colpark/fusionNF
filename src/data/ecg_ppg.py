"""ECG + PPG cardiac generator -- a cheap, conventional real-world instantiation of
the fusion structure (drop-in for the chirp/AM generator; same Sample interface).

Shared latent f(t) = instantaneous HEART RATE HR(t) (Hz). Both modalities are driven
by the same cardiac phase phi(t) = 2*pi * integral HR, but through different
morphologies:
  - Modality A = ECG: sharp P-QRS-T waveform of phase -> high-frequency, "FM/phase"-like.
  - Modality B = PPG: smooth systolic/diastolic pulse, with a respiratory amplitude
    modulation (1 + m*sin(2*pi*f_resp*t)) -> low-frequency, "AM"-like.
Different sample rates (ECG > PPG), optional jitter, and a multiscale 1/f^beta
background whose level is set by snr_db (PPG under motion is the buried regime).

Label 1: ECG and PPG share HR(t); label 0: independent HR draws. Each modality's own
HR is always a fresh prior draw, so per-modality marginals are identical across
classes (criterion C2 holds by construction).
"""
from __future__ import annotations

import numpy as np

from .generator import (Sample, sample_traj_params, eval_traj, octave_background,
                        _apply_snr, _uniform_grid, _jittered_grid)
from ..config import DataConfig


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _ecg_wave(theta):
    """Compact P-QRS-T morphology as a function of cardiac phase theta in [0,2pi)."""
    p = 0.12 * np.exp(-(_wrap(theta + 1.2) / 0.18) ** 2)     # P wave
    q = -0.15 * np.exp(-(_wrap(theta + 0.25) / 0.05) ** 2)   # Q
    r = 1.00 * np.exp(-(_wrap(theta - 0.0) / 0.05) ** 2)     # R spike (sharp)
    s = -0.20 * np.exp(-(_wrap(theta - 0.25) / 0.05) ** 2)   # S
    t = 0.30 * np.exp(-(_wrap(theta - 1.6) / 0.18) ** 2)     # T wave
    return p + q + r + s + t


def _ppg_wave(theta):
    """Smooth systolic + diastolic pulse as a function of cardiac phase."""
    systolic = np.exp(-(_wrap(theta - 0.5) / 0.30) ** 2)
    diastolic = 0.4 * np.exp(-(_wrap(theta - 1.7) / 0.45) ** 2)
    return systolic + diastolic


def _phase(hr, t):
    """Cardiac phase phi(t) = 2*pi * cumulative integral of HR over time."""
    phase = np.zeros_like(t)
    if t.shape[0] > 1:
        dt = np.diff(t)
        incr = 0.5 * (hr[1:] + hr[:-1]) * dt
        phase[1:] = 2.0 * np.pi * np.cumsum(incr)
    return phase


def generate_ecg_ppg(config: DataConfig, seed: int, return_components: bool = False) -> Sample:
    rng = np.random.default_rng(seed)
    label = int(rng.random() < config.p_positive)

    hr_params_A = sample_traj_params(rng, config.trajectory)          # HR(t) for ECG
    hr_params_B = hr_params_A if label == 1 else sample_traj_params(rng, config.trajectory)

    t_A = _uniform_grid(config.duration, config.modality_a.rate)
    t_B = _uniform_grid(config.duration, config.modality_b.rate)
    t_B = _jittered_grid(t_B, config.modality_b.rate, config.jitter, rng, config.duration)

    hr_A = eval_traj(hr_params_A, t_A)                                # Hz (beats/sec)
    hr_B = eval_traj(hr_params_B, t_B)

    # ECG: sharp morphology driven by cardiac phase
    clean_A = _ecg_wave(_phase(hr_A, t_A) % (2 * np.pi))
    noise_A = octave_background(rng, t_A.shape[0], config.modality_a.rate,
                               config.modality_a.n_octaves, config.modality_a.noise_beta)
    A, n_A = _apply_snr(clean_A, noise_A, config.snr_db)

    # PPG: smooth pulse with respiratory amplitude modulation
    f_resp = config.modality_b.carrier                               # reuse carrier as resp freq (Hz)
    resp = 1.0 + config.modality_b.am_depth * 0.3 * np.sin(2 * np.pi * f_resp * t_B)
    clean_B = resp * _ppg_wave(_phase(hr_B, t_B) % (2 * np.pi))
    noise_B = octave_background(rng, t_B.shape[0], config.modality_b.rate,
                               config.modality_b.n_octaves, config.modality_b.noise_beta)
    B, n_B = _apply_snr(clean_B, noise_B, config.snr_db)

    s = Sample(
        A=A.astype(np.float32), t_A=t_A.astype(np.float32),
        B=B.astype(np.float32), t_B=t_B.astype(np.float32),
        f_A=hr_A.astype(np.float32), f_B=hr_B.astype(np.float32),
        label=label,
    )
    if return_components:
        s.clean_A, s.noise_A = clean_A.astype(np.float32), n_A.astype(np.float32)
        s.clean_B, s.noise_B = clean_B.astype(np.float32), n_B.astype(np.float32)
    return s
