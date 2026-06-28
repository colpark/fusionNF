"""Classical coherence baseline -- the model-structure-aware anchor (Phase 3).

Unlike the oracle, this method never sees the ground-truth trajectories. It only
uses the raw observations ``A`` (an FM chirp) and ``B`` (an AM tone) together with
config constants that describe the *known* observation model (the AM carrier and
the trajectory's [f_min, f_max] range). This mirrors a classical signal-processing
engineer who knows the modulation scheme but not the latent message.

Pipeline
--------
1. ``f_A_hat(t)`` from A via FM demodulation: instantaneous frequency of the
   analytic signal -- ``Hilbert(A)`` -> unwrap phase -> ``dphi/dt / (2*pi)`` --
   then a moving-average smooth.
2. ``f_B_hat(t)`` from B via AM demodulation: bandpass around the carrier,
   envelope ``|Hilbert(.)|``, smooth, then invert the generator's linear map
   ``g`` to recover an estimate ``f_mid + f_half * (env - 1) / m``.
3. Resample both estimates onto a common uniform grid, z-score each, and take the
   Pearson correlation. High correlation => the two modalities carry the same
   trajectory (label 1).

A single threshold (fit on train) turns the correlation into a decision. The
feature is affine-invariant, so the exact gain of the demodulators is irrelevant;
only the *shape* agreement between the recovered trajectories matters.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert

from ..config import DataConfig
from ..data.dataset import SignalPairDataset
from ..data.generator import Sample
from .threshold import accuracy_at, fit_threshold, pearson

# Common analysis grid and edge-trim fraction for all pairs.
_COMMON_POINTS = 512
_EDGE_TRIM = 0.05          # drop 5% of samples at each end (Hilbert/filter edges)
_SMOOTH_SEC = 0.10         # moving-average window, in seconds


def _smooth(x: np.ndarray, rate: float, smooth_sec: float = _SMOOTH_SEC) -> np.ndarray:
    """Centred moving-average smoother with reflective edges."""
    win = max(3, int(round(smooth_sec * rate)))
    return uniform_filter1d(x, size=win, mode="reflect")


def _instantaneous_frequency(A: np.ndarray, t: np.ndarray, rate: float) -> np.ndarray:
    """FM demodulation: instantaneous frequency (Hz) of A on its own grid."""
    analytic = hilbert(A)
    phase = np.unwrap(np.angle(analytic))
    # dphi/dt via central differences on the (uniform) t_A grid; /2pi -> Hz.
    inst = np.gradient(phase, t) / (2.0 * np.pi)
    return _smooth(inst, rate)


def _am_envelope(B: np.ndarray, rate: float, carrier: float) -> np.ndarray:
    """AM demodulation: bandpass around carrier, then Hilbert envelope, smoothed."""
    nyq = 0.5 * rate
    # Passband around the carrier, kept strictly inside (0, nyq) with a margin so
    # it is valid for both the matched (128 Hz) and asymmetric (96 Hz) rates.
    half_bw = min(15.0, carrier - 2.0, nyq - carrier - 2.0)
    if half_bw > 1.0:
        lo = (carrier - half_bw) / nyq
        hi = (carrier + half_bw) / nyq
        b, a = butter(4, [lo, hi], btype="band")
        # padlen guard for short signals.
        padlen = 3 * max(len(a), len(b))
        if B.shape[0] > padlen:
            B = filtfilt(b, a, B)
    env = np.abs(hilbert(B))
    return _smooth(env, rate)


def coherence_feature(sample: Sample, data_cfg: DataConfig) -> float:
    """Classical correspondence feature for a single pair (no ground truth).

    Returns the Pearson correlation between the FM-demodulated trajectory of A and
    the AM-demodulated trajectory of B, after resampling both to a common grid and
    z-scoring. Higher => stronger evidence the pair shares one trajectory.
    """
    t_A = np.asarray(sample.t_A, dtype=np.float64)
    A = np.asarray(sample.A, dtype=np.float64)
    t_B = np.asarray(sample.t_B, dtype=np.float64)
    B = np.asarray(sample.B, dtype=np.float64)

    rate_a = float(data_cfg.modality_a.rate)
    rate_b = float(data_cfg.modality_b.rate)
    carrier = float(data_cfg.modality_b.carrier)
    f_min = float(data_cfg.trajectory.f_min)
    f_max = float(data_cfg.trajectory.f_max)
    f_mid = 0.5 * (f_min + f_max)
    f_half = 0.5 * (f_max - f_min)
    m = float(data_cfg.modality_b.am_depth)

    f_A_hat = _instantaneous_frequency(A, t_A, rate_a)
    env = _am_envelope(B, rate_b, carrier)
    # Invert the generator's linear g-map (affine; correlation-invariant but kept
    # so the estimate is a calibrated frequency in Hz).
    f_B_hat = f_mid + f_half * (env - 1.0) / m

    # Resample both estimates to a common uniform grid over their overlapping span.
    t0 = max(t_A[0], t_B[0])
    t1 = min(t_A[-1], t_B[-1])
    if not (t1 > t0):
        return 0.0
    grid = np.linspace(t0, t1, _COMMON_POINTS)
    a_on = np.interp(grid, t_A, f_A_hat)
    b_on = np.interp(grid, t_B, f_B_hat)

    # Trim edges where Hilbert/filtering transients dominate.
    trim = int(_EDGE_TRIM * _COMMON_POINTS)
    if trim > 0:
        a_on = a_on[trim:-trim]
        b_on = b_on[trim:-trim]

    # z-score each, then correlate (z-scoring is redundant for Pearson but makes
    # the "compare normalized trajectories" step explicit).
    def _z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    return pearson(_z(a_on), _z(b_on))


def _features_and_labels(data_cfg: DataConfig, base_seed: int, n: int, split: str):
    """Compute the coherence feature and label for ``n`` samples of a split."""
    ds = SignalPairDataset(data_cfg, base_seed, split, n)
    feats = np.empty(n, dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = ds.raw(i)
        feats[i] = coherence_feature(s, data_cfg)
        labels[i] = int(s.label)
    return feats, labels


def coherence_accuracy(data_cfg: DataConfig, base_seed: int, n: int,
                       split: str = "test") -> dict:
    """Fit the coherence threshold on train and evaluate on ``split``.

    Same fit/eval protocol as the oracle. Returns ``{"accuracy", "threshold",
    "n"}``. Expected above chance on easy (>0.65); weaker on hard (reported as-is).
    """
    train_feats, train_labels = _features_and_labels(data_cfg, base_seed, n, "train")
    threshold = fit_threshold(train_feats, train_labels)

    if split == "train":
        eval_feats, eval_labels = train_feats, train_labels
    else:
        eval_feats, eval_labels = _features_and_labels(data_cfg, base_seed, n, split)

    accuracy = accuracy_at(eval_feats, eval_labels, threshold)
    return {"accuracy": accuracy, "threshold": threshold, "n": n}
