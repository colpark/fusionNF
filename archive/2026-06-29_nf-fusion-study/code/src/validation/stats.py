"""Statistical helpers for the dataset validation suite (Phase 2).

Kept dependency-light: numpy + scipy only. All randomness is explicitly seeded.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq


def octave_band_powers(x: np.ndarray, rate: float, n_octaves: int) -> np.ndarray:
    """Power of x within each of the top `n_octaves` octave bands (high->low)."""
    X = rfft(x)
    freqs = rfftfreq(len(x), d=1.0 / rate)
    psd = (np.abs(X) ** 2)
    nyq = rate / 2.0
    powers = np.zeros(n_octaves)
    for s in range(n_octaves):
        f_hi = nyq / (2.0 ** s)
        f_lo = nyq / (2.0 ** (s + 1))
        band = (freqs >= f_lo) & (freqs < f_hi)
        powers[s] = psd[band].sum()
    return powers


def summary_stats(x: np.ndarray, rate: float, n_spec: int = 24,
                  n_hist: int = 16, n_lags: int = 16) -> np.ndarray:
    """A per-signal summary vector: log power spectrum (log-binned) + amplitude
    histogram (on standardized values) + autocorrelation at fixed lags.

    Marginal invariance (C2) means these vectors are indistinguishable between the
    two label classes for a single modality.
    """
    # --- log-binned power spectrum ---
    X = rfft(x)
    freqs = rfftfreq(len(x), d=1.0 / rate)
    psd = np.abs(X) ** 2
    fmin = max(freqs[1], 1e-3)
    edges = np.logspace(np.log10(fmin), np.log10(freqs[-1]), n_spec + 1)
    spec = np.zeros(n_spec)
    for i in range(n_spec):
        band = (freqs >= edges[i]) & (freqs < edges[i + 1])
        spec[i] = psd[band].mean() if np.any(band) else 0.0
    spec = np.log(spec + 1e-12)

    # --- amplitude histogram on standardized signal ---
    xs = (x - x.mean()) / (x.std() + 1e-8)
    hist, _ = np.histogram(xs, bins=n_hist, range=(-4, 4), density=True)

    # --- autocorrelation at fixed lags ---
    xc = x - x.mean()
    denom = np.sum(xc * xc) + 1e-12
    ac = np.array([np.sum(xc[: len(xc) - k] * xc[k:]) / denom
                   for k in range(1, n_lags + 1)])

    return np.concatenate([spec, hist, ac]).astype(np.float64)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD^2 with an RBF kernel (median heuristic bandwidth)."""
    if gamma is None:
        Z = np.vstack([X, Y])
        d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, -1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (med + 1e-12)
    m, n = len(X), len(Y)
    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_x = Kxx.sum() / (m * (m - 1))
    term_y = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.sum() / (m * n)
    return float(term_x + term_y - 2 * term_xy)


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
                         seed: int = 0) -> dict:
    """Permutation p-value for H0: X and Y from the same distribution.

    Standardize columns jointly first so no single scale dominates the kernel.
    """
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Xs, Ys = (X - mu) / sd, (Y - mu) / sd
    Zs = np.vstack([Xs, Ys])
    m = len(Xs)
    # fix one bandwidth for all permutations (computed on the pooled set)
    d2 = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, -1)
    gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-12)
    obs = mmd2_unbiased(Xs, Ys, gamma)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(Zs))
        Xp, Yp = Zs[perm[:m]], Zs[perm[m:]]
        if mmd2_unbiased(Xp, Yp, gamma) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"mmd2": obs, "p_value": p, "gamma": float(gamma)}
