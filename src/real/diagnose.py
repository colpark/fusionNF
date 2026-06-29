"""Diagnostics for the real ECG+PPG (BIDMC) correspondence task.

The neural models sat at chance on real data. Before touching the models, bracket the
TASK: is it solvable at all, and where does the information break? We compute, on the
real correspondence pairs:

  - ORACLE accuracy: threshold on the ground-truth HR-correspondence statistic
    (uses HR from ECG R-peaks). ~1.0 confirms the label is well-defined by HR.
  - CLASSICAL accuracy: estimate HR from the RAW ECG and the RAW PPG independently
    (autocorrelation) and compare -> the realistic ceiling for an HR-extraction method.
    High => task solvable from signals (neural failure is preprocessing/training).
    Chance => extracting+comparing HR from real signals is hard / HR overlaps too much.
  - HR-separability: within-window HR variation (is f(t) ~constant?) and the
    distribution of |meanHR_A - meanHR_B| for NEGATIVE pairs (how many coincidentally
    share HR -> irreducible label noise).
  - HR-estimation error of the classical estimator vs ground truth.

Run: python -m src.real.diagnose   (uses data/bidmc; CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bidmc import RealConfig, build_windows, RealECGPPG, hr_from_ecg
from ..baselines.threshold import fit_threshold, accuracy_at

_REPO = Path(__file__).resolve().parents[2]


def estimate_hr_autocorr(x, rate, lo_hz=0.6, hi_hz=3.0):
    """Dominant periodicity (Hz) of a signal via autocorrelation -> heart rate."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.std() < 1e-8:
        return np.nan
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag_lo, lag_hi = int(rate / hi_hz), int(rate / lo_hz)
    seg = ac[lag_lo:lag_hi]
    if len(seg) < 2:
        return np.nan
    return rate / (lag_lo + int(np.argmax(seg)))


def _pairs(windows, cfg, split_seed, n):
    ds = RealECGPPG(windows, cfg, split_seed=split_seed, length=n)
    return [ds.raw(i) for i in range(n)]


def _stats(pairs, cfg):
    """Return (oracle_stat, classical_stat, labels) arrays. stat high => 'same'."""
    o, c, y = [], [], []
    for s in pairs:
        o.append(-abs(float(np.mean(s["f_A"])) - float(np.mean(s["f_B"]))))   # true HR
        hr_e = estimate_hr_autocorr(s["A"], cfg.ecg_rate)
        hr_p = estimate_hr_autocorr(s["B"], cfg.ppg_rate)
        c.append(-abs(hr_e - hr_p) if np.isfinite(hr_e) and np.isfinite(hr_p) else -99.0)
        y.append(int(s["label"]))
    return np.array(o), np.array(c), np.array(y)


def run(n_train=600, n_test=600):
    cfg = RealConfig()
    wins, paths = build_windows(cfg)
    tr = _pairs(wins["train"], cfg, 0, n_train)
    te = _pairs(wins["test"], cfg, 2, n_test)

    o_tr, c_tr, y_tr = _stats(tr, cfg)
    o_te, c_te, y_te = _stats(te, cfg)
    th_o = fit_threshold(o_tr, y_tr); th_c = fit_threshold(c_tr, y_tr)
    oracle_acc = accuracy_at(o_te, y_te, th_o)
    classical_acc = accuracy_at(c_te, y_te, th_c)

    # within-window HR variation (is f(t) ~constant?) -- on test ECG windows
    hr_ranges = []
    for w in wins["test"][:400]:
        hr = w["hr_ecg"]; hr_ranges.append((hr.max() - hr.min()) * 60.0)  # bpm spread
    hr_spread = float(np.median(hr_ranges))

    # negative-pair true HR gap distribution -> coincidental-match (label-noise) rate
    neg = [abs(np.mean(s["f_A"]) - np.mean(s["f_B"])) * 60.0 for s in te if s["label"] == 0]
    neg = np.array(neg)
    frac = {f"<{k}bpm": float(np.mean(neg < k)) for k in (2, 5, 10)}

    # classical HR-estimation error vs ground truth (on positives: both true HRs equal)
    err_e, err_p = [], []
    for s in te:
        hr_e = estimate_hr_autocorr(s["A"], cfg.ecg_rate)
        hr_p = estimate_hr_autocorr(s["B"], cfg.ppg_rate)
        if np.isfinite(hr_e): err_e.append(abs(hr_e - np.mean(s["f_A"])) * 60.0)
        if np.isfinite(hr_p): err_p.append(abs(hr_p - np.mean(s["f_B"])) * 60.0)
    est_err = {"ecg_bpm": float(np.median(err_e)), "ppg_bpm": float(np.median(err_p))}

    out = {"records": len(paths), "n_test": n_test,
           "oracle_acc": oracle_acc, "classical_acc": classical_acc,
           "within_window_HR_spread_bpm_median": hr_spread,
           "negative_HR_gap_frac": frac,
           "classical_HR_estimation_error_bpm": est_err}

    print(f"BIDMC diagnostics  ({len(paths)} records, n_test={n_test})\n")
    print(f"  ORACLE (true HR)        accuracy = {oracle_acc:.3f}   (label well-defined by HR?)")
    print(f"  CLASSICAL (HR from sig) accuracy = {classical_acc:.3f}   (solvable from signals?)")
    print(f"  within-window HR spread (median) = {hr_spread:.1f} bpm   (f(t) ~constant if small)")
    print(f"  negatives with small true HR gap : {frac}   (coincidental matches = label noise)")
    print(f"  classical HR est. error (median) : ECG {est_err['ecg_bpm']:.1f} bpm, "
          f"PPG {est_err['ppg_bpm']:.1f} bpm")
    print("\nReading:")
    if oracle_acc > 0.95 and classical_acc > 0.8:
        print("  Task IS solvable (oracle & classical high) -> neural chance is a "
              "preprocessing/training failure, not the task. Fix extraction.")
    elif oracle_acc > 0.95 and classical_acc < 0.65:
        print("  Label is HR-defined (oracle high) but NOT recoverable from raw signals "
              "(classical ~chance): HR is near-constant/overlapping &/or estimation error "
              "exceeds the negative HR gap -> task is ill-conditioned; REDESIGN "
              "(richer latent: longer windows/HRV; harder negatives with min HR separation).")
    else:
        print("  Even the oracle is weak -> the label itself is poorly separated; redesign needed.")

    (_REPO / "reports").mkdir(exist_ok=True)
    (_REPO / "reports" / "real_ecg_ppg_diagnostics.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/real_ecg_ppg_diagnostics.json")
    return out


if __name__ == "__main__":
    run()
