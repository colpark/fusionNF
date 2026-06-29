"""Real ECG+PPG data (BIDMC, PhysioNet) for the cardiac fusion demonstration.

BIDMC CSVs have columns: 'Time [s], RESP, PLETH, V, AVR, II' at 125 Hz. We use
II (ECG) and PLETH (PPG). Ground-truth heart rate HR(t) is derived from ECG R-peaks
(the reference), serving as the shared latent f(t) for the probe.

We segment each record into fixed windows; a window stores (ecg, ppg, hr, record_id).
Correspondence pairs: positive = (ECG, PPG) from the SAME window (share HR); negative =
ECG from one window + PPG from a different record's window (independent HR). PPG is
downsampled to a lower rate than ECG to exercise the grid-mismatch axis. Splits are by
RECORD (subject) -- no leakage. Optional synthetic motion noise at a target SNR recreates
the P1 difficulty sweep on real morphology.

Download: scripts/download_bidmc.sh  ->  data/bidmc/bidmc_XX_Signals.csv
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import find_peaks, resample
from torch.utils.data import Dataset, DataLoader

from ..data.generator import octave_background, _apply_snr

FS = 125.0  # BIDMC sampling rate (Hz)


def load_record(path):
    """Return (ecg, ppg) float arrays at 125 Hz from a BIDMC Signals CSV."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    # column names get sanitized by genfromtxt: 'II', 'PLETH'
    cols = arr.dtype.names
    ecg_col = next(c for c in cols if c.upper() == "II")
    ppg_col = next(c for c in cols if "PLETH" in c.upper())
    ecg = np.asarray(arr[ecg_col], dtype=np.float64)
    ppg = np.asarray(arr[ppg_col], dtype=np.float64)
    m = np.isfinite(ecg) & np.isfinite(ppg)
    return ecg[m], ppg[m]


def hr_from_ecg(ecg, fs=FS, t_grid=None):
    """Instantaneous HR (Hz) from ECG R-peaks, interpolated to t_grid (seconds)."""
    x = (ecg - np.median(ecg)) / (np.std(ecg) + 1e-8)
    peaks, _ = find_peaks(x, distance=int(0.4 * fs), prominence=1.0)  # <=150 bpm
    if len(peaks) < 3:
        return None
    tp = peaks / fs
    rr = np.diff(tp)
    hr = 1.0 / rr                      # Hz at midpoints
    thr = 0.5 * (tp[1:] + tp[:-1])
    hr = np.clip(hr, 0.6, 3.0)
    if t_grid is None:
        return hr
    return np.interp(t_grid, thr, hr, left=hr[0], right=hr[-1])


@dataclass
class RealConfig:
    data_dir: str = "data/bidmc"
    window_s: float = 8.0
    ecg_rate: float = 125.0
    ppg_rate: float = 64.0            # downsample PPG -> grid mismatch
    stride_s: float = 4.0            # window hop (overlap)
    snr_db: float | None = None      # None = real noise only; else inject motion noise
    noise_octaves: int = 6
    seed: int = 0


def _windows_from_record(ecg, ppg, cfg: RealConfig):
    """Yield dicts: ecg (W_e,), ppg (W_p,), t_ecg, t_ppg, hr_ecg, hr_ppg."""
    n = min(len(ecg), len(ppg))
    w = int(cfg.window_s * FS); hop = int(cfg.stride_s * FS)
    We = int(cfg.window_s * cfg.ecg_rate); Wp = int(cfg.window_s * cfg.ppg_rate)
    t_e = np.arange(We) / cfg.ecg_rate; t_p = np.arange(Wp) / cfg.ppg_rate
    out = []
    for s in range(0, n - w + 1, hop):
        e = ecg[s:s + w]; p = ppg[s:s + w]
        hr_full = hr_from_ecg(e, FS, t_grid=np.arange(w) / FS)
        if hr_full is None:
            continue
        # resample signals to their target rates; HR onto each grid
        e_r = resample(e, We); p_r = resample(p, Wp)
        hr_e = np.interp(t_e, np.arange(w) / FS, hr_full)
        hr_p = np.interp(t_p, np.arange(w) / FS, hr_full)
        out.append(dict(ecg=e_r.astype(np.float32), ppg=p_r.astype(np.float32),
                        t_ecg=t_e.astype(np.float32), t_ppg=t_p.astype(np.float32),
                        hr_ecg=hr_e.astype(np.float32), hr_ppg=hr_p.astype(np.float32)))
    return out


def build_windows(cfg: RealConfig):
    """Load all records, split by record, return {split: [windows]} and record list."""
    paths = sorted(glob.glob(os.path.join(cfg.data_dir, "*_Signals.csv")))
    if not paths:
        raise FileNotFoundError(f"no BIDMC CSVs in {cfg.data_dir} (run scripts/download_bidmc.sh)")
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(paths))
    n = len(paths)
    n_tr = max(1, int(0.6 * n)); n_va = max(1, int(0.2 * n))
    split_of = {}
    for j, idx in enumerate(order):
        split_of[idx] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
    wins = {"train": [], "val": [], "test": []}
    for idx, path in enumerate(paths):
        ecg, ppg = load_record(path)
        for w in _windows_from_record(ecg, ppg, cfg):
            w["record"] = idx
            wins[split_of[idx]].append(w)
    return wins, paths


@dataclass
class NormStats:
    a_mean: float; a_std: float; b_mean: float; b_std: float; duration: float


def fit_norm(train_windows, cfg: RealConfig) -> NormStats:
    a = np.concatenate([w["ecg"] for w in train_windows])
    b = np.concatenate([w["ppg"] for w in train_windows])
    return NormStats(float(a.mean()), float(a.std() + 1e-8),
                     float(b.mean()), float(b.std() + 1e-8), float(cfg.window_s))


class RealECGPPG(Dataset):
    """Correspondence pairs from real ECG+PPG windows. label 1 = same window (shared HR);
    label 0 = ECG and PPG from different records (independent HR)."""

    def __init__(self, windows, cfg: RealConfig, split_seed: int, length: int | None = None):
        self.W = windows; self.cfg = cfg; self.split_seed = split_seed
        self.length = length or len(windows)

    def __len__(self):
        return self.length

    def _maybe_noise(self, x, rate, rng):
        if self.cfg.snr_db is None:
            return x
        nb = octave_background(rng, len(x), rate, self.cfg.noise_octaves, 1.0)
        out, _ = _apply_snr(x.astype(np.float64) - x.mean(), nb, self.cfg.snr_db)
        return (out + x.mean()).astype(np.float32)

    def raw(self, i):
        rng = np.random.default_rng(self.split_seed * 1_000_003 + i)
        wa = self.W[i % len(self.W)]
        label = int(rng.random() < 0.5)
        if label == 1:
            wb = wa
        else:  # negative: PPG from a different record
            j = i
            for _ in range(20):
                j = int(rng.integers(len(self.W)))
                if self.W[j]["record"] != wa["record"]:
                    break
            wb = self.W[j]
        A = self._maybe_noise(wa["ecg"], self.cfg.ecg_rate, rng)
        B = self._maybe_noise(wb["ppg"], self.cfg.ppg_rate, rng)
        return dict(A=A, t_A=wa["t_ecg"], B=B, t_B=wb["t_ppg"],
                    f_A=wa["hr_ecg"], f_B=wb["hr_ppg"], label=float(label))

    def __getitem__(self, i):
        s = self.raw(i)
        return {k: torch.tensor(v) if np.isscalar(v) else torch.from_numpy(np.asarray(v))
                for k, v in s.items()}


def make_real_loaders(cfg: RealConfig, batch_size=32, per_split=None):
    wins, paths = build_windows(cfg)
    stats = fit_norm(wins["train"], cfg)

    def collate(b):
        return {k: torch.stack([x[k] for x in b]) for k in b[0]}
    loaders = {}
    for split, off in [("train", 0), ("val", 1), ("test", 2)]:
        n = per_split or len(wins[split])
        ds = RealECGPPG(wins[split], cfg, split_seed=cfg.seed * 10 + off, length=n)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    collate_fn=collate, drop_last=(split == "train"))
    return loaders, stats, {k: len(v) for k, v in wins.items()}, len(paths)


if __name__ == "__main__":
    # self-test on downloaded records
    cfg = RealConfig()
    wins, paths = build_windows(cfg)
    print(f"records={len(paths)}  windows train/val/test="
          f"{len(wins['train'])}/{len(wins['val'])}/{len(wins['test'])}")
    ds = RealECGPPG(wins["train"], cfg, split_seed=0, length=200)
    cs1, cs0 = [], []
    for i in range(200):
        s = ds.raw(i)
        fb_on_a = np.interp(s["t_A"], s["t_B"], s["f_B"])
        c = float(np.corrcoef(s["f_A"], fb_on_a)[0, 1]) if s["f_A"].std() > 1e-6 else 0.0
        (cs1 if s["label"] == 1 else cs0).append(c)
    s0 = ds.raw(0)
    print(f"shapes: ECG {s0['A'].shape} @ {cfg.ecg_rate}Hz | PPG {s0['B'].shape} @ {cfg.ppg_rate}Hz")
    print(f"HR range (ECG window): {s0['f_A'].min()*60:.0f}-{s0['f_A'].max()*60:.0f} bpm")
    print(f"HR corr  label1 mean={np.mean(cs1):.3f}  label0 mean={np.mean(cs0):.3f}  "
          f"(n1={len(cs1)}, n0={len(cs0)})")
