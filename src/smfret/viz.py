"""Visualize smFRET data: simulated photon streams, the discriminative signals, the SNR
knob, and (once downloaded) the real kinSoftChallenge files. Headless (saves PNGs), so it
runs on the remote. Run:  python -m src.smfret.viz
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SimConfig, simulate_trace, fourier_freqs

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUT = os.path.join(_REPO, "reports", "smfret_examples")


def _bins(x, cfg):
    h, edges = np.histogram(x, bins=cfg.n_bins, range=(0, cfg.duration))
    return h, 0.5 * (edges[1:] + edges[:-1])


def plot_trace(cfg, tau, seed, title, path):
    rng = np.random.default_rng(seed)
    d, a, seg_t, seg_s = simulate_trace(rng, cfg, tau)
    e_state = np.where(seg_s == 1, cfg.e_high, cfg.e_low)
    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    ax[0].step(seg_t, e_state, where="post", color="k", lw=1.2, label="hidden FRET E(t)")
    ax[0].plot(a, np.full_like(a, 1.06), "|", color="C3", ms=6, label="acceptor photons")
    ax[0].plot(d, np.full_like(d, -0.06), "|", color="C0", ms=6, label="donor photons")
    ax[0].set_ylim(-0.15, 1.15); ax[0].set_ylabel("E / photons"); ax[0].legend(fontsize=7, ncol=3, loc="upper right")
    ax[0].set_title(f"{title}  (tau={tau*1000:.0f} ms, rate={cfg.photon_rate:.0f} ph/s, "
                    f"N={len(d)+len(a)} photons)")
    hd, t = _bins(d, cfg); ha, _ = _bins(a, cfg)
    ax[1].plot(t, hd, color="C0", lw=1, label="donor (binned)")
    ax[1].plot(t, ha, color="C3", lw=1, label="acceptor (binned)")
    ax[1].set_ylabel("counts/bin"); ax[1].legend(fontsize=7, loc="upper right")
    fret = ha / np.clip(hd + ha, 1, None)
    ax[2].plot(t, fret, color="C2", lw=1); ax[2].axhline(cfg.e_low, ls=":", c="grey"); ax[2].axhline(cfg.e_high, ls=":", c="grey")
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("apparent E"); ax[2].set_xlabel("time (s)")
    ax[2].set_title("binned FRET efficiency = acceptor/(donor+acceptor) — the anti-correlation signal")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_classes(cfg, path):
    """Fast vs slow: cheap-comparator (acceptor autocorr) + neural-field (Fourier) signals."""
    rng = np.random.default_rng(0); freqs = fourier_freqs(cfg)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for lbl, name, col, tau in [(1, "fast", "C1", np.mean(cfg.tau_fast)),
                                (0, "slow", "C4", np.mean(cfg.tau_slow))]:
        acs, fts = [], []
        for _ in range(40):
            d, a, _, _ = simulate_trace(rng, cfg, tau)
            ha, _ = _bins(a, cfg)
            x = ha - ha.mean(); ac = np.correlate(x, x, "full")[cfg.n_bins - 1:]; acs.append(ac / (ac[0] + 1e-9))
            # empirical Fourier transform magnitude of acceptor photon times (the NF feature)
            if len(a):
                ft = np.abs(np.exp(1j * 2 * np.pi * np.outer(freqs, a)).sum(1)) / len(a)
            else:
                ft = np.zeros(len(freqs))
            fts.append(ft)
        lag = np.arange(cfg.n_bins) * (cfg.duration / cfg.n_bins)
        ax[0].plot(lag[:cfg.n_bins // 2], np.mean(acs, 0)[:cfg.n_bins // 2], col, label=f"{name} (tau={tau*1000:.0f}ms)")
        ax[1].semilogx(freqs, np.mean(fts, 0), col, label=name)
    ax[0].set_xlabel("lag (s)"); ax[0].set_ylabel("acceptor autocorr"); ax[0].legend(fontsize=8)
    ax[0].set_title("CHEAP COMPARATOR signal: autocorr decay timescale")
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("|FT| of photon times")
    ax[1].set_title("NEURAL-FIELD signal: multi-band Fourier of raw arrival times"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_snr(cfg, tau, rates, path):
    """Same fast molecule as photon rate drops: the fast switching gets buried."""
    fig, ax = plt.subplots(len(rates), 1, figsize=(11, 1.5 * len(rates)), sharex=True)
    for j, rate in enumerate(rates):
        c = SimConfig(**{**cfg.__dict__, "photon_rate": rate})
        rng = np.random.default_rng(7)
        d, a, seg_t, seg_s = simulate_trace(rng, c, tau)
        e_state = np.where(seg_s == 1, cfg.e_high, cfg.e_low)
        ax[j].step(seg_t, e_state, where="post", color="k", lw=0.8, alpha=0.5)
        ax[j].plot(a, np.full_like(a, 1.06), "|", color="C3", ms=5)
        ax[j].plot(d, np.full_like(d, -0.06), "|", color="C0", ms=5)
        ax[j].set_ylim(-0.15, 1.15); ax[j].set_ylabel(f"{rate:.0f} ph/s\nN={len(d)+len(a)}", fontsize=8)
    ax[0].set_title(f"SNR knob: same fast molecule (tau={tau*1000:.0f} ms), brightness 高→低 — signal buried")
    ax[-1].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def inspect_real(data_dir, path):
    """Best-effort load+plot of downloaded kinSoftChallenge files (format-agnostic)."""
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    print(f"kinSoftChallenge dir: {data_dir} | {len(files)} files")
    for f in files[:40]:
        print("  ", os.path.relpath(f, data_dir), f"({os.path.getsize(f)//1024} KB)")
    # try common array formats; plot the first donor/acceptor-looking pair found
    series = None
    for f in files:
        ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
        try:
            if ext == "npy":
                series = np.load(f); break
            if ext == "npz":
                z = np.load(f); series = [z[k] for k in z.files][:2]; break
            if ext in ("csv", "dat", "txt"):
                series = np.loadtxt(f, delimiter="," if ext == "csv" else None); break
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {os.path.basename(f)}: {e})")
    if series is None:
        print("  No directly-loadable array file found; inspect formats above (likely .mat/.hdf5 "
              "TTTR — add a parser in src/smfret once the layout is confirmed).")
        return False
    arr = np.atleast_2d(np.asarray(series, dtype=float))
    fig, ax = plt.subplots(figsize=(11, 3))
    for i, row in enumerate(arr[:2]):
        ax.plot(row[:2000], lw=0.7, label=f"channel {i}")
    ax.set_title(f"Real kinSoftChallenge sample: {os.path.relpath(files[0], data_dir)}")
    ax.legend(fontsize=8); ax.set_xlabel("sample"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    return True


def main():
    os.makedirs(_OUT, exist_ok=True)
    cfg = SimConfig(photon_rate=300.0)
    plot_trace(cfg, np.mean(cfg.tau_fast), 1, "FAST switching molecule", os.path.join(_OUT, "trace_fast.png"))
    plot_trace(cfg, np.mean(cfg.tau_slow), 2, "SLOW switching molecule", os.path.join(_OUT, "trace_slow.png"))
    plot_classes(cfg, os.path.join(_OUT, "class_signals.png"))
    plot_snr(cfg, np.mean(cfg.tau_fast), [1000, 100, 20], os.path.join(_OUT, "snr_knob.png"))
    print(f"wrote figures to {_OUT}")
    real = os.path.join(_REPO, "data", "kinsoft")
    if os.path.isdir(real) and glob.glob(os.path.join(real, "**", "*"), recursive=True):
        inspect_real(real, os.path.join(_OUT, "real_sample.png"))
    else:
        print("(no real data yet; run scripts/download_kinsoft.sh to fetch kinSoftChallenge)")


if __name__ == "__main__":
    main()
