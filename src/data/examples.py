"""Render example pairs (signals + spectrograms + f(t) overlay) for the Phase 1 gate.

Run: python -m src.data.examples
Writes PNGs to reports/phase1_examples/.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from ..config import load_experiment
from .generator import generate, Sample


def _first_with_label(cfg, label: int, return_components=True) -> Sample:
    for seed in range(200):
        s = generate(cfg, seed, return_components=return_components)
        if s.label == label:
            return s
    raise RuntimeError("no sample with requested label found")


def _panel(ax_sig, ax_spec, t, x, rate, title, f_overlay=None, t_f=None):
    # short time window so the oscillation is visible
    win = t <= min(t[-1], 1.0)
    ax_sig.plot(t[win], x[win], lw=0.6)
    ax_sig.set_title(title, fontsize=9)
    ax_sig.set_xlabel("t (s)")
    nperseg = min(128, len(x) // 4)
    f, tt, Sxx = spectrogram(x, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
    ax_spec.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    if f_overlay is not None:
        ax_spec.plot(t_f, f_overlay, "r-", lw=1.2, label="f(t)")
        ax_spec.legend(fontsize=7, loc="upper right")
    ax_spec.set_ylabel("freq (Hz)")
    ax_spec.set_xlabel("t (s)")


def render(tag: str, out_dir: str):
    cfg = load_experiment(f"configs/{tag}.yaml").data
    for label in (1, 0):
        s = _first_with_label(cfg, label)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        _panel(axes[0, 0], axes[1, 0], s.t_A, s.A, cfg.modality_a.rate,
               f"A (chirp/FM)  label={label}", f_overlay=s.f_A, t_f=s.t_A)
        # For B, overlay its trajectory but note B's energy sits around the carrier.
        _panel(axes[0, 1], axes[1, 1], s.t_B, s.B, cfg.modality_b.rate,
               f"B (AM)  label={label}", f_overlay=s.f_B, t_f=s.t_B)
        fig.suptitle(f"{tag}  |  snr_db={cfg.snr_db}  r_A={cfg.modality_a.rate} "
                     f"r_B={cfg.modality_b.rate}  jitter={cfg.jitter}", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{tag}_label{label}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)

    # Also a direct f_A vs f_B overlay (the crux) for a pos and neg pair.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, label in zip(axes, (1, 0)):
        s = _first_with_label(cfg, label)
        ax.plot(s.t_A, s.f_A, label="f_A")
        ax.plot(s.t_B, s.f_B, "--", label="f_B")
        ax.set_title(f"{tag} trajectories  label={label}", fontsize=9)
        ax.set_xlabel("t (s)"); ax.set_ylabel("f (Hz)"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_trajectories.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    out_dir = "reports/phase1_examples"
    os.makedirs(out_dir, exist_ok=True)
    for tag in ("easy", "hard"):
        render(tag, out_dir)


if __name__ == "__main__":
    main()
