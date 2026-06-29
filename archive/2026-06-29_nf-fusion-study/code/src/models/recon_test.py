"""Per-frequency-band INR reconstruction test for the NF families (Phase 4).

A neural field is only useful for this task if its latent ``z`` actually captures
the signal -- in particular the *matching band* that carries the cross-modal
correspondence. A field with strong spectral bias (or an under-fit latent) can post
a low global reconstruction error while completely missing a specific band, which
would silently destroy the matching signal. So we decompose both the original
(normalized) signal and its reconstruction into the SAME octave bands the generator
uses and report reconstruction quality per band.

Metric: per-band reconstruction SNR in dB,

    SNR_band = 10 * log10( sum(orig_band^2) / sum((orig_band - recon_band)^2) )

aggregated over a batch of test signals. ~0 dB means the field puts no correct
energy in that band (junk for that band); higher is better. Bands below a small
threshold are flagged.

Run: ``uv run python -m src.models.recon_test`` -> writes reports/phase4_recon.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.fft import irfft, rfft, rfftfreq

from ..config import ModelConfig, TrainConfig, load_experiment
from ..data.dataset import make_loaders
from ..data.transforms import apply_norm, fit_norm
from .train import train_model, seeded_build

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORTS = _REPO_ROOT / "reports"
_FLAG_DB = 3.0   # reconstruction SNR below this (dB) => band effectively not recovered


def octave_bands(rate: float, n_octaves: int) -> list[tuple[float, float]]:
    """Octave band edges (Hz), matching generator.octave_background, plus a low
    remainder band ``[0, f_lo_min)`` so the decomposition covers the full spectrum.
    """
    nyq = rate / 2.0
    bands = [(nyq / 2.0 ** (s + 1), nyq / 2.0 ** s) for s in range(n_octaves)]
    bands.append((0.0, nyq / 2.0 ** n_octaves))   # low-frequency remainder
    return bands


def _band_filter(x: np.ndarray, rate: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band-limit of a 1-D signal via rfft masking."""
    n = x.shape[-1]
    freqs = rfftfreq(n, d=1.0 / rate)
    mask = (freqs >= lo) & (freqs < hi)
    X = rfft(x)
    return irfft(X * mask, n=n)


def band_snr(orig: np.ndarray, recon: np.ndarray, rate: float,
             bands: list[tuple[float, float]]) -> list[float]:
    """Aggregate per-band reconstruction SNR (dB) over a batch ``(B, N)``."""
    out = []
    for lo, hi in bands:
        po = 0.0
        pe = 0.0
        for i in range(orig.shape[0]):
            ob = _band_filter(orig[i], rate, lo, hi)
            rb = _band_filter(recon[i], rate, lo, hi)
            po += float(np.sum(ob ** 2))
            pe += float(np.sum((ob - rb) ** 2))
        snr = 10.0 * np.log10(po / pe) if pe > 0 and po > 0 else float("nan")
        out.append(snr)
    return out


@torch.no_grad()
def _reconstruct(model, batch: dict) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Return (orig_A, recon_A, orig_B, recon_B) as numpy arrays (normalized)."""
    recon_a, recon_b = model.reconstruct(batch["A"], batch["t_A"],
                                         batch["B"], batch["t_B"])
    return (batch["A"].cpu().numpy(), recon_a.cpu().numpy(),
            batch["B"].cpu().numpy(), recon_b.cpu().numpy())


def run() -> str:
    exp = load_experiment(str(_REPO_ROOT / "configs" / "easy.yaml"))
    data_cfg = exp.data
    base_seed = exp.seed
    rate_a = data_cfg.modality_a.rate
    rate_b = data_cfg.modality_b.rate
    oct_a = data_cfg.modality_a.n_octaves
    oct_b = data_cfg.modality_b.n_octaves
    bands_a = octave_bands(rate_a, oct_a)
    bands_b = octave_bands(rate_b, oct_b)

    # Short training focused on reconstruction quality (not classification).
    from ..utils.device import auto_device
    tc = TrainConfig(steps=600, batch_size=32, lr=1e-3, optimizer="adam",
                     n_train=512, n_val=64, n_test=128, log_every=300,
                     device=auto_device())
    stats = fit_norm(data_cfg, base_seed, tc.n_train)
    test_loader = make_loaders(data_cfg, base_seed, tc.n_train, tc.n_val,
                               tc.n_test, tc.batch_size)["test"]
    eval_batch = apply_norm(next(iter(test_loader)), stats)
    eval_batch = {k: v.to(tc.device) for k, v in eval_batch.items()}

    variants = {"nf_lainr": "LAINR (amortized; locality-aware multi-band decoder)",
                "nf_omnifield": "OmniField (amortized; cross-modal crosstalk field)"}

    sections = []
    summary = {}
    for family, desc in variants.items():
        model_cfg = ModelConfig(family=family, hidden=exp.model.hidden,
                                depth=exp.model.depth,
                                latent_dim=exp.model.latent_dim)
        model = seeded_build(family, model_cfg, data_cfg, base_seed)
        res = train_model(model, data_cfg, model_cfg, tc, base_seed)
        orig_a, recon_a, orig_b, recon_b = _reconstruct(model, eval_batch)
        snr_a = band_snr(orig_a, recon_a, rate_a, bands_a)
        snr_b = band_snr(orig_b, recon_b, rate_b, bands_b)
        summary[family] = {"test_acc": res["test_acc"], "snr_a": snr_a,
                           "snr_b": snr_b}

        lines = [f"## {family}", "", f"_{desc}_", "",
                 f"- test accuracy: **{res['test_acc']:.3f}**  "
                 f"(params={res['n_params']:,})", ""]
        for tag, rate, bands, snr in [("A (chirp/FM)", rate_a, bands_a, snr_a),
                                      ("B (AM tone)", rate_b, bands_b, snr_b)]:
            lines.append(f"### Modality {tag}  (rate={rate:g} Hz)")
            lines.append("")
            lines.append("| band (Hz) | recon SNR (dB) | status |")
            lines.append("|---|---:|---|")
            for (lo, hi), s in zip(bands, snr):
                status = "ok" if (s == s and s >= _FLAG_DB) else "FLAG: not recovered"
                sval = f"{s:.2f}" if s == s else "n/a"
                lines.append(f"| {lo:.1f}–{hi:.1f} | {sval} | {status} |")
            lines.append("")
        sections.append("\n".join(lines))

    md = ["# Phase 4 -- Neural-field reconstruction by frequency band", "",
          f"Config: **easy** (duration={data_cfg.duration:g}s, "
          f"snr_db={data_cfg.snr_db:g}). Trained {tc.steps} steps on "
          f"{tc.n_train} signals; evaluated on {eval_batch['A'].shape[0]} test "
          "signals.", "",
          "Metric: per-band reconstruction SNR (dB) of the field against the "
          "normalized signal, using the generator's octave bands plus a "
          "low-frequency remainder band. A band flagged 'not recovered' "
          f"(SNR < {_FLAG_DB:g} dB) means the latent carries little correct energy "
          "there -- if that is the matching band, the latent is junk for fusion.",
          ""]
    md.extend(sections)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS / "phase4_recon.md"
    out_path.write_text("\n".join(md))

    # Console summary.
    print(f"Wrote {out_path}\n")
    for family, s in summary.items():
        best_a = max((v for v in s["snr_a"] if v == v), default=float("nan"))
        best_b = max((v for v in s["snr_b"] if v == v), default=float("nan"))
        print(f"{family:<16} test_acc={s['test_acc']:.3f}  "
              f"best-band SNR  A={best_a:.1f}dB  B={best_b:.1f}dB")
    return str(out_path)


if __name__ == "__main__":
    run()
