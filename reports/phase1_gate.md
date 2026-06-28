# Phase 1 Gate — Data Generator

**Built:** `src/data/generator.py` (chirp/FM modality A, AM modality B, multiscale
1/f^β octave background, exact SNR scaling, grid mismatch + timing jitter, shared
trajectory prior with label-by-coupling construction), `dataset.py` (seed-disjoint
train/val/test splits), `transforms.py` (train-only normalization), and an example
renderer. Example figures in `reports/phase1_examples/`.

**Acceptance criteria (visual + numeric on `easy` and `hard`):**
- Class-1 pairs share f(t); class-0 don't — **PASS** (label-1 corr(f_A, f_B)=1.0000
  exactly; label-0 decoupled; trajectory-overlay figures confirm).
- Both modalities carry recognizable multiscale background — **PASS** (spectrograms
  show 1/f^β energy across octaves over the clean component).
- Intended observation models visible — **PASS** (A's spectral ridge tracks f(t);
  B shows the 40 Hz AM carrier with f(t) encoded in sidebands).
- Configured difficulty realized — **PASS** (measured SNR matches config exactly:
  easy=12.00 dB, hard=−3.00 dB on both modalities; `hard` grids mismatch 640 vs
  384 samples; jitter present when configured, std(Δt_B)≈3e-4).

**Surprising / notes:**
- Independent (label-0) trajectories still have positive corr (~0.5–0.7) on a
  single sample because both hover around f0 within a bounded range. This is fine
  and expected: the oracle separates label-1 (corr exactly 1.0) from the label-0
  *distribution* (spread below 1). It does not create a unimodal shortcut — C1
  (Phase 2) tests that observations alone don't leak the label.
- Marginal invariance (C2) holds *by construction*: each modality's own trajectory
  is always a fresh prior draw, so per-modality marginals are identical across
  classes; only the coupling differs. Phase 2 verifies this empirically.

**Decision needed:** proceed to Phase 2 (validation), or adjust the generator?
(Proceeding to build Phases 2–4 per request; will report each gate.)
