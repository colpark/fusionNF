# Gate — ECG + PPG cardiac demo (cheap, conventional real-world case)

A 1-D, conventional instantiation of the fusion structure, as a cheaper external-
validity test than EBC. Shared latent = instantaneous **heart rate HR(t)**; modality
**A = ECG** (sharp P-QRS-T, phase/FM-like), **B = PPG** (smooth pulse + respiratory
AM); different rates, jitter, 1/fᵝ background scaled by `snr_db` (PPG-under-motion =
the buried regime). Drop-in via `config.generator="ecg_ppg"`; the whole harness
(validation / four families / probe / sweep) is generator-agnostic, so compute stays
~1-D cheap.

**Dataset soundness — C1–C5 PASS on `ecg_ppg_easy` and `ecg_ppg_hard`:**
- C1 joint-only label (unimodal classifiers at chance), C2 marginal invariance,
  C3 multiscale + measured SNR, C4 grid mismatch (easy 1024=1024; hard 1024≠512),
  C5 ground-truth HR present.
- Generator checks: HR ≈ 42–94 bpm; label-1 HR corr = 1.000; label-0 decoupled;
  measured SNR exact. Example figures: `reports/ecg_ppg_examples/`.

**Why this confirms the theory cheaply:** PPG's HR signal gets buried by motion (our
SNR axis), respiratory sinus arrhythmia gives HR nonstationarity, and ECG~128 / PPG~64
Hz gives the grid-mismatch axis — all three P1 knobs on realistic physiology, at 1-D cost.

**Run on the remote GPU** (same targets, just point `--config`/`BASE` at the cardiac
configs; the generator dispatch is automatic):
```
uv run python -m src.models.train --config ecg_ppg_easy --steps 5000      # 4-family gate
uv run python -m src.models.train --config ecg_ppg_hard  --steps 5000
uv run make sweep BASE=ecg_ppg_easy KNOB=snr_db     VALUES="12 6 3 0 -3 -6" SEEDS="0 1 2"
uv run make sweep BASE=ecg_ppg_easy KNOB=rate_ratio VALUES="1.0 0.5 0.25"  SEEDS="0 1 2"
uv run python -m src.probe.frequency_probe --config ecg_ppg_easy --steps 5000
uv run make report
```

**Decision:** dataset validated and ready. Run the GPU sweep to test whether the
LAINR-style fusion advantage (P1/P3) reproduces on cardiac data — especially the
**rate-mismatch** axis, which is the most NF-specific and untested.
