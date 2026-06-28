# Phase 4 Gate — Four Fusion Families

**Families (per user decision):** `late`, `early`, `nf_lainr`, `nf_omnifield`.
The two neural-field arms are both **amortized** real architectures (not a CNN→latent
strawman); the auto-decoded/functa arm was **dropped**, so the original P4
amortization-spectrum prediction is reframed as **LAINR vs OmniField** (a
per-modality field + head vs a cross-modal crosstalk field). This is recorded for
Phase 7.

**Built / kept:**
- `late_fusion.py` (per-modality 1D-CNN, pooling variants), `early_fusion.py`
  (cross-attention transformer over A+B tokens) — shared `encode/fuse` contract.
- `neural_field/field_decoder.py` (FiLM Fourier-feature field, no ReLU-MLP spectral
  bias), `fusion_head.py` (agreement features / cross-attention).
- **`neural_field/lainr.py`** — faithful 1-D LAINR: conv-stem token encoder →
  locality-aware cross-attention decoder with multi-band coarse-to-fine modulation.
- **`neural_field/omnifield.py`** — faithful 1-D OmniField: grid-free point-observation
  encoder (value-modulated Fourier features + query-local locality-biased attention,
  **no conv**) → bidirectional cross-modal crosstalk (ICMR) → fusion head.
- `registry.py`, `train.py` (generic loop + recon aux loss; `gate_run` CLI, GPU
  auto-detect), `recon_test.py` (per-band PSNR), and Phase-5 `probe/frequency_probe.py`.

**Acceptance criteria:**
- All four share one interface and forward correctly (shapes verified) — **PASS**.
- Encode-vs-fuse FLOPs measured separately — **PASS**. Fusion-FLOP spectrum (tiny):
  late 83K < **LAINR 148K** < OmniField 15.9M < early 17M — i.e. LAINR's fusion is
  ~100× cheaper than early fusion's (the P2 setup).
- Each family trains above chance on `easy` — **VERIFIED LEARNABLE locally; final
  gate numbers to be produced on the remote GPU.** Local CPU evidence:
  early reaches test 0.867 (train 1.0) with adequate budget; LAINR test 0.766;
  OmniField overfits 32 samples to train 1.000 (proven learnable) and needs the
  longer GPU budget to generalize (it is the heaviest arm).
- NF reconstruction PSNR per band — runner ready (`make recon`); to be reported from
  the remote run.

**Debugging notes (OmniField, all fixed):** its grid-free encoder initially collapsed
to a constant. Root causes, fixed in order: (1) `adaptive_avg_pool` averaged out the
oscillation — switched to true point observations; (2) a 1-D value concatenated with
64-D Fourier position let position swamp the value — added value-modulated Fourier
features so frequency survives attention-averaging; (3) near-uniform attention averaged
the whole signal — added an `M²`-scaled query-local locality bias. After (3) it overfits
to 1.000. No conv stem was used (would change OmniField's architecture).

**How to run (remote GPU):**
```
uv run make train   CONFIG=easy STEPS=3000 NTRAIN=1024   # all four -> reports/phase4_results.json
uv run make recon                                        # NF per-band reconstruction
```

**Decision needed:** confirm the gate on the remote GPU (expect all four > chance,
early ≳ LAINR > late; OmniField depends on budget), then proceed.
