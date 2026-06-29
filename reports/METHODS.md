# Methods — Synthetic Neural-Field Fusion Study (comprehensive, reproducible log)

This document records exactly what was done to produce the parameter-matched synthetic
(chirp/AM) result in `reports/results_chirp_param_matched.md`. All components are in
`src/`; commands are in the Makefile. Everything is seeded; each run writes its resolved
config, git SHA, and seed to its output directory.

---

## 1. Task and problem formulation

Two 1-D modalities A and B each carry one slowly-varying instantaneous-frequency
trajectory f(t). A pair is labeled **y=1** iff A and B carry the *same* f(t), else **y=0**.
A model receives only the four arrays (A, t_A, B, t_B) and outputs a logit; the task is
binary correspondence (chance = 0.5). The label is **joint-only** by construction: each
modality's own f(t) is always a fresh prior draw, so neither modality alone is predictive.

---

## 2. Data generator (`src/data/generator.py`)

### 2.1 Shared trajectory prior
f(t) = clip( f0 + Σ_{k=1..K} a_k · sin(2π ν_k t + φ_k), f_min, f_max ), with per-sample
draws ν_k ~ U(nu_min, nu_max), a_k ~ U(amp_min, amp_max), φ_k ~ U(0, 2π). K = n_components.

### 2.2 Observation models
- **Modality A — FM chirp:** A(t) = c·sin(2π ∫₀ᵗ f_A(s) ds) + n_A(t), uniform grid at
  rate r_A over [0, T]. Phase via cumulative trapezoid of f_A. (c = `signal_amp`.)
- **Modality B — AM tone:** B(t) = (1 + m·g(f_B(t)))·sin(2π·f_carrier·t) + n_B(t), at
  rate r_B, where g maps f∈[f_min,f_max] to [−1,1] (g=(f−f_mid)/f_half) and m = `am_depth`.
  Guard: f_carrier must satisfy f_max < f_carrier < r_B/2 (else B aliases; enforced).

### 2.3 Multiscale background and SNR
n_A, n_B are sums of the top S=`n_octaves` octave bands of white noise, each band weighted
∝ center_freq^(−β/2) (β = `noise_beta`), giving a 1/f^β multiscale background, unit-variance
normalized. Noise is then scaled so that **SNR_dB = 10·log10(var(clean)/var(noise))**
exactly equals `snr_db` (the difficulty knob); measured SNR matches config to ~1e-2 dB.

### 2.4 Grids, jitter, labels
t_A, t_B are uniform at r_A, r_B (r_A ≠ r_B ⇒ different lengths). With `jitter`>0, t_B is
perturbed by `jitter`·Δt·N(0,1) and made monotone (irregular sampling). Label y ~
Bernoulli(`p_positive`); y=1 ⇒ params_B = params_A; y=0 ⇒ params_B is an independent draw.
`generate(config, seed)` is a pure function of (config, seed) and can return clean/noise
components for validation.

### 2.5 Canonical configs (`configs/`)
- **easy:** T=4 s, snr_db=12, jitter=0; trajectory f0=5, [f_min,f_max]=[1,12],
  n_components=2, nu∈[0.05,0.15], amp∈[0.5,1.0]; A rate 128, S=4; B rate 128, carrier 40,
  S=4. (512/512 samples.)
- **hard:** snr_db=−3, jitter=0.02; n_components=4, nu∈[0.1,0.5], amp∈[0.5,2.5];
  A rate 160 (S=6); B rate 96, carrier 40 (S=6). (640/384 samples.)

---

## 3. Splits, normalization (`src/data/{dataset,transforms}.py`)

Train/val/test use **disjoint generative-seed ranges** (seed = base·1e8 + offset + i, with
offsets train 0 / val 1e7 / test 2e7) — no sample can appear in two splits. Normalization
statistics (per-modality mean/std) are computed on **train only**; signals are standardized
and coordinates mapped to [0,1] before models see them.

---

## 4. Dataset validation (`src/validation/`, criteria C1–C7)

Run before any modeling; all pass on easy and hard.
- **C1 joint-only label:** train high-capacity 1-D-CNN classifiers on A-only and B-only;
  assert test accuracy not significantly above chance (one-sided binomial, α=0.01).
- **C2 marginal invariance:** per modality, summary-stat vectors (log power spectrum,
  amplitude histogram, autocorrelation) for class 0 vs 1; unbiased MMD with a 200-permutation
  test; assert non-rejection (p>0.05).
- **C3 multiscale + SNR:** octave band-power decomposition shows ≥S resolvable bands;
  measured clean-vs-noise SNR within 1 dB of config.
- **C4 grid mismatch/jitter:** lengths differ iff rates differ; jitter present iff configured.
- **C5 ground truth:** f_A, f_B present, finite, aligned to A, B.
- **C6 controllable difficulty:** SNR mini-sweep ⇒ oracle flat (~1.0), classical baseline
  monotone in SNR.
- **C7 in-principle solvable:** oracle ≈ 1.0 ≫ classical wavelet-coherence > chance.

---

## 5. Baselines / brackets (`src/baselines/`)

- **Oracle (upper bound):** sees f_A, f_B; statistic = Pearson corr of f_A vs f_B
  resampled to t_A; a single threshold fit on train. ≈1.0 (easy 1.000, hard 0.988).
- **Classical anchor (wavelet coherence):** no ground truth — FM-demodulate A by Hilbert
  instantaneous frequency, AM-demodulate B (carrier bandpass → Hilbert envelope → invert g),
  z-score both on a common grid, Pearson corr, threshold on train. (easy 0.688/0.715,
  hard 0.613.) Chance ≈ 0.5. Establishes oracle ≫ classical ≫ chance.

---

## 6. Fusion families (`src/models/`)

Common interface (`common.py`): every model implements `encode(A,t_A,B,t_B)→dict`
(must contain a flat representation `z` of shape (B, D) for the probe) and `fuse(encoded)→
logit`; `forward = fuse(encode(...))`. This two-stage split lets FLOPs be measured
separately for **representation** (encode) vs **fusion** (fuse). All families share a
common width `hidden`, latent_dim=32, depth=2 unless parameter-matching changes `hidden`.

### 6.1 Late fusion (`late_fusion.py`)
Per-modality `SignalCNN`: Conv1d(1→h, k7,s2,p3)–GELU – Conv1d(h→h, k5,s2,p2)–GELU –
Conv1d(h→latent, k3,s2,p1), then pooling ∈ {mean (default), attention, last} → z_A, z_B.
Fuse: MLP over [z_A, z_B]. Fusion cost is O(latent), independent of sequence length.

### 6.2 Early fusion (`early_fusion.py`)
Per modality: Conv1d patch embedding (patch = round(N/32) ⇒ ~32 tokens) + coordinate
positional MLP + **LayerNorm**; add a modality-type embedding; prepend a CLS token. Fuse:
a Transformer encoder (d=hidden, 4 heads, FFN=2·hidden, GELU, norm-first, `depth` layers)
over the concatenated A+B tokens; CLS → logit. Fusion cost scales with total token count
(the expensive cross-modal reference). z = projected mean token.

### 6.3 LAINR — per-modality neural field (`neural_field/lainr.py`)
Amortized, locality-aware, multi-band (Lee et al., NeurIPS 2023), 1-D adaptation.
- **Encoder (per modality):** conv stem (k7s2 / k5s2 / k5s2 ⇒ /8) → adaptive-pool to
  **n_tokens=32** tokens → add positional MLP of token-center time τ → `depth`-layer
  Transformer (4 heads). Tokens carry local information; token i has center time τ_i.
- **Locality-aware decoder:** query coordinate → cross-attention to tokens with a learnable
  Gaussian **locality bias** −softplus(log_locality)·(t−τ)² (init 2.0); the resulting
  modulation drives a **multi-band coarse-to-fine** field: **n_bands=4** Fourier-feature
  groups (16 feats/band, geometric bands 1→f_max), each FiLM-modulated and accumulated.
  f_max = clip(⌈f_max·T·1.5⌉, 32, 128) (=72 on easy).
- **Fusion:** z_A,z_B = projected mean tokens (latent_dim=32) → shared `FusionHead`.
  No cross-modal interaction before the head. Reconstruction aux loss decodes each
  modality from its own tokens.

### 6.4 OmniField — cross-modal conditioned field (`neural_field/omnifield.py`)
Grid-free (no conv), 1-D adaptation of OmniField (arXiv 2511.02205).
- **Encoder (per modality):** point observations (true (t,value), subsampled to ≤512 if
  longer — never averaged). Per-observation feature = [value, GFF(t), value·GFF(t)] with
  Gaussian Fourier features (n_gff=32, σ=8) — the value·Fourier terms keep frequency
  recoverable under attention. **n_latents=16** learnable query tokens, anchored at
  linspace(0,1), cross-attend to observations with a **query-local** locality bias scaled
  by 4·n_latents² (=1024) so each query summarizes a sharp local window.
- **Cross-modal crosstalk (ICMR):** 3 bidirectional cross-attention blocks (4 heads)
  between the two modalities' latent sets — used in the **reconstruction** path.
- **Decision:** made on the **pre-crosstalk per-modality field latents** (z_A,z_B →
  `FusionHead`); crosstalk-mixed summaries homogenize the modalities and were verified to
  sit at chance, so crosstalk serves reconstruction, not the decision.

### 6.5 Shared fusion head (`neural_field/fusion_head.py`)
`FusionHead` (mode mlp): features [z_A, z_B, z_A·z_B, |z_A−z_B|] → MLP
[4·latent, hidden, hidden, 1]. Identical across both NF arms so accuracy differences
reflect latent quality, not the classifier. (An "xattn" variant exists but mlp was used.)

### 6.6 Field decoder rationale
NF decoders use **Fourier-feature / multi-band** bases, never a plain ReLU MLP: spectral
bias of a ReLU coordinate-MLP would erase the high-frequency matching band — a confound,
not a finding.

---

## 7. Budget and parameter matching (`src/models/train.py`)

Comparisons match **optimizer, steps, batch, and data**. Additionally, **parameter
matching** (`matched_plan`) widens each family's `hidden` (search step 4, param count
monotone in hidden) so all four reach the largest family's parameter count, with an
**assertion that max/min param ratio ≤ 1.6**. On easy this gives: late h168→357,065,
early h140→367,113, nf_lainr h64→351,109, nf_omnifield h76→367,909 (ratio 1.05).
`--no-match-params` recovers the native-size comparison.

---

## 8. Training protocol

- Loss: BCEWithLogits; NF arms add `recon_weight`·reconstruction MSE (default 0.3) so the
  field's reconstruction objective shapes the latent without starving the classifier.
- Optimizer Adam, lr 1e-3, **linear LR warmup** over the first 10% of steps, **gradient
  clipping** at norm 1.0 (applied uniformly to all families — fair, and required for the
  transformer's stability).
- Gate/sweep budget: **steps 5000, n_train 1024, batch 32, n_test 256.**
- **Determinism:** `seeded_build` seeds the RNG *before* model construction (so init is
  reproducible) and `train_model` reseeds before the loop; `enable_determinism` sets
  deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG`. CPU same-seed runs are bit-identical;
  residual GPU nondeterminism is why sweeps use 3 seeds.
- Device auto-detected (CUDA→MPS→CPU).

---

## 9. FLOP accounting (`src/utils/flops.py`)

`encode` and `fuse` are each wrapped in a torch `FlopCounterMode` scope on one batch, so
representation-FLOPs and fusion-FLOPs are reported separately (the split where the cost/
fidelity tension lives).

---

## 10. Probe diagnostic (`src/probe/frequency_probe.py`)

The representation z is **frozen**; we regress the ground-truth trajectory (resampled to
M=64 points) from z and report test R². Two probes: a **closed-form ridge** linear probe
(λ=1.0) and a **small-MLP** probe (hidden 128, 800 Adam steps, lr 1e-3, wd 1e-4). We probe
f_A and f_B and average. This factors the hypothesis: probe✓&fusion✓ = mechanism confirmed;
probe✓&fusion✗ = fusion-head bottleneck; probe✗ = representation discarded the factor.

---

## 11. Sweep and controls (`src/sweep/`)

`runner.py` runs {difficulty knob × family × seed}. Knobs: `snr_db`, `jitter`,
`nonstationarity` (trajectory nu_max), `n_components`, `rate_ratio` (B-rate / A-rate). Per
cell it records accuracy (mean±std over seeds), probe linear-R², and the encode/fuse FLOP
split. **Controls:** (i) shuffled-pairs — train with B permuted across the batch
(correspondence destroyed) ⇒ must be chance (detects leakage); (ii) unimodal C1 ⇒ chance.
`pareto.py` plots accuracy vs fusion-FLOPs and vs total-FLOPs, and accuracy/probe-R² vs the
knob with seed error bars.

---

## 12. Metrics, reporting, reproduction

Primary metric: test correspondence accuracy (chance 0.5). Diagnostic: probe R². Cost:
encode/fuse FLOPs and parameter counts. Headline figures: accuracy-vs-compute Pareto and
accuracy/probe-R²-vs-difficulty. The findings generator (`src/findings.py`) reads the saved
JSON artifacts and renders verdicts; `make report` regenerates them.

Reproduce (CUDA auto-detected):
```
uv sync
uv run make validate
uv run make baselines
uv run python -m src.models.train --config easy --steps 5000        # param-matched gate
uv run make sweep BASE=easy KNOB=snr_db VALUES="12 6 3 0 -3 -6" SEEDS="0 1 2"
uv run python -m src.probe.frequency_probe --config easy --steps 5000
uv run make report
```
Results of this exact protocol are tabulated in `reports/results_chirp_param_matched.md`.
