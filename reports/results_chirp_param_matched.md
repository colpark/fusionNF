# Results — Synthetic (chirp/AM) study, parameter-matched

Definitive run on the synthetic FM/AM generator. CUDA, budget-matched
(steps=5000, n_train=1024, n_test=256, Adam lr=1e-3 + warmup + grad-clip,
recon-weight 0.3 for NF arms), 3 seeds for the sweep. **Parameter-matched** unless
noted (late/early widened to LAINR's ~351K params); the assertion holds (max/min ≤ 1.6).

## Phase-4 gate (easy) — parameter-matched
Plan: late h168→357,065 · early h140→367,113 · nf_lainr h64→351,109 · nf_omnifield h76→367,909

| family | test_acc | val_acc | params | enc_FLOPs | fuse_FLOPs |
|---|---|---|---|---|---|
| late | 0.922 | 0.883 | 357,065 | 2,482,765,824 | 2,505,216 |
| early | 0.898 | 0.898 | 367,113 | 90,316,800 | 1,456,008,960 |
| **nf_lainr** | 0.918 | 0.891 | 351,109 | 837,287,936 | **790,528** |
| nf_omnifield | 0.891 | 0.867 | 367,909 | 2,375,811,072 | 997,120 |

All four > chance (min 0.891). On `easy` all are comparable; **LAINR has the lowest
total compute** (enc+fuse ≈ 0.84 B vs late 2.48 B, omnifield 2.38 B, early 1.55 B) and
the lowest fusion FLOPs → Pareto-dominant once parameters are matched.

## Phase-4 gate (easy) — NOT parameter-matched (native sizes), for contrast
| family | test_acc | val_acc | params | enc_FLOPs | fuse_FLOPs |
|---|---|---|---|---|---|
| late | 0.926 | 0.914 | 62,849 | 400,556,032 | 528,384 |
| early | 0.930 | 0.852 | 80,289 | 21,364,736 | 341,856,256 |
| nf_lainr | 0.895 | 0.891 | 351,109 | 837,287,936 | 790,528 |
| nf_omnifield | 0.922 | 0.906 | 269,125 | 1,791,492,096 | 790,528 |

## Phase-6 SNR sweep (base=easy, parameter-matched, mean±std over seeds 0,1,2)
test accuracy (linear probe R² of f(t) in parentheses):

| SNR dB | late | early | **nf_lainr** | nf_omnifield | LAINR−late |
|---|---|---|---|---|---|
| 12 | 0.835±.105 (.613) | 0.913±.010 (.606) | 0.910±.019 (.879) | 0.859±.023 (.902) | +0.075 |
| 6 | 0.553±.073 (.481) | 0.803±.016 (.516) | **0.868±.013 (.827)** | 0.819±.022 (.881) | +0.315 |
| 3 | 0.492±.023 (.391) | 0.596±.067 (.493) | **0.842±.016 (.859)** | 0.715±.064 (.777) | **+0.350** |
| 0 | 0.504±.031 (.371) | 0.552±.046 (.427) | **0.806±.005 (.840)** | 0.608±.039 (.616) | +0.302 |
| −3 | 0.496±.028 (.325) | 0.505±.014 (.381) | **0.758±.018 (.753)** | 0.536±.038 (.397) | +0.262 |
| −6 | 0.487±.013 (.316) | 0.521±.027 (.348) | 0.547±.078 (.560) | 0.530±.029 (.389) | +0.060 |

**Controls (shuffled-pairs, correspondence destroyed)** — all at chance:
late 0.504 · early 0.496 · nf_lainr 0.516 · nf_omnifield 0.488. (Unimodal-at-chance is
C1.) → the accuracies are genuine joint inference, not leakage.

## Verdicts (synthetic study)
- **P1 (advantage widens with difficulty): SUPPORTED, and confound-hardened.** With
  late/early carrying **5.7×/4.6× more params** (matched to LAINR), late still collapses
  to chance by SNR 3 dB while LAINR holds 0.76–0.87 through 0/−3 dB. The LAINR−late gap
  runs +0.08 → +0.35 (at 3 dB) → +0.06 (both floor at −6). The advantage is the
  neural-field inductive bias, **not capacity**.
- **P2 (NF ≈ early at lower fusion FLOPs): SUPPORTED; under matching, strengthened.**
  LAINR matches/exceeds early accuracy at 0.79 M fusion FLOPs vs early 1.46 B (~1800×);
  and at matched params LAINR is also the lowest **total** compute → Pareto-dominant.
- **P3 (f(t) more decodable from NF latent): SUPPORTED.** LAINR probe-R² 0.88→0.56 vs
  late 0.61→0.32 across the sweep; the representation gap tracks the accuracy gap.
- **P4 (LAINR vs OmniField): LAINR wins** at every SNR below 12 dB (e.g. 0 dB 0.806 vs
  0.608); the cross-modal-crosstalk field is less robust than the per-modality field+head.

## Notes
- Giving late more params added **variance, not robustness** (SNR 12: 0.835±0.105 vs the
  native-size 0.926) — consistent with a representational, not capacity, bottleneck.
- At −6 dB everything floors near chance (below the classical wavelet-coherence anchor
  ~0.61 from Phase 3) — the deep methods do not beat a model-aware demodulator in the
  hardest regime; the advantage is regime-bounded to the intermediate band.
