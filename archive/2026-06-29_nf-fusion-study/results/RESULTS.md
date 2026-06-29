# Results — Neural-Field Fusion Study (2026-06-29)

Code @ `345c17b`. Device: CUDA. Budget-matched: steps=5000, n_train=1024, n_test=256,
Adam lr=1e-3 with linear warmup + grad-clip 1.0, recon-weight 0.3 (NF arms). 3 seeds
for sweeps. Controls (shuffled-pairs, unimodal-C1) hold at chance throughout.

## Bracket (Phase 3) — establishes solvability and ceiling
| config | chance | classical wavelet-coherence | oracle (sees f) |
|---|---|---|---|
| easy | ~0.56 | 0.688 | 1.000 |
| hard | ~0.52 | 0.613 | 0.988 |

## Phase 4 gate — accuracy at the two endpoints
| family | easy test | easy val | hard test | hard val | params | enc FLOPs | fuse FLOPs |
|---|---|---|---|---|---|---|---|
| late | 0.926 | 0.914 | 0.488 | 0.531 | 62,849 | 400.6M | 528K |
| early | 0.914 | 0.883 | 0.477 | 0.523 | 80,289 | 21.4M | 341.9M |
| nf_lainr | 0.902 | 0.898 | 0.551 | 0.406 | 351,109 | 837.3M | 790K |
| nf_omnifield | 0.922 | 0.906 | 0.512 | 0.445 | 269,125 | 1.79B(easy)/1.57B(hard) | 790K |

Endpoints are uninformative on their own: easy is saturated (all ≈0.91); hard is a
floor (all ≈0.50, below the classical 0.61). The signal lives in the transition.

## Phase 6 — SNR sweep (base=easy, lower SNR), mean±std over 3 seeds
test accuracy (linear probe R² of f(t) in parentheses):
| SNR dB | late | early | nf_lainr | nf_omnifield |
|---|---|---|---|---|
| 12 | 0.924±.024 (.615) | 0.898±.008 (.667) | 0.898±.027 (.886) | 0.915±.005 (.944) |
| 6  | 0.665±.134 (.529) | 0.779±.029 (.492) | **0.854±.016 (.831)** | 0.805±.017 (.902) |
| 3  | 0.486±.024 (.436) | 0.676±.045 (.429) | **0.836±.015 (.835)** | 0.788±.026 (.820) |
| 0  | 0.488±.013 (.392) | 0.526±.013 (.427) | **0.797±.015 (.811)** | 0.633±.011 (.624) |
| -3 | 0.495±.007 (.348) | 0.534±.009 (.364) | **0.714±.096 (.773)** | 0.516±.017 (.404) |
| -6 | 0.493±.021 (.329) | 0.513±.010 (.351) | 0.552±.092 (.617) | 0.488±.015 (.365) |
shuffled-pairs control: late .512, early .512, lainr .516, omni .488 (≈chance ✓)

LAINR−late accuracy gap: −.03 (12) → +.19 (6) → +.35 (3) → +.31 (0) → +.22 (−3) → +.06 (−6).
Curves coincide at both ends, separate in the middle: the predicted P1 shape.

## Phase 6 — Nonstationarity sweep (base=easy, raise trajectory nu_max), 3 seeds
| nu_max | late | early | nf_lainr | nf_omnifield |
|---|---|---|---|---|
| 0.15 | 0.922±.019 (.613) | 0.906±.022 (.615) | 0.896±.036 (.885) | 0.915±.005 (.944) |
| 0.3  | 0.762±.194 (.286) | 0.872±.038 (.495) | **0.924±.004 (.770)** | 0.759±.051 (.720) |
| 0.5  | 0.706±.154 (.159) | 0.747±.073 (.266) | **0.887±.000 (.600)** | 0.555±.055 (.564) |
| 0.8  | 0.693±.067 (.081) | 0.574±.085 (.167) | **0.797±.075 (.447)** | 0.637±.109 (.396) |

LAINR is most robust and lowest-variance; late is high-variance and degrades; early
degrades faster than LAINR.

## Phase 5 — probe diagnostic (easy; the mechanism test)
| family | fusion acc | linear R² of f(t) | MLP R² |
|---|---|---|---|
| late | 0.922 | 0.649 | 0.341 |
| early | 0.910 | 0.608 | 0.353 |
| nf_lainr | 0.891 | **0.919** | 0.667 |
| nf_omnifield | 0.922 | **0.951** | 0.691 |

Even when fusion accuracies match (~0.91), NF latents retain f(t) (R²≈0.92–0.95)
while pooled embeddings have largely discarded it (R²≈0.6). In the SNR sweep the
probe-R² tracks the accuracy collapse — the representation gap *predicts* robustness.

## Verdicts
- **P1 (advantage widens with difficulty): SUPPORTED** — SNR and nonstationarity.
- **P2 (NF ≈ early at lower fusion FLOPs): SUPPORTED with cost caveat** — ~430× cheaper
  fusion than early, but higher *total* compute (representation cost).
- **P3 (f(t) more decodable from NF latent): SUPPORTED strongly** — the mechanism.
- **P4 (reframed LAINR vs OmniField): LAINR wins** — per-modality field + agreement
  head beats the cross-modal-crosstalk field; extra machinery reduces robustness.
