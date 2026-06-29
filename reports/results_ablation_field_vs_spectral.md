# Ablation — Field interface vs. multi-band spectral basis (what causes LAINR's gap?)

LAINR's SNR-robustness headline bundles two things the main result does not separate:
**(1) the neural-field interface** (coordinate query → cross-attention to per-modality
latent tokens, Gaussian locality bias, continuous decoding in t) and **(2) a multi-band
coarse-to-fine Fourier-feature decoder** (n_bands=4, geometric 1→f_max, ~16 feats/band)
trained with a reconstruction aux (recon_weight=0.3). This ablation isolates which is
causal for the **accuracy** gap. Falsification-first: implemented faithfully, nothing
tuned toward an outcome.

## Variants (all reuse the shared FusionHead `mlp` and expose flat z; existing families untouched)
| arm | field interface? | decoder basis | isolates |
|---|---|---|---|
| `late_mbff` | **no** (CNN pool) | multi-band Fourier (same as LAINR) + recon aux | spectral basis WITHOUT the field |
| `lainr_relu` | yes (LAINR) | plain ReLU coord-MLP, matched depth (no Fourier) | field WITHOUT the multi-band basis |
| `nf_lainr_single` | yes (LAINR) | single wide Gaussian FF, matched total feats | multi-scale vs "just Fourier features" |
| `nf_lainr_nb{1,2,4,8}` | yes (LAINR) | multi-band with n_bands ∈ {1,2,4,8} | multi-scale depth (nb4 == `nf_lainr`) |
| (+ `nf_lainr` at `--recon-weight 0`) | yes | multi-band, objective OFF | basis vs reconstruction objective |

## Design choices (stated, not silently guessed)
- **Multi-band construction and the locality cross-attention modulation are verbatim copies**
  of `neural_field/lainr._LocalityAwareDecoder` (LAINR itself is unmodified), so the only
  thing that changes between LAINR and `lainr_relu`/`nf_lainr_single` is the decoder basis.
- **`lainr_relu` matched capacity:** `n_blocks = n_bands` (=4) FiLM-ReLU blocks at width
  `hidden`, input = raw coordinate (no Fourier) → exposes spectral bias. (Overall param
  count is then equalized by parameter matching below.)
- **`nf_lainr_single`:** `n_freq = n_bands*feats_per_band = 64` Gaussian features at a
  single scale `std = f_max/3` (so support ≈ [0, f_max]); one FiLM block (mirrors a band).
- **`late_mbff`:** the multi-band decoder is conditioned on the *pooled CNN latent*
  (broadcast, FiLM) and decodes the signal on its own time grid — multi-band spectral
  pressure on a pooled latent, with no coordinate cross-attention.

## Fairness (held fixed)
- **Parameter-matched across the FULL set** via `matched_plan` (assert max/min ≤ 1.6).
  On `easy` (hidden widened per arm): late h188→440,925 · early h156→453,993 ·
  nf_lainr h72→439,341 · nf_omnifield h84→442,245 · late_mbff h116→443,303 ·
  lainr_relu h72→462,669 · nf_lainr_single h80→459,189 · nf_lainr_nb1 h80→443,829 ·
  nf_lainr_nb2 h80→475,029 · nf_lainr_nb8 h64→434,565. **ratio = 1.09**.
- Identical budget/optimizer: steps 5000, n_train 1024, batch 32, n_test 256, Adam lr 1e-3,
  10% linear warmup, grad-clip 1.0, recon_weight 0.3 (except the recon=0 cell), seeds 0/1/2,
  determinism via seeded_build/enable_determinism. Same data: base=`easy`, SNR sweep
  {12,6,3,0,−3,−6}. Generator unchanged.
- Controls: shuffled-pairs (per arm, must be chance) via the sweep; unimodal C1 via
  `make validate`.

## Pre-registered predictions (recorded before the full run)
1. `late_mbff` recovers ≥ half the +0.35 LAINR−late gap at 3 dB (probe R² ~0.39→~0.86)
   ⇒ the **spectral basis** drives the accuracy gap.
2. `lainr_relu` collapses toward late-CNN (loses most of the gap; probe R² drops)
   ⇒ field-ness alone is insufficient.
3. accuracy/probe-R² rise with n_bands and saturate by ~4; single-σ underperforms the
   geometric multi-band at low SNR ⇒ the causal ingredient is **multi-scale** specifically.
4. (recon=0) LAINR probe R² falls from ~0.88 but stays above the CNN ⇒ architecture and
   objective both contribute.

## CPU sanity (1 seed, 300 steps, n_train 256, NOT param-matched) — pipeline only
*Undertrained by ~16× vs the real budget, so accuracies are at chance for all arms and
are NOT informative; shown only to confirm the wiring runs and produces signal. The probe
already differentiates field vs non-field arms, but this is a hint, not a result.*

| arm (SNR 3 dB) | test_acc | linear probe R² |
|---|---|---|
| late | 0.488 | 0.524 |
| late_mbff | 0.500 | 0.533 |
| nf_lainr | 0.508 | 0.658 |
| lainr_relu | 0.512 | 0.712 |
| nf_lainr_single | 0.555 | 0.643 |

## Full results (GPU) — to fill from the runs below
Tabulate the new arms beside the four existing families in the format of
`reports/results_chirp_param_matched.md`: test accuracy (mean±std, 3 seeds) and probe R²
(ridge + MLP) at each SNR {12,6,3,0,−3,−6}, plus encode/fuse FLOPs and params, plus the
shuffled-pairs control row.

## Run the full matrix (first GPU)
```bash
git checkout ablation/field-vs-spectral && uv sync
uv run make validate                                  # C1-C7 sanity
FAM="late early nf_lainr nf_omnifield late_mbff lainr_relu nf_lainr_single nf_lainr_nb1 nf_lainr_nb2 nf_lainr_nb8"
uv run make sweep DEVICE=cuda:0 BASE=easy KNOB=snr_db VALUES="12 6 3 0 -3 -6" SEEDS="0 1 2" \
     STEPS=5000 NTRAIN=1024 FAMILIES="$FAM"
# recon=0 de-confound cell (architecture present, objective off):
uv run python -m src.sweep.runner --base easy --knob snr_db --values 12 6 3 0 -3 -6 \
     --seeds 0 1 2 --steps 5000 --n-train 1024 --families nf_lainr --recon-weight 0 --device cuda:0
```
Outputs `reports/phase6_sweep.json` (+ `phase6_pareto.png`, `phase6_knob.png` via pareto.py).

## Verdict (fill after the GPU run)
- P1 (spectral basis drives the gap): **supported / not supported / partial** — evidence: …
- P2 (field-ness alone insufficient): … — evidence (lainr_relu): …
- P3 (multi-scale is the ingredient): … — evidence (n_bands trend + single-σ): …
- P4 (objective also contributes): … — evidence (recon=0): …
- **Bottom line:** the inductive bias is *spectral/multi-scale* vs *the field paradigm* vs *both*.
