# Testing the Neural-Field Fusion Hypothesis on Multiscale Spatiotemporal Signals

*Extended abstract — 2026-06-29. Code @ `345c17b`; full numbers in `results/`.*

## Background

Two temporal modalities each carry a single, slowly-varying instantaneous-frequency
trajectory **f(t)**. A pair corresponds (label 1) iff the two modalities share the
*same* f(t). Three fusion strategies are in tension. **Late/mid fusion** (separate
encoders → pooled latents → fuse) is cheap but pooling tends to discard the fine,
temporally-localized component that carries the label. **Early fusion**
(cross-attention on raw tokens) preserves everything but scales badly with sequence
length. **Neural-field (NF) fusion** — the candidate — fits a per-modality implicit
neural representation (INR) to each signal and fuses on the field *latents*; the
reconstruction objective should force the latent to preserve the fine matching
component, and querying fields in a shared continuous coordinate space should make
fusion natural even across mismatched sampling grids.

We pre-registered four predictions: **P1** NF's accuracy advantage over late fusion
*widens* as SNR drops and f(t) becomes more nonstationary; **P2** NF approaches
early-fusion accuracy at far lower *fusion* compute; **P3** f(t) is more decodable
(higher probe R²) from the NF latent than from the pooled embedding; **P4** behaviour
varies along the amortization spectrum. The harness is built to *falsify*.

## Method — the hypotheses are encoded in the dataset, not just the models

The central design principle: **every claim is operationalized as a generative knob
or invariant, and verified before any model is trained.** A synthetic generator draws
f(t) from a fixed prior and renders it through two *different* observation models —
modality A as an FM chirp, modality B as amplitude modulation — added to a multiscale
1/fᵝ background across S octaves, sampled on *different* rates with timing jitter.

| Design choice | Hypothesis it operationalizes | Verified by |
|---|---|---|
| **Label = correspondence**; each modality's own f(t) is *always* a fresh prior draw, so only the *coupling* differs | Forces genuine **fusion**: I(label;A)=I(label;B)=0 but I(label;A,B)>0 — no unimodal shortcut can "win" | **C1** unimodal classifiers at chance; **C2** per-modality marginals identical across classes (MMD) |
| **Low-energy matching component buried in 1/fᵝ background**, with an explicit **SNR knob** | The regime where pooling *should* fail — "fine, temporally-localized component" → **P1** (SNR axis) | **C3** band-power + measured SNR matches config |
| **Trajectory complexity knob** (nonstationarity = max \|df/dt\|) | Pooling loses more as f(t) becomes less summarizable → **P1** (nonstationarity axis) | **C6** difficulty monotonic for the extractor, flat for oracle |
| **Different observation models on different grids + jitter** | The shared-continuous-coordinate advantage of fields over grid-bound encoders | **C4** grids mismatch; jitter present |
| **Ground-truth f(t) saved per sample** | Enables the probe that *factors* the hypothesis → **P3** (does the representation keep f(t)?) | **C5** trajectories present & aligned |
| **Oracle (sees f) + classical wavelet-coherence baseline** | Establishes the achievable ceiling and a non-neural floor, so failures are attributable | **C7** oracle≫classical≫chance |

All of C1–C7 passed on both the easy and hard configs *before* modelling, so any
downstream win cannot be a dataset artifact. Four fusion families (late, early, and
two amortized neural fields — **LAINR**, locality-aware per-modality INR, and
**OmniField**, a cross-modal-crosstalk conditioned field) were trained at **matched
budget** (params/steps/optimizer/data), logging FLOPs **separately for representation
and fusion**. A linear/MLP **probe** regresses f(t) from each frozen representation.

## Results

- **P1 — supported, both axes.** Sweeping SNR from the easy baseline, the four
  accuracy curves coincide at the extremes (all ≈0.91 at 12 dB; all ≈0.50 at −6 dB)
  and **separate in the middle**. The LAINR−late gap runs −0.03 → +0.19 → **+0.35**
  (at 3 dB) → +0.31 → +0.22 → +0.06. Late fusion hits chance by 3 dB while LAINR
  holds 0.71–0.85 through 0/−3 dB. The nonstationarity sweep shows the same ordering
  (LAINR 0.92→0.80 vs late 0.92→0.69, with late high-variance and LAINR near-zero
  variance).
- **P3 — supported; the mechanism.** Even on *easy*, where fusion accuracies match
  (~0.91), f(t) is far more decodable from NF latents (linear R² 0.92–0.95) than from
  pooled late/early embeddings (0.61–0.65). Across the SNR sweep, each method's
  probe-R² *tracks* its accuracy collapse — the representation gap **predicts** the
  robustness gap. Late fusion fails at low SNR because its representation never held
  the matching component.
- **P2 — supported, with a cost caveat.** LAINR matches early-fusion accuracy at
  **791K fusion FLOPs vs 342M** (~430× cheaper fusion), but NF pays in *representation*
  FLOPs, so on **total** compute it is ~2× the others. The efficiency claim holds on
  the fusion axis, not total compute.
- **P4 — LAINR wins.** The simpler per-modality field + agreement head beats the
  heavier cross-modal-crosstalk field everywhere off the easy endpoint (e.g. 0 dB:
  0.797 vs 0.633). Extra cross-modal machinery reduced robustness here.
- **Controls hold.** Shuffled-pairs training stays at chance (0.49–0.52) for all four
  families; combined with C1 this confirms the accuracies are genuine joint inference.

## Discussion

In this regime the hypothesis is **supported, with a clear and mechanistically
explained winner**: reconstruction-trained per-modality neural-field latents (LAINR)
preserve the fine matching factor that pooling discards, yielding large robustness as
SNR drops and f(t) becomes nonstationary, at far lower *fusion* cost than
cross-attention. The probe makes the causal story concrete — it is the *representation
stage*, not the fusion head, that separates the methods.

Three honest boundaries. (i) The advantage is **regime-bounded**: it vanishes where
the task is trivial (high SNR, all win) and where it is impossible (very low SNR, all
fail at the floor, *below* the classical 0.61 anchor — deep fusion does not beat a
model-aware demodulator there). (ii) It is **LAINR-specific**, not neural-fields in
general — OmniField, a SOTA-style multimodal field, was mid-pack. (iii) The efficiency
is **axis-specific** (fusion FLOPs, not total compute), the canonical amortization
tension: a faithful field moves cost from fusion to field-fitting. External validity to
real (non-synthetic) multimodal signals is untested, and the auto-decoded end of the
amortization spectrum was not measured. The decisive evidence came not from any single
accuracy number but from the *shape* of degradation across the difficulty knobs that
the dataset was designed to expose.
