# Neural-Field Fusion Preserves a Buried Shared Factor that Pooling Discards

## Background

Fusing time-series modalities is hard when the link between them is fine, low-energy, and
buried in a multiscale background. Three strategies compete. *Late fusion* encodes each
modality and pools to a vector — cheap, but pooling discards the fine, temporally-localized
component carrying the link. *Early fusion* cross-attends over raw tokens — faithful but
scales badly with length. *Neural-field (NF) fusion* fits a per-modality implicit neural
representation (INR) to each signal and fuses on the field latents; a reconstruction
objective should force the latent to retain the fine component, and fields queried in a
shared continuous coordinate fuse naturally across mismatched sampling grids. In a
falsification-first design we ask: does NF fusion preserve the buried shared factor better
than pooling, and reach early-fusion accuracy at lower cost?

## Method

A controlled generator emits two heterogeneous 1-D modalities — a frequency-modulated
chirp (A) and an amplitude-modulated tone (B) — that share one slowly-varying
instantaneous-frequency trajectory f(t); a pair is labeled 1 iff A and B carry the same
f(t). The matching component is low-energy within a 1/f^β multiscale background (an SNR
knob), with different sampling rates and jitter. The dataset is pre-validated: the label is
joint-only (high-capacity single-modality classifiers sit at chance), per-modality
marginals match across classes, and an oracle on f(t) is near-perfect while a classical
time-frequency baseline beats chance. Four families are compared at matched parameters,
steps, optimizer, and data, with FLOPs logged separately for representation vs fusion:
late (pooled), early (cross-attention), and two amortized neural fields — LAINR
(per-modality locality-aware multi-band INR plus a small fusion head) and OmniField
(cross-modal conditioned field). A linear probe regresses f(t) from each frozen
representation to isolate the mechanism. We sweep SNR over three seeds with shuffled-pair
and unimodal controls.

## Results

Four predictions were pre-registered. **P1 — the NF advantage over late fusion widens with
difficulty: SUPPORTED, under strict parameter matching.** As SNR falls 12→−6 dB the
accuracies coincide at the extremes and separate between; late collapses to chance by 3 dB
while LAINR holds 0.76–0.87 through 0 and −3 dB (gap up to +0.35). Given 5.7× more
parameters, late *still* collapses — the advantage is the field's inductive bias, not
capacity. **P3 — f(t) is more decodable from the NF latent: SUPPORTED.** Even where
accuracies match, linear-probe R² is ≈0.9 for NF vs ≈0.6 for pooling, and the probe gap
tracks the accuracy collapse: the difference is representational, set at encoding. **P2 —
NF reaches early-fusion accuracy at lower fusion cost: SUPPORTED.** LAINR matches or
exceeds early accuracy at ~1800× fewer fusion FLOPs and, at matched parameters, the lowest
total compute (Pareto-dominant). **P4 — the per-modality LAINR beats the cross-modal
OmniField** across the regime. Controls remain at chance (no leakage). Boundaries are
honest: at very high SNR all methods tie, and at −6 dB all fall to chance (below a
classical baseline) — the advantage is bounded to an intermediate-difficulty band.

## Conclusion

In low-SNR, nonstationary, mismatched-grid correspondence problems, reconstruction-trained
per-modality neural-field latents preserve the fine shared factor that pooling discards —
giving large, mechanistically-explained robustness at lower fusion cost, shown under strict
parameter matching with controls. The effect is specific to the per-modality field design
(not neural fields broadly) and to an intermediate-difficulty band, and the efficiency
holds on the fusion axis (representation cost rises). Practically: when a shared latent is
faint and modalities are sampled incompatibly, fuse on neural-field latents, not pooled
embeddings. The harness is falsification-first and reproducible, and the same 1-D pipeline
transfers to real heterogeneous signals such as ECG+PPG cardiac fusion.
