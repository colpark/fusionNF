# Simulated Results — Consolidated (Neural-Field Fusion Study)

_Last updated 2026-06-30. Scope: all **simulated** experiments. Real-data results (BIDMC HR
correspondence, respiratory-rate estimation, diagnostics) live in `real_ecg_ppg_gate.md`,
`real_ecg_ppg_diagnostics.md`, `real_rr_fix.md`. Falsification-first throughout: nothing was
tuned toward a desired outcome; negatives are reported plainly._

## TL;DR — the one-paragraph state of knowledge
On a controlled synthetic correspondence task, a reconstruction-trained neural-field arm
(LAINR) preserves a fine, low-energy, buried shared factor f(t) far better than pooled
(late) or cross-attention (early) fusion — **parameter-matched**, with the advantage widening
as SNR drops through an intermediate band, and **mechanistically** explained by a frozen-z
probe. The field-vs-spectral **ablation** then decomposes *why*: the **reconstruction
auxiliary** drives robust accuracy, the **field interface** drives the linearly-decodable
representation (the probe), and the **multi-band Fourier basis is largely incidental**. The
same theme recurs everywhere we look (cardiac sim, smFRET sim, real RR): the
spectral/reconstruction machinery carries accuracy; the field's distinctive contribution is
representational, not raw accuracy.

---

## 1. Synthetic chirp correspondence (canonical, parameter-matched)
FM-chirp(A)/AM-tone(B), buried shared factor f(t), multiscale 1/fᵝ background, grid mismatch
+ jitter, joint-only label (validated C1–C7). 4 families with a shared FusionHead, **matched
to ~351K params** (late/early widened 4.6–5.7×), steps 5000, n_train 1024, 3 seeds, CUDA.
**test accuracy (frozen-z probe R²)**:

| SNR | late | early | nf_lainr | nf_omnifield |
|---|---|---|---|---|
| 12 dB | 0.835 (.61) | 0.913 (.61) | 0.910 (.88) | 0.859 (.90) |
| 6 dB | 0.553 (.48) | 0.803 (.52) | **0.868 (.83)** | 0.819 (.88) |
| 3 dB | 0.492 (.39) | 0.596 (.49) | **0.842 (.86)** | 0.715 (.78) |
| 0 dB | 0.504 (.37) | 0.552 (.43) | **0.806 (.84)** | 0.608 (.62) |
| −3 dB | 0.496 (.33) | 0.505 (.38) | **0.758 (.75)** | 0.536 (.40) |
| −6 dB | 0.487 (.32) | 0.521 (.35) | 0.547 (.56) | 0.530 (.39) |

Shuffled-pairs controls 0.49–0.52 (chance). **P1** (advantage widens with difficulty):
late collapses to chance by 3 dB *even at 5.7× params*; LAINR holds 0.76–0.84 through 0/−3 dB;
gap peaks **+0.35** at 3 dB → not a capacity artifact, an inductive-bias effect. **P3**
(mechanism): probe R² ≈0.84–0.88 for LAINR vs ≈0.39–0.49 for pooled, tracking the accuracy
collapse → the difference is **representational**, set at encoding (the FusionHead is shared).
**P2**: LAINR ≈/> early accuracy at far lower *fusion* FLOPs and Pareto-dominant total compute
under matching. **P4** (reframed LAINR vs OmniField): LAINR > OmniField on chirp.
**Boundaries:** at 12 dB everyone ties; at −6 dB everyone floors below the classical demod
anchor → the advantage is **regime-bounded** (intermediate SNR) and LAINR-specific, not NF-generic.

## 2. Simulated cardiac correspondence (`ecg_ppg`)
Synthetic ECG (sharp P-QRS-T) + PPG (smooth pulse + respiratory AM), shared HR latent;
parameter-matched, 3 seeds. SNR sweep, **test accuracy (probe R²)**:

| SNR | late | early | nf_lainr | nf_omnifield |
|---|---|---|---|---|
| 12 dB | 0.661 (.30) | 0.853 (.05) | 0.848 (.59) | **0.928 (.38)** |
| 6 dB | 0.590 (.16) | 0.706 (.00) | 0.763 (.54) | **0.915 (.36)** |
| 3 dB | 0.547 (.01) | 0.500 (−.02) | 0.751 (.42) | **0.900 (.35)** |

(Hard gate: all four ~chance — a floor, like chirp-hard.) **Reproduces the core P1 effect on
a different signal family** (quasi-periodic pulses vs FM/AM): as SNR drops, the NF arms hold
while late/early collapse, parameter-matched. **Two honest wrinkles:** (a) the best NF arm is
**dataset-dependent** — OmniField dominates here (was mid-pack on chirp), LAINR is robust on
both; (b) a **probe↔accuracy dissociation** appears — OmniField has top accuracy but only
mid probe R², so the clean chirp "probe explains accuracy" link is murkier on cardiac.
**Caveat:** HR is the *dominant* feature here, not a buried one — so this is a weaker test of
the buried-factor claim than the chirp.

## 3. Field-vs-spectral ablation (the decomposition) — easy chirp, matched ~434K, 3 seeds
Arms isolate the ingredients: `late_mbff` = CNN + multi-band Fourier **recon**, *no field*;
`lainr_relu` = field + locality cross-attn with a ReLU decoder, *no Fourier basis*;
`nf_lainr_single`/`nb{1,2,8}` vary the multi-scale basis. **test accuracy (probe R²)**:

| arm | 12 dB | 6 dB | 3 dB |
|---|---|---|---|
| late | 0.863 (.60) | 0.624 (.49) | 0.495 (.43) |
| early | 0.866 (.71) | 0.784 (.56) | 0.616 (.53) |
| nf_lainr | 0.895 (.90) | 0.854 (.85) | **0.837 (.89)** |
| nf_omnifield | 0.882 (.91) | 0.805 (.86) | 0.730 (.78) |
| **late_mbff** (spectral, no field) | **0.927 (.61)** | **0.883 (.59)** | _pending_ |
| **lainr_relu** (field, no Fourier) | **0.943 (.96)** | 0.845 (.89) | _pending_ |
| nf_lainr_single | 0.917 (.89) | 0.867 (.87) | _pending_ |
| nf_lainr_nb1 / nb2 / nb8 | 0.895 (.84) / 0.896 (.87) / 0.914 (.86) | 0.840 / 0.867 / 0.857 | _pending_ |

**recon=0 de-confound** (nf_lainr, recon weight 0, matched 351K; vs canonical 351K+recon):
| SNR | 12 | 6 | 3 | 0 | −3 | −6 |
|---|---|---|---|---|---|---|
| recon=0 acc (probe) | 0.763 (.89) | 0.635 (.74) | 0.514 (.59) | 0.488 (.33) | 0.507 (.47) | 0.517 (.39) |
| canonical +recon acc | 0.910 | 0.868 | 0.842 | 0.806 | 0.758 | 0.547 |

Controls all ~chance (0.44–0.53). **Three de-confounded findings:**
1. **Reconstruction auxiliary → robust accuracy.** Adding multi-band recon to a plain CNN
   (`late`→`late_mbff`) jumps accuracy 0.863→0.927 (12 dB) with *no* probe change; removing
   recon from LAINR **collapses it to chance by 3 dB** (0.842→0.514) at matched size. Recon,
   not the field, is the dominant accuracy/robustness driver.
2. **Field interface → linear decodability.** Probe R² is high *only* for field arms
   (0.84–0.96) and low for non-field arms (late 0.60, early 0.71, late_mbff 0.61) — even
   though late_mbff has high accuracy. The field, even with a ReLU decoder (`lainr_relu`,
   probe 0.96), drives the probe mechanism.
3. **Multi-band Fourier basis → largely incidental.** `lainr_relu` (no Fourier) is the *best*
   arm at 12 dB (0.943); the `n_bands` sweep moves probe only modestly (nb1 .84 < nb2 .87 < nb4 .90).

**Decisive comparison still pending:** `late_mbff` and `lainr_relu` at **3 / 0 / −3 / −6 dB**.
If `late_mbff` (CNN+recon, no field) also holds at 3 dB like full LAINR → the field is
unnecessary even for low-SNR robustness. If it collapses while LAINR holds → the field adds a
distinct robustness edge at the hardest SNR. (Verdict to be locked in
`results_ablation_field_vs_spectral.md` once those rows are read from
`reports/phase6_sweep_ablation.json`.)

## 4. smFRET kinetics simulator (the killer-demo domain, in progress)
Photon-by-photon 2-state FRET; FAST vs SLOW switching at **sub-bin** timescales (2–4 vs
6–12 ms, 31 ms bins); photon brightness = SNR knob. Arms: cheap comparator, binned CNN
(late/early), `nf_relu` (naive coordinate field, control), `nf_fourier` (periodogram +
cross-spectrum of raw arrival times). **Smoke (1 seed, ~600 steps — direction only):**

| photons/s | comparator | late_cnn | early_cnn | nf_relu | nf_fourier |
|---|---|---|---|---|---|
| 3000 | 0.67 | 0.92 | **0.94** | 0.50 | 0.78 |
| 1000 | 0.60 | 0.75 | **0.87** | 0.51 | 0.77 |
| 300 | 0.53 | 0.59 | **0.69** | 0.47 | 0.67 |
| 100 | ~chance everywhere | | | | |

- Fixed an architecture bug (mean-pooled first-order Fourier → second-order periodogram);
  `nf_fourier` then jumped from 0.56→0.94 (clean). The `nf_relu` control fails (~chance) as
  predicted — isolating the spectral basis.
- **But the 2-channel binned CNN is currently the strongest baseline, even sub-bin** — it
  learns count statistics that survive binning. So in this *simulated* instantiation the C4
  advantage does **not** yet appear; the simulator was **not** re-tuned to manufacture a win.
- Verdict deferred to full GPU training (3 seeds) and the **real** kinSoftChallenge data,
  where the microsecond inter-photon timing (which any binning discards, and which this
  count-statistic-friendly simulator under-represents) actually lives. See `smfret_plan.md`.

---

## 5. Cross-cutting insights (latest)

**The recurring theme.** Across chirp, cardiac, the ablation, real RR, and smFRET: the
**spectral/reconstruction machinery carries the accuracy**, while the **field's distinctive
contribution is a linearly-decodable representation (the probe), not raw accuracy.** A plain
CNN with a multi-band reconstruction head (`late_mbff`) matches/beats the field on accuracy;
on real RR a spectral read-out (not the field) helped; on smFRET a binned CNN is competitive.
"Neural field wins" is more precisely **"reconstruction objective + (for decodability) a
coordinate field; the Fourier basis is mostly incidental."**

**When the field should win *structurally* (the four-condition rubric).** From the cross-domain
survey, the NF advantage is sharp only when accuracy is bottlenecked by *per-modality recovery
of a cross-scale signal*, not by the cross-modal interaction — i.e. all four hold:
C1 cheap comparator on the recovered factors (correlation/distance) so there's nothing for
early/mid fusion to model; C2 discriminative signal in a high-frequency / multi-scale band
that strided-conv / patch / ReLU encoders attenuate; C3 intermediate (recoverable-but-hard)
SNR; C4 irregular / multi-rate sampling that grid encoders must resample. The synthetic chirp
satisfies C1–C3 but is uniformly gridded (C4=0) → it is the **control**, which is exactly why
the win there is real but the strongest real-world fit is photon-timing data (smFRET).

**The trap (why some tasks fail by construction).** The advantage collapses when the signal
is low-frequency **or** the cross-modal coupling is itself the hard problem. Respiratory-rate
from ECG/PPG fails the C2 half (the factor is 0.1–0.6 Hz, recovered by a one-line bandpass);
EEG-fMRI fails the C1 half (the coupling *is* the modeling target). A single tempting
condition is not enough — the conjunction is the point.

## What is and isn't established (simulated)
- **Established:** a mechanistically-explained, parameter-matched, controlled advantage for
  reconstruction-trained NF fusion in an intermediate-SNR band, reproduced across two
  synthetic signal families; and a de-confounding ablation attributing accuracy to the
  reconstruction auxiliary and decodability to the field, with the Fourier basis incidental.
- **Not yet established:** whether the **field** adds robustness beyond recon at the *lowest*
  SNR (pending the `late_mbff`/`lainr_relu` low-SNR rows); and whether any of this transfers
  to a real C4 domain (smFRET real data, in progress). The clean synthetic win is real but
  regime-bounded; the real-world payoff is unproven.

## Pointers
Canonical chirp: `results_chirp_param_matched.{md,json}` · methods: `METHODS.md` ·
ablation (to be finalized): `results_ablation_field_vs_spectral.md` · smFRET: `smfret_plan.md`
· figures: `smfret_examples/`, `ecg_ppg_examples/` · real data: `real_rr_fix.md`,
`real_ecg_ppg_diagnostics.md`.
