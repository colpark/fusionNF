# smFRET experiment — design, predictions, status

The cross-domain survey ranked **single-molecule FRET (photon-by-photon)** the cleanest
12/12 fit for the structural neural-field win: a *cheap comparator* (donor/acceptor
(anti)correlation timescale, C1), a *high-frequency buried* signal (fast conformational
transitions vs shot noise, C2), a *tunable intermediate-SNR* knob (photon brightness, C3),
and *irregular temporal sampling* (photon arrival times; binning is lossy, C4).

## Task
A molecule switches between low- and high-FRET states as a 2-state Markov chain with
relaxation time `tau`. Photons arrive as a Poisson process at `photon_rate` (the SNR knob);
each is routed to acceptor w.p. E(state), donor otherwise → the two channels anti-correlate
on the switching timescale. **Binary classification: FAST vs SLOW switching.** FRET levels
are fixed, so mean intensities are identical across classes — the *only* signal is the
temporal/cross-correlation structure. Default `tau` are **sub-bin** (2–4 ms vs 6–12 ms with
31 ms bins): the regime where binning aliases the dynamics and photon-by-photon analysis is
required (the reason methods like H2MM exist).

## Arms (`src/smfret/models.py`)
- `comparator` — logistic/MLP on the acceptor autocorrelation (cheap timescale read).
- `late_cnn`, `early_cnn` — CNN on **binned** intensities (grid resampling; the C4 disadvantage).
- `nf_relu` — control: naive coordinate field (ReLU-MLP over raw times, mean-pool → first-order).
- `nf_fourier` — the neural field: **periodogram + cross-spectrum of the raw photon arrival
  times** (lossless spectral ingestion, no binning). `src/smfret/models.py:SpectralField`.

## Pre-registered, falsifiable predictions
1. `nf_fourier` (lossless raw times) **beats the binned CNNs** when transitions are sub-bin,
   and the margin is largest at **intermediate** photon rates.
2. `nf_fourier` **ties the cheap comparator** at high brightness (no recovery problem left).
3. **All arms floor to chance** at very low brightness (unrecoverable) — `nf_fourier` must NOT win there.
4. `nf_relu` ≈ chance / binned — isolating the **spectral basis** as the mechanism, not "neural field" as a brand.
A *monotone* `nf_fourier` win everywhere would **falsify** the structural claim.

## Honest status (simulator smoke; 1 seed, ~600 steps — undertrained, direction only)
| photons/s | comparator | late_cnn | early_cnn | nf_relu | nf_fourier |
|---|---|---|---|---|---|
| 3000 | 0.67 | 0.92 | **0.94** | 0.50 | 0.78 |
| 1000 | 0.60 | 0.75 | **0.87** | 0.51 | 0.77 |
| 300 | 0.53 | 0.59 | **0.69** | 0.47 | 0.67 |
| 100 | ~0.50 | ~0.50 | ~0.53 | ~0.50 | ~0.51 |

- The periodogram NF is healthy (was an architecture bug when mean-pooling first-order
  Fourier features; fixed to a second-order power/cross-spectrum). The `nf_relu` control
  correctly fails (~chance), isolating the spectral basis.
- **But the 2-channel binned CNN is currently the strongest baseline, even sub-bin** — it
  learns higher-order *count statistics* that survive binning. So in this **simulated**
  instantiation the C4 advantage does NOT yet appear; `nf_fourier` ties/trails `early_cnn`.
- Per falsification-first, the simulator was **not** re-tuned to manufacture a win.

## Why this is still the right demo, and what decides it
The simulator validates the pipeline and makes the predictions testable, but the real C4
hardness lives in the **microsecond inter-photon timing** of true TTTR data, which any fixed
binning discards (and which my count-statistic-friendly simulator under-represents). The
verdict comes from:
1. **Full GPU training** (3 seeds, more steps) on the simulator — does `nf_fourier` separate
   from `early_cnn` with proper optimization?
2. **Real kinSoftChallenge data** (Zenodo 10.5281/zenodo.5701310) — the actual demonstration;
   reported either way.

## Run (remote)
```bash
git pull && uv sync
# self-contained simulator + visualizations (GPU 0):
DEVICE=cuda:0 nohup bash scripts/run_smfret_gpu.sh >/dev/null 2>&1 &   # logs/smfret_gpu.log
# real data download + visualization:
bash scripts/download_kinsoft.sh                                       # data/kinsoft/, logs/smfret_download.log
```
Outputs: `reports/smfret_sweep.json`, figures in `reports/smfret_examples/`
(`trace_fast/slow.png`, `class_signals.png`, `snr_knob.png`, and `real_sample.png` once
downloaded). The real-data **parser** (TTTR → photon streams) is the next step, pending
confirmation of the kinSoftChallenge file layout from `inspect_real`.
