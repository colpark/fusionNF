# RR task fix — rate-aware read-out + fair (mean) bracket

Driven by a 5-agent cross-check of why every neural arm sat at the predict-mean floor on
real ECG+PPG respiratory-rate (RR) regression. Two changes, both diagnostic-led:

## 1. Rate-aware read-out (`spec_*` arms)
**Diagnosis (cross-checked, agents 2 & 5):** the fusion arms read out a **temporal mean**
(`late` mean-pool; LAINR `z = tokens.mean`), and **a temporal mean cannot encode a
frequency — but RR *is* a frequency.** Tokenization to ~1 token/s (Nyquist 0.5 Hz) also
aliases the top of the respiratory band. LAINR's locality/INR decoder feeds only
`recon_loss`, never `fuse` — explaining why NF underperformed plain late fusion.

**Fix:** new arms `spec_ecg`, `spec_ppg`, `spec_fuse` (`src/real/bidmc_rr.py:_SpectralEncoder`/
`SpectralReg`). Each takes the band-limited **log-magnitude spectrum** of the window
(`|rFFT|`, freqs ≤ 4 Hz) → MLP → scalar RR. Collapsing in the spectral domain preserves
frequency (unlike mean-pooling the signal), so respiratory rate is directly readable.
`spec_fuse` concatenates both modalities' spectra (the rate-aware fusion arm).

## 2. Fair bracket: report MEAN MAE, not just median
**Diagnosis (agents 3 & 4):** the classical bracket reported **median** MAE while neural
arms reported **mean** MAE — not apples-to-apples. `classical_bracket` now returns both.

## Local check (16 records, 1 seed, 1200 steps — undertrained; direction only)
| arm | RR MAE (bpm) |
|---|---|
| predict-mean floor | 2.59 |
| classical PPG (median) | 0.67 |
| **classical PPG (MEAN, fair)** | **2.81** |
| classical ECG (mean/median) | 7.09 / 6.35 |
| late (rate-blind) | 5.22 |
| uni_ppg (rate-blind) | 4.21 |
| nf_lainr (rate-blind) | 3.73 |
| **spec_ppg (rate-aware)** | **2.71** |
| spec_fuse (rate-aware) | 3.38 |

**Two takeaways, both honest:**
1. The rate-aware read-out roughly halves the error of the rate-blind arms (2.71 vs 4–5),
   confirming the pooling read-out was a genuine bottleneck.
2. On a **fair mean basis the classical bracket is 2.81 ≈ the 2.59 floor** — the earlier
   "DSP beats deep nets ~2.5×" was largely a **median-vs-mean artifact**. The real RR task
   has **little learnable headroom** above predicting the mean (small per-subject RR
   variance; ±1–2 bpm target quantization; ~31 independent subjects). See the 5-agent
   findings: effective N ~31 subjects, target noise caps MAE near ~1–2 bpm.

## Status / next
- GPU re-run (53 records, 3 seeds, 4000 steps) via `scripts/run_rr_gpu.sh` (now includes
  the `spec_*` arms) will give the real numbers; then the PPG-degradation sweep
  (`scripts/run_rr_degrade_gpu.sh`) tests whether `spec_fuse` holds RR via buried ECG as
  PPG is corrupted — the actual buried-factor test.
- Caveat: even with a rate-aware read-out, the headroom is small; a clean separation may
  require a task with larger RR variance (e.g. longer windows, or pooling subjects with
  diverse respiratory rates).
