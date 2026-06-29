# Diagnostics — Real ECG+PPG (BIDMC) correspondence task

The four neural families sat at **chance** on real BIDMC (clean run), with probe-R² ≈ 0.
Before touching the models, we bracketed the task. Result (16 records, n_test=600;
run `python -m src.real.diagnose`, or `make bidmc-diagnose`):

| diagnostic | value | meaning |
|---|---|---|
| **Oracle** (threshold on true HR correspondence) | **1.000** | label is well-defined by heart rate |
| **Classical** (HR estimated from RAW ECG & PPG by autocorrelation, compared) | **0.875** | **task IS solvable directly from the signals** |
| within-window HR spread (median) | 2.3 bpm | HR is ~constant within an 8 s window (low HRV) |
| negatives within 2 / 5 / 10 bpm true-HR gap | 6% / 16% / 21% | coincidental HR matches = modest irreducible label noise |
| classical HR-estimation error (median) | ECG 0.4 bpm, PPG 1.1 bpm | a trivial autocorrelation extracts HR accurately |

## Verdict (corrects the earlier read)
The earlier hypothesis — *"the real task is ill-conditioned because within-window HR is a
near-constant, overlapping scalar"* — is **refuted**. HR *is* near-constant, but negatives
rarely coincide (only 6% within 2 bpm), and a classical autocorrelation HR-extractor
**solves the task at 0.875** straight from the raw signals. Therefore:

> **The neural models' chance performance is a learnability / preprocessing failure, NOT
> a task problem.** The nets (incl. the NF arms that worked on synthetic) failed to learn
> what a 5-line autocorrelation does — extract the periodic HR and compare it — on real,
> variable-morphology signals. The probe-R²≈0 corroborates: the learned latent carries no
> HR. The classical **0.875 is now the bracket the neural methods must reach** (they are
> currently far below it, at ~0.50).

## Why the nets likely failed (to fix next, not tune)
- **No periodicity-friendly preprocessing.** Autocorrelation latches onto the dominant
  period; the raw CNN/transformer/field encoders apparently do not, and real signals carry
  baseline wander / amplitude drift the synthetic lacked. Standard **band-pass + detrend**
  (ECG ~0.5–40 Hz, PPG ~0.5–8 Hz) before normalization is the first fix to try.
- **Weak supervision for a hard extraction.** A binary same/different label is a thin
  signal to force HR extraction from real morphology; on clean synthetic, HR extraction was
  easy so the same label sufficed. An HR-aware auxiliary (or the recon aux actually helping
  HR) may be needed.

## Honest status of the synthetic→real transfer
- Synthetic (chirp + simulated cardiac): controlled, parameter-matched advantage — real.
- Real BIDMC: **the advantage has not transferred yet, but the blocker is now identified
  and is the extraction stage, not the task** (the task is solvable at 0.875). This is a
  concrete, fixable target rather than a dead end.

## Next (diagnostic-driven, report regardless of outcome)
1. Add band-pass/detrend preprocessing to the real loader; re-run the 4-family gate. Does
   any family climb toward the 0.875 classical bracket?
2. Report the classical 0.875 line alongside the neural arms in every real-data table.
3. If neural still lags after preprocessing, add an HR-extraction auxiliary and re-test.
4. (Robustness) re-run this diagnostic on all 53 records on the remote (`make bidmc-diagnose`).
