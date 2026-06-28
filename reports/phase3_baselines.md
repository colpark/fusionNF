# Phase 3 -- Baselines & Brackets

Correspondence accuracy bracketed by a trivial floor (chance), a classical signal-processing anchor (FM/AM demodulation + coherence, no ground truth), and a ground-truth ceiling (oracle).

- Base seed: `0`
- Samples per stage: capped at `256` (train for threshold fit, test for eval)
- Threshold fit on the train split, evaluated on the disjoint test split.

| Tag | n | Chance | Classical | Oracle |
|-----|---|--------|-----------|--------|
| easy | 128 | 0.562 | 0.688 | 1.000 |
| hard | 256 | 0.516 | 0.613 | 0.988 |

## Interpretation

- **Chance** ~ 0.5 confirms balanced classes; nothing is learnable for free.
- **Oracle** ~ 1.0 confirms the matching signal is fully present: if a model could recover the latent trajectories, the task is solved.
- **Classical** sits between the two. On `easy` it clears chance comfortably, showing hand-built demodulation already extracts real cross-modal structure. On `hard` (low SNR, timing jitter, mismatched rates, richer trajectories) the classical anchor degrades -- this is the gap a learned fusion model must close.
