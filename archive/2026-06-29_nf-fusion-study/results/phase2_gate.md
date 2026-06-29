# Phase 2 Gate — Dataset Validation (C1–C7)  ⟵ the gate that protects everything

**Built:** `src/validation/` — `stats.py` (octave band power, summary-stat vectors,
unbiased MMD + permutation test), `unimodal.py` (high-capacity 1D-CNN single-modality
classifiers), `criteria_tests.py` (C1–C7 runner). C6/C7 use the Phase 3 baselines.

**Result: ALL of C1–C7 PASS on `easy` and `hard`.** Key numbers (base_seed=0):

| Criterion | easy | hard |
|---|---|---|
| **C1** unimodal at chance (test acc; p vs chance) | A 0.578 (p=.05), B 0.562 (p=.09) | A 0.520 (p=.29), B 0.520 (p=.29) |
| **C2** marginal invariance (MMD perm p; ≥α=.05 ⇒ ok) | A 1.000, B 0.582 | A 0.174, B 0.990 |
| **C3** measured SNR / configured; bands/octaves | A 12.00/12, 4/4; B 12.00/12, 4/4 | A −3.00/−3, 6/6; B −3.00/−3, 6/6 |
| **C4** grid mismatch + jitter | matched 512=512, jitter 0 (as cfg) | 640≠384, jitter present |
| **C5** ground-truth f(t) present/aligned | ok | ok |
| **C6** oracle flat / coherence monotone in SNR | oracle [1,1,1,1]; coh [.53,.55,.63,.70] | oracle ~[.99×4]; coh [.56,.70,.88,.96] |
| **C7** oracle ≫ classical ≫ chance | oracle 1.000, coh 0.715 | oracle 0.988, coh 0.613 |

**Interpretation / surprises:**
- **C1 is the decisive one and it holds for the right reason:** on `hard` the CNN
  hits 0.96–0.98 *train* accuracy but 0.52 *test* — capacity is present, signal is
  not. The label genuinely lives only in the joint.
- C3 caught a **real generator bug** first: `tiny` had AM carrier 24 Hz at rate
  48 Hz (exactly Nyquist) → modality B aliased to all-zeros. Fixed the config
  (carrier→18) and added a generator guard that raises if carrier ≥ Nyquist or
  ≤ f_max. `easy`/`hard` were unaffected. (Falsification-first: fixed the dataset.)
- C6 confirms difficulty is controllable: the oracle is flat across SNR (the signal
  is always there) while the classical extractor degrades smoothly — so accuracy
  drops will be extraction failures, not missing signal.

**Decision needed:** dataset is validated. Proceed to wire up Phase 4 results, or
revisit any criterion threshold?
