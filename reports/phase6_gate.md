# Phase 6 Gate — The Sweep (headline result)

**Built:** `src/sweep/runner.py` (difficulty-knob × family × seed matrix; records test
accuracy, probe-R², and the encode/fuse FLOP split per cell; plus a **shuffled-pairs
control** and reuse of the C1 unimodal control) and `src/sweep/pareto.py` (Pareto
figures: accuracy vs **fusion** FLOPs and vs **total** FLOPs, plus accuracy and
probe-R² vs the difficulty knob with seed error bars).

Knobs supported (criterion C6): `snr_db`, `jitter`, `nonstationarity` (trajectory
nu_max), `n_components`, `rate_ratio` (B-rate / A-rate grid mismatch).

**Status:** infrastructure complete and **smoke-verified end-to-end** locally (sweep
→ JSON → Pareto/knob PNGs). The full multi-seed sweep is a **GPU job for the remote
server** — not run on this CPU box. Even the 60-step local smoke already showed the
predicted P3 ordering (LAINR probe-R² > late), which the real run will quantify.

**Controls (re-checked under sweep):** shuffled-pairs training must stay at chance
(detects leakage); unimodal classifiers stay at chance (C1). Reported in the sweep
JSON.

**How to run (remote GPU):**
```
uv run make sweep BASE=hard KNOB=snr_db VALUES="-6 -3 0 6 12" SEEDS="0 1 2" STEPS=3000 NTRAIN=1024
# repeat with KNOB=nonstationarity / rate_ratio / jitter for the full C6 study
```
Writes `reports/phase6_sweep.json`, `reports/phase6_pareto.png`, `reports/phase6_knob.png`.

**Decision needed:** run the sweep matrix on the remote GPU, then regenerate findings.
