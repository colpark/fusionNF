# Secure snapshot — Neural-Field Fusion Study (2026-06-29)

Self-contained capsule of the code and results behind the extended abstract.

```
EXTENDED_ABSTRACT.md   one-pager: background / method / results / discussion
                       (special focus: how each hypothesis is encoded in the dataset)
results/
  RESULTS.md           all tables (bracket, Phase 4, SNR & nonstationarity sweeps, probe)
  all_results.json     machine-readable numbers + pre-registered verdicts
  phase*_gate.md       per-phase gate reports
  phase3_baselines.md  chance / classical / oracle bracket
code/                  frozen source (src/, configs/, Makefile, pyproject.toml, uv.lock)
CODE_SHA.txt           git commit the results were produced at (345c17b)
```

**Provenance.** Numbers are from CUDA runs at commit `345c17b`, budget-matched
(steps=5000, n_train=1024, Adam lr=1e-3 + warmup + grad-clip, recon-weight 0.3, 3 seeds
for sweeps). Controls (shuffled-pairs, unimodal C1) held at chance throughout.

**Reproduce** (from the live repo at this SHA, on a CUDA box):
```bash
git checkout 345c17b && uv sync
uv run make validate
uv run make train CONFIG=easy && uv run make train CONFIG=hard
uv run make sweep BASE=easy KNOB=snr_db VALUES="12 6 3 0 -3 -6" SEEDS="0 1 2"
uv run make sweep BASE=easy KNOB=nonstationarity VALUES="0.15 0.3 0.5 0.8" SEEDS="0 1 2"
uv run make probe CONFIG=easy
uv run make report
```
Figures (`phase6_pareto.png`, `phase6_knob.png`) regenerate via `make sweep` / 
`python -m src.sweep.pareto` from the sweep JSON.

**One-line result.** The NF-fusion hypothesis is supported in the intermediate-difficulty
regime: LAINR-style per-modality field latents preserve f(t) (probe R²≈0.92 vs ≈0.65 for
pooled) and degrade far more gracefully than late/early fusion as SNR drops and f(t)
becomes nonstationary, at ~430× lower fusion FLOPs than early fusion — though at higher
total compute, and not uniformly across all neural-field designs (OmniField was mid-pack).
