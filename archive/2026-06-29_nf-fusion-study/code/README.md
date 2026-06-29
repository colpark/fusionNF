# nf-fusion

Controlled experimental harness for testing the **neural-field fusion hypothesis**
on multiscale spatiotemporal signals.

The scientific question, dataset criteria, phase plan, and pre-registered
predictions (P1–P4) are documented in the project brief. This harness is built to
*falsify* the hypothesis if the data warrants it — not to make any method win.

## Status

All phases built. Gate reports in `reports/phase*_gate.md`.

- **Phase 0** scaffolding & determinism — complete.
- **Phase 1** data generator — complete (gate passed).
- **Phase 2** dataset validation C1–C7 — complete (all pass on easy + hard).
- **Phase 3** baselines & bracket — complete (oracle 1.0/0.99, classical 0.72/0.61,
  chance ~0.5).
- **Phase 4** four fusion families (`late`, `early`, `nf_lainr`, `nf_omnifield`) —
  built & verified learnable; final gate numbers come from the remote GPU run.
- **Phase 5** probe diagnostic — built.
- **Phase 6** sweep + Pareto — built; the heavy matrix is a remote GPU job.
- **Phase 7** findings generator — built (`make report`).

The two neural-field arms are amortized real architectures: **LAINR** (locality-aware
generalizable INR) and **OmniField** (conditioned multimodal neural field). The
auto-decoded/functa arm was dropped by design; P4 is reframed as LAINR vs OmniField.

## Running the scientific pipeline (remote GPU)

`make` targets auto-detect CUDA. Heavy training belongs on the GPU server:

```bash
uv sync                       # one-time env setup
uv run make data              # Phase 1: example figures
uv run make validate          # Phase 2: dataset criteria C1–C7
uv run make baselines         # Phase 3: chance / classical / oracle bracket
uv run make train             # Phase 4: train 4 families -> reports/phase4_results.json
uv run make recon             # Phase 4: NF per-band reconstruction
uv run make probe             # Phase 5: f(t) decodability vs fusion accuracy
uv run make sweep             # Phase 6: difficulty x family -> Pareto + knob figures
uv run make report            # Phase 7: regenerate findings.md from artifacts
```

Tunable knobs (all have defaults): `CONFIG`, `STEPS`, `NTRAIN` for training;
`BASE`, `KNOB`, `VALUES`, `SEEDS` for the sweep. Example full sweep:

```bash
uv run make sweep BASE=hard KNOB=snr_db VALUES="-6 -3 0 6 12" SEEDS="0 1 2" STEPS=3000 NTRAIN=1024
uv run make sweep BASE=hard KNOB=rate_ratio VALUES="0.4 0.6 0.8 1.0" SEEDS="0 1 2"
uv run make report
```

## Environment (uv)

```bash
uv sync                 # create .venv and install pinned deps
uv run make smoke       # Phase 0 end-to-end smoke on the tiny config
uv run make test        # config round-trip + determinism gate
```

Or with an activated venv:

```bash
source .venv/bin/activate
make smoke
make test
```

## Layout

```
configs/      tiny.yaml, easy.yaml, hard.yaml   (difficulty presets)
src/
  config.py   dataclass config schema + YAML round-trip
  smoke.py    Phase-0 end-to-end harness (placeholder model/data)
  data/       generator (Phase 1)
  validation/ dataset criteria tests C1–C7 (Phase 2)
  baselines/  oracle, wavelet coherence (Phase 3)
  models/     late / early / neural_field fusion families (Phase 4)
  probe/      frequency-trajectory probe (Phase 5)
  sweep/      difficulty × family matrix → Pareto (Phase 6)
  utils/      seeding, rundir, logging, flops
tests/        smoke + determinism
reports/      gate reports, figures, findings.md
```

## Operating rules

Phase-gated; falsification-first; fully seeded and reproducible; budget-matched
comparisons with FLOPs logged separately for representation vs fusion; splits by
generative seed (no leakage). Each run writes its resolved config, git SHA, and
seed to its output directory.
