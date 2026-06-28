# nf-fusion

Controlled experimental harness for testing the **neural-field fusion hypothesis**
on multiscale spatiotemporal signals.

The scientific question, dataset criteria, phase plan, and pre-registered
predictions (P1–P4) are documented in the project brief. This harness is built to
*falsify* the hypothesis if the data warrants it — not to make any method win.

## Status

- **Phase 0 — Scaffolding & determinism**: complete (config schema, seeding,
  run-dir convention, logging, param/FLOP accounting, smoke loop, determinism
  tests). See `reports/phase0_gate.md`.

Later phases (data generator, dataset validation C1–C7, baselines, four fusion
families, probe diagnostic, sweep, findings) are not yet implemented.

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
