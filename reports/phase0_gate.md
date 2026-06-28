# Phase 0 Gate — Scaffolding & Determinism

**Built:** repo structure (§4), dataclass config schema with lossless YAML
round-trip, global seeding + determinism control, run-dir convention (writes
config.yaml + meta.json with git SHA/seed/lib versions), JSONL+CSV run logger,
param + FLOP accounting (torch FlopCounterMode), and an end-to-end smoke loop on
the `tiny` config using a placeholder model/data stand-in. Environment managed by
**uv** (`pyproject.toml` + `uv.lock`); pinned `requirements.txt` retained.

**Acceptance criteria:**
- Smoke runs `tiny` end-to-end on CPU in seconds — **PASS** (~2 s; `make smoke`).
- Fixed-seed run bit-reproducible — **PASS** (`final_loss=0.7257770299911499`
  identical across reruns and across the system-python vs uv venv; abs_tol 1e-6).
- Config round-trip lossless; unknown keys rejected — **PASS** (`make test`, 4/4).
- Different seeds diverge — **PASS** (control that determinism isn't a constant).
- Run dir carries config + git SHA + seed — **PASS** (after initial commit, SHA
  resolves; pre-commit it correctly recorded `nogit`).

**Surprising / notes:**
- No CUDA on dev box (Apple MPS only); smoke pinned to CPU for determinism. uv on
  the remote (Linux/CUDA) server will resolve the matching torch wheel from
  `torch>=2.4` automatically.
- One `git commit` invocation hung once (no gpg/hooks configured); retried clean.
  Not reproducible.
- The smoke model/data are explicit placeholders — Phase 1 (generator) and Phase 4
  (fusion families) replace them while keeping this harness identical.

**Decision needed:** proceed to **Phase 1 (data generator)**, fix-and-rerun, or
adjust plan?
