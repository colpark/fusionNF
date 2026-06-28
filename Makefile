.PHONY: smoke test validate baselines train sweep report clean

PY ?= python3

# Phase 0: end-to-end scaffolding smoke on the tiny config (CPU, seconds).
smoke:
	$(PY) -m src.smoke --config configs/tiny.yaml

# Phase 0 gate: config round-trip + determinism.
test:
	$(PY) -m pytest tests/ -q

# Placeholders wired up in later phases:
validate:   ## Phase 2: dataset criteria C1-C7
	@echo "Phase 2 not yet implemented"

baselines:  ## Phase 3: oracle / wavelet-coherence / chance brackets
	@echo "Phase 3 not yet implemented"

train:      ## Phase 4: the four fusion families
	@echo "Phase 4 not yet implemented"

sweep:      ## Phase 6: difficulty x family matrix -> Pareto
	@echo "Phase 6 not yet implemented"

report:     ## Phase 7: regenerate findings from saved artifacts
	@echo "Phase 7 not yet implemented"

clean:
	rm -rf runs/_test runs/tiny_* runs/rep_* runs/diff_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
