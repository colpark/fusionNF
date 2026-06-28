.PHONY: smoke test data validate baselines train sweep report clean

PY ?= python3

# Phase 0: end-to-end scaffolding smoke on the tiny config (CPU, seconds).
smoke:
	$(PY) -m src.smoke --config configs/tiny.yaml

# Phase 0 gate: config round-trip + determinism.
test:
	$(PY) -m pytest tests/ -q

# Phase 1: render example pairs (signals + spectrograms + f(t)) to reports/.
data:
	$(PY) -m src.data.examples

# Phase 2: dataset criteria C1-C7 (easy + hard). --no-baselines skips C6/C7.
validate:
	$(PY) -m src.validation.criteria_tests

# Phase 3: oracle / wavelet-coherence / chance brackets (easy + hard).
baselines:
	$(PY) -m src.baselines.brackets

train:      ## Phase 4: the four fusion families
	@echo "Phase 4 not yet implemented"

sweep:      ## Phase 6: difficulty x family matrix -> Pareto
	@echo "Phase 6 not yet implemented"

report:     ## Phase 7: regenerate findings from saved artifacts
	@echo "Phase 7 not yet implemented"

clean:
	rm -rf runs/_test runs/tiny_* runs/rep_* runs/diff_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
