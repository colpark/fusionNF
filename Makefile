.PHONY: smoke test data validate baselines train recon probe sweep report clean

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

# Phase 4: train the four fusion families (auto-detects GPU). Override CONFIG/STEPS/NTRAIN.
CONFIG ?= easy
STEPS  ?= 3000
NTRAIN ?= 1024
train:
	$(PY) -m src.models.train --config $(CONFIG) --steps $(STEPS) --n-train $(NTRAIN)

# Phase 4: per-frequency-band reconstruction test for the NF arms.
recon:
	$(PY) -m src.models.recon_test

# Phase 5: probe diagnostic (f(t) decodability vs fusion accuracy).
probe:
	$(PY) -m src.probe.frequency_probe --config $(CONFIG) --steps $(STEPS) --n-train $(NTRAIN)

# Phase 6: difficulty x family sweep -> Pareto + knob figures. Override BASE/KNOB/etc.
BASE   ?= hard
KNOB   ?= snr_db
VALUES ?= -6 -3 0 6 12
SEEDS  ?= 0 1 2
sweep:
	$(PY) -m src.sweep.runner --base $(BASE) --knob $(KNOB) --values $(VALUES) \
		--seeds $(SEEDS) --steps $(STEPS) --n-train $(NTRAIN)
	$(PY) -m src.sweep.pareto

# Phase 7: regenerate findings.md from saved artifacts (single reproduce command).
report:
	$(PY) -m src.findings

clean:
	rm -rf runs/_test runs/tiny_* runs/rep_* runs/diff_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
