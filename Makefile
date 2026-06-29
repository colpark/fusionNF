.PHONY: smoke test data validate baselines train recon probe sweep report clean bidmc-data real-ecg-ppg

PY ?= python3

# GPU selection: `make train DEVICE=cuda:1` runs on the second GPU. Equivalent:
#   export NF_DEVICE=cuda:1   (honored by every entry point, incl. make targets)
#   export CUDA_VISIBLE_DEVICES=1   (OS-level; "cuda" then = physical GPU 1)
DEVICE  ?=
DEVFLAG := $(if $(strip $(DEVICE)),--device $(DEVICE),)

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
STEPS  ?= 5000
NTRAIN ?= 1024
train:
	$(PY) -m src.models.train --config $(CONFIG) --steps $(STEPS) --n-train $(NTRAIN) $(DEVFLAG)

# Phase 4: per-frequency-band reconstruction test for the NF arms.
recon:
	$(PY) -m src.models.recon_test

# Phase 5: probe diagnostic (f(t) decodability vs fusion accuracy).
probe:
	$(PY) -m src.probe.frequency_probe --config $(CONFIG) --steps $(STEPS) --n-train $(NTRAIN) $(DEVFLAG)

# Real ECG+PPG (BIDMC): download data, then train the four families on real correspondence.
N ?= 53
bidmc-data:
	bash scripts/download_bidmc.sh $(N)

real-ecg-ppg:
	$(PY) -m src.real.train_real --steps $(STEPS) --seeds 0 1 2 $(DEVFLAG)

# Phase 6: difficulty x family sweep -> Pareto + knob figures. Override BASE/KNOB/etc.
BASE   ?= hard
KNOB   ?= snr_db
VALUES ?= -6 -3 0 6 12
SEEDS  ?= 0 1 2
sweep:
	$(PY) -m src.sweep.runner --base $(BASE) --knob $(KNOB) --values $(VALUES) \
		--seeds $(SEEDS) --steps $(STEPS) --n-train $(NTRAIN) $(DEVFLAG)
	$(PY) -m src.sweep.pareto

# Phase 7: regenerate findings.md from saved artifacts (single reproduce command).
report:
	$(PY) -m src.findings

clean:
	rm -rf runs/_test runs/tiny_* runs/rep_* runs/diff_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
