# Gate — Real ECG+PPG fusion (BIDMC, PhysioNet)

Moves the cardiac fusion demo from simulation to **real data**: the BIDMC PPG+ECG
dataset (53 recordings, 8 min, 125 Hz; ECG lead II + PPG PLETH, simultaneous). Public,
small, conventional. Raw data is **not committed** (`data/` is gitignored); fetch with
`scripts/download_bidmc.sh` (or `make bidmc-data`).

## Pipeline (`src/real/`)
- **`bidmc.py`** — load II (ECG) + PLETH (PPG); derive ground-truth **HR(t)** from ECG
  R-peaks (scipy `find_peaks`, the reference); segment into windows (8 s, 4 s hop);
  downsample **PPG → 64 Hz** while ECG stays 125 Hz (grid mismatch); build correspondence
  pairs — positive = ECG+PPG from the **same window** (shared HR), negative = ECG and PPG
  from **different records** (independent HR). **Splits are by record** (no subject leakage).
  Optional synthetic motion-noise injection at a target SNR recreates the difficulty sweep.
- **`train_real.py`** — reuses the model registry, the budget-/parameter-matched training
  loop, and the ridge HR-probe; GPU auto-detected. ECG = modality A (sharp, FM-like),
  PPG = modality B (smooth, AM-like).

## Verified (locally, CPU)
- Loader on 16 downloaded records → 1071/357/476 train/val/test windows.
- Shapes: ECG (1000,)@125 Hz, PPG (512,)@64 Hz (grid mismatch ✓).
- **Correspondence structure on REAL data:** HR corr label-1 = 1.000, label-0 = 0.048
  (independent records → genuinely independent HR — cleaner than synthetic).
- Training pipeline runs end-to-end (build → param-match → train → accuracy + HR-probe →
  JSON). *No GPU training has been run; the smoke was 60 steps (undertrained, chance).*

## Run on the remote GPU
```bash
git pull
uv sync
bash scripts/download_bidmc.sh 53          # fetch all records (or: make bidmc-data)
uv run python -m src.real.train_real --steps 4000 --seeds 0 1 2          # clean real data
# difficulty sweep via injected motion noise (P1 on real morphology):
for s in 12 6 3 0 -3 -6; do
  uv run python -m src.real.train_real --snr $s --steps 4000 --seeds 0 1 2
done
```
Outputs `reports/real_ecg_ppg[...]_results.json` per condition (test accuracy + HR
probe-R² per family).

## What this tests
Whether the LAINR fusion advantage (P1: widening as PPG SNR drops; P3: higher HR
probe-R²) and the grid-mismatch benefit reproduce on **real cardiac signals** — the
external-validity step. Clean BIDMC has high SNR (expect all families competitive, like
synthetic `easy`); the injected-noise sweep is where the advantage should separate.
