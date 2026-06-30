#!/usr/bin/env bash
# smFRET kinetics experiment on GPU (override DEVICE=cuda:N). Self-contained simulator,
# so it runs without any download. Survives logout via nohup/tmux.
# Logs -> logs/smfret_gpu.log ; outputs reports/smfret_sweep.json + reports/smfret_examples/*.png
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"
mkdir -p logs reports
{
  echo "==== [$(date)] smFRET START  DEVICE=$DEVICE ===="
  echo "[$(date)] generating data visualizations -> reports/smfret_examples/"
  uv run python -m src.smfret.viz
  echo "[$(date)] brightness (SNR) sweep across arms"
  uv run python -m src.smfret.run --rates 1000 300 100 50 20 10 \
       --n-train 2000 --n-test 1000 --steps 1500 --seeds 0 1 2 --device "$DEVICE"
  echo "==== [$(date)] smFRET DONE -> reports/smfret_sweep.json ===="
} 2>&1 | tee -a logs/smfret_gpu.log
