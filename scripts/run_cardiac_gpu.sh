#!/usr/bin/env bash
# Real ECG+PPG (BIDMC) cardiac fusion on GPU 1 (override: DEVICE=cuda:N).
# Survives logout when launched with nohup/tmux (see bottom). Logs -> logs/cardiac_gpu.log
# Outputs: reports/real_ecg_ppg_results.json (clean) + reports/real_ecg_ppg_snr<S>_results.json
set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root
DEVICE="${DEVICE:-cuda:1}"
mkdir -p logs reports data/bidmc
LOG="logs/cardiac_gpu.log"
{
  echo "==== [$(date)] cardiac START  DEVICE=$DEVICE ===="
  if ! ls data/bidmc/*_Signals.csv >/dev/null 2>&1; then
    echo "[$(date)] downloading BIDMC..."; bash scripts/download_bidmc.sh 53
  fi
  echo "==== [$(date)] clean real run ===="
  uv run python -m src.real.train_real --steps 4000 --seeds 0 1 2 --device "$DEVICE"
  echo "==== [$(date)] injected-motion-noise SNR sweep (P1 on real morphology) ===="
  for s in 12 6 3 0 -3 -6; do
    echo "[$(date)] snr=$s"
    uv run python -m src.real.train_real --snr "$s" --steps 4000 --seeds 0 1 2 --device "$DEVICE"
  done
  echo "==== [$(date)] cardiac DONE -> reports/real_ecg_ppg*_results.json ===="
} 2>&1 | tee -a "$LOG"
