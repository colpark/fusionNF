#!/usr/bin/env bash
# Real respiratory-rate (RR) estimation from ECG+PPG on BIDMC, GPU 1 (override DEVICE=).
# Survives logout via nohup/tmux (see bottom). Logs -> logs/rr_gpu.log
# Output: reports/real_rr_results.json  (unimodal vs fusion RR MAE + classical bracket)
set -uo pipefail
export PYTHONUNBUFFERED=1   # flush prints through tee/nohup (else log looks stalled)
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:1}"
mkdir -p logs reports data/bidmc
LOG="logs/rr_gpu.log"
{
  echo "==== [$(date)] RR START  DEVICE=$DEVICE ===="
  if ! ls data/bidmc/*_Signals.csv >/dev/null 2>&1; then
    echo "[$(date)] downloading BIDMC..."; bash scripts/download_bidmc.sh 53
  fi
  uv run python -m src.real.bidmc_rr \
       --families uni_ecg uni_ppg late early nf_lainr nf_omnifield \
       --steps 4000 --seeds 0 1 2 --device "$DEVICE"
  echo "==== [$(date)] RR DONE -> reports/real_rr_results.json ===="
} 2>&1 | tee -a "$LOG"
