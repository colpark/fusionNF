#!/usr/bin/env bash
# RR estimation under PROGRESSIVE PPG DEGRADATION (GPU 1) -- the buried-factor test.
# As PPG is corrupted, PPG-alone RR collapses; can NF fusion hold by using the buried
# ECG respiration? Band-pass preprocessing on. Survives logout (nohup/tmux).
# Logs -> logs/rr_degrade_gpu.log ; outputs reports/real_rr_pp[_ppgsnrX]_results.json
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:1}"
mkdir -p logs reports data/bidmc
LOG="logs/rr_degrade_gpu.log"
FAM="uni_ecg uni_ppg late early nf_lainr nf_omnifield spec_ecg spec_ppg spec_fuse"
{
  echo "==== [$(date)] RR-degradation START  DEVICE=$DEVICE ===="
  if ! ls data/bidmc/*_Signals.csv >/dev/null 2>&1; then
    echo "[$(date)] downloading BIDMC..."; bash scripts/download_bidmc.sh 53
  fi
  echo "[$(date)] clean PPG (reference)"
  uv run python -m src.real.bidmc_rr --families $FAM --preprocess --steps 4000 --seeds 0 1 2 --device "$DEVICE"
  for s in 6 0 -6 -12; do
    echo "[$(date)] PPG SNR = $s dB"
    uv run python -m src.real.bidmc_rr --families $FAM --preprocess --ppg-snr "$s" \
         --steps 4000 --seeds 0 1 2 --device "$DEVICE"
  done
  echo "==== [$(date)] RR-degradation DONE -> reports/real_rr_pp*_results.json ===="
} 2>&1 | tee -a "$LOG"
