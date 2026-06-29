#!/usr/bin/env bash
# Field-vs-spectral ablation sweep on GPU 0 (override: DEVICE=cuda:N).
# Survives logout when launched with nohup/tmux (see bottom). Logs -> logs/ablation_gpu.log
# Outputs: reports/phase6_sweep_ablation.json (+ _pareto/_knob png), reports/phase6_sweep_recon0.json
set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root
DEVICE="${DEVICE:-cuda:0}"
mkdir -p logs reports
LOG="logs/ablation_gpu.log"
FAM="late early nf_lainr nf_omnifield late_mbff lainr_relu nf_lainr_single nf_lainr_nb1 nf_lainr_nb2 nf_lainr_nb8"
{
  echo "==== [$(date)] ablation START  DEVICE=$DEVICE ===="
  uv run make sweep DEVICE="$DEVICE" BASE=easy KNOB=snr_db VALUES="12 6 3 0 -3 -6" \
       SEEDS="0 1 2" STEPS=5000 NTRAIN=1024 FAMILIES="$FAM" OUT="reports/phase6_sweep_ablation.json"
  echo "==== [$(date)] recon=0 de-confound cell ===="
  uv run python -m src.sweep.runner --base easy --knob snr_db --values 12 6 3 0 -3 -6 \
       --seeds 0 1 2 --steps 5000 --n-train 1024 --families nf_lainr --recon-weight 0 \
       --device "$DEVICE" --out "reports/phase6_sweep_recon0.json"
  echo "==== [$(date)] ablation DONE -> reports/phase6_sweep_ablation.json, _recon0.json ===="
} 2>&1 | tee -a "$LOG"
