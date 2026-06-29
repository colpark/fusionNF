#!/usr/bin/env bash
# Download the BIDMC PPG+ECG dataset (PhysioNet, public) into data/bidmc/.
# Usage: bash scripts/download_bidmc.sh [N]   (N = number of records, default 53)
set -euo pipefail
N="${1:-53}"
DEST="data/bidmc"
mkdir -p "$DEST"
base="https://physionet.org/files/bidmc/1.0.0/bidmc_csv"
echo "Downloading $N BIDMC records to $DEST ..."
ok=0
for i in $(seq -w 1 "$N"); do
  url="$base/bidmc_${i}_Signals.csv"
  if curl -fsS --max-time 120 -o "$DEST/bidmc_${i}_Signals.csv" "$url"; then
    ok=$((ok+1))
  else
    rm -f "$DEST/bidmc_${i}_Signals.csv"
  fi
done
echo "Downloaded $ok/$N records into $DEST"
