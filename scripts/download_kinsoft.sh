#!/usr/bin/env bash
# Download the kinSoftChallenge smFRET dataset (Zenodo) and visualize it.
# Usage: bash scripts/download_kinsoft.sh [zenodo_record_id]   (default 5701310)
# Runs on the remote; logs -> logs/smfret_download.log ; data -> data/kinsoft/
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
mkdir -p data/kinsoft logs reports
REC="${1:-5701310}"
{
  echo "==== [$(date)] kinSoftChallenge download (Zenodo record $REC) ===="
  uv run python - "$REC" <<'PY'
import sys, json, os, urllib.request
rec = sys.argv[1]; out = "data/kinsoft"; os.makedirs(out, exist_ok=True)
api = f"https://zenodo.org/api/records/{rec}"
print("querying", api)
data = json.load(urllib.request.urlopen(api))
files = data.get("files", [])
print(f"{len(files)} files, {sum(f.get('size',0) for f in files)/1e6:.0f} MB total")
for f in files:
    url = f["links"]["self"]; name = f["key"]; dst = os.path.join(out, name)
    if os.path.exists(dst) and os.path.getsize(dst) == f.get("size", -1):
        print("have", name); continue
    print("downloading", name, f"({f.get('size',0)/1e6:.1f} MB)")
    urllib.request.urlretrieve(url, dst)
print("done ->", out)
PY
  echo "==== [$(date)] inspecting + visualizing real data ===="
  uv run python -m src.smfret.viz
  echo "==== [$(date)] kinSoft DONE (data/kinsoft, fig reports/smfret_examples/real_sample.png) ===="
} 2>&1 | tee -a logs/smfret_download.log
