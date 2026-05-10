#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p results/rerun/pantheon_axis_scan

"$PYTHON_BIN" src/cos_crossprobe/cos_pantheon_axis_scan.py \
  --data data/pantheon/Pantheon+SH0ES.dat \
  --cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov \
  --out results/rerun/pantheon_axis_scan/pantheon_axis_scan.json \
  --statistic abs_delta_q0 \
  --n-random-axes 1000 \
  --run-sky-scramble-null \
  --n-null 500

echo "[OK] Wrote results/rerun/pantheon_axis_scan/pantheon_axis_scan.json"
