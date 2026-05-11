#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/current
N_JOBS="${N_JOBS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "src/cos_pantheon/axis_scan.py" \
  --data "data/pantheon/Pantheon+SH0ES.dat" \
  --cov "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov" \
  --out "results/current/pantheon_axis_scan_q0_fixed.json" \
  --zmin 0.01 \
  --zmax 0.10 \
  --cos-coords gal \
  --cos-lon 0 \
  --cos-lat 90 \
  --statistic abs_delta_q0 \
  --run-null \
  --null-mode sky-scramble \
  --n-random-axes 1000 \
  --n-null 500 \
  --axis-seed 12345 \
  --null-seed 24680 \
  --progress-every 10 \
  --n-jobs "$N_JOBS"
