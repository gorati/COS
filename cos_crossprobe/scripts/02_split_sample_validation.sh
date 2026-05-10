#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p results/rerun/split_validation

"$PYTHON_BIN" src/cos_crossprobe/cos_split_sample_crossprobe_standalone_checkpointed.py \
  --data data/pantheon/Pantheon+SH0ES.dat \
  --cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov \
  --out results/rerun/split_validation/split_crossprobe.json \
  --checkpoint-out results/rerun/split_validation/split_crossprobe.checkpoint.json \
  --statistic abs_delta_q0 \
  --n-random-axes 1000 \
  --n-splits 20 \
  --validation-fraction 0.5 \
  --stratify-z \
  --z-strat-bins 5 \
  --n-validation-null 200

echo "[OK] Wrote results/rerun/split_validation/split_crossprobe.json"
echo "[OK] Wrote results/rerun/split_validation/split_crossprobe.checkpoint.json"
