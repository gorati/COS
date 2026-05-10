#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p results/rerun/recurrence_map

"$PYTHON_BIN" src/cos_crossprobe/cos_axis_recurrence_map.py \
  --split-json results/rerun/split_validation/split_crossprobe.json \
  --out results/rerun/recurrence_map/axis_recurrence_map.json \
  --plot results/rerun/recurrence_map/axis_recurrence_map.png \
  --cluster-radius-deg 25 \
  --min-cluster-members 2 \
  --ranking-mode hybrid \
  --export-kind centroid \
  --emit-cmb-command \
  --crossprobe-script src/cos_crossprobe/cos_crossprobe_fixed_axis_patched.py \
  --map data/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --mask data/cmb/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits

echo "[OK] Wrote results/rerun/recurrence_map/axis_recurrence_map.json"
echo "[OK] Wrote results/rerun/recurrence_map/axis_recurrence_map.png"
