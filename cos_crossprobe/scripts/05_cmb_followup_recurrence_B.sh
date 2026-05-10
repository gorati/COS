#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p results/rerun/cmb_followup

MAP="data/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
MASK="data/cmb/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
[[ -f "$MAP" ]] || { echo "[ERROR] Missing $MAP"; exit 1; }
[[ -f "$MASK" ]] || { echo "[ERROR] Missing $MASK"; exit 1; }

"$PYTHON_BIN" src/cos_crossprobe/cos_crossprobe_fixed_axis_patched.py \
  --axis-lon 298.7011922545 \
  --axis-lat 52.4384716902 \
  --axis-coords gal \
  --map "$MAP" \
  --mask "$MASK" \
  --work-nside 256 \
  --lmax-grid 8,16,24,32,48,64,96,128,192,256 \
  --mi-estimator knn \
  --knn-k 5 \
  --mi-sample-size 20000 \
  --n-phase-null 200 \
  --out results/rerun/cmb_followup/crossprobe_recurrence_B.json \
  --plot results/rerun/cmb_followup/crossprobe_recurrence_B.png

echo "[OK] Wrote results/rerun/cmb_followup/crossprobe_recurrence_B.json"
