#!/usr/bin/env bash
set -euo pipefail

# Robustness-only run across all 4 PR3 component-separated maps.
# No mocks/global p by default.

export PY_SCRIPT="${PY_SCRIPT:-./cmb_time_arrow_MI_scan_axes.py}"
export DATA_FITS_DIR="${DATA_FITS_DIR:-.}"
export MASK_PATH="${MASK_PATH:-${DATA_FITS_DIR%/}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits}"
export MAP_LIST="${MAP_LIST:-smica sevem nilc commander}"
export OUT_BASE="${OUT_BASE:-cmb_timearrow_runs_pr3_allmaps_robustness}"

export RUN_MAIN_DATA=1
export RUN_RESOLUTION_TESTS=0
export RUN_MI_BIN_TESTS=0
export RUN_ALT_AXIS_TESTS=0
export RUN_MOCK=0
export RUN_GLOBAL_REEVAL=0

bash "${ENGINE_SH:-./cmb_time_arrow_MI_scan_axes_full_main.sh}"
