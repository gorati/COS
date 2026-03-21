#!/usr/bin/env bash
set -euo pipefail

# Main publication-oriented run:
#   SEVEM full map + mask + synthetic/mock null ensemble + global p-value

export PY_SCRIPT="${PY_SCRIPT:-./cmb_time_arrow_MI_scan_axes.py}"
export DATA_FITS_DIR="${DATA_FITS_DIR:-.}"
export MASK_PATH="${MASK_PATH:-${DATA_FITS_DIR%/}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits}"
export MAP_LIST="${MAP_LIST:-sevem}"
export OUT_BASE="${OUT_BASE:-cmb_timearrow_runs_pr3_sevem_main}"

export RUN_MAIN_DATA=1
export RUN_RESOLUTION_TESTS=0
export RUN_MI_BIN_TESTS=0
export RUN_ALT_AXIS_TESTS=1
export RUN_MOCK=1
export RUN_GLOBAL_REEVAL=1

# Use root-level mock_fits by default for the single-map SEVEM run.
export MOCK_FITS_ROOT="${MOCK_FITS_ROOT:-./mock_fits}"
export MOCK_LIMIT="${MOCK_LIMIT:-300}"

bash "${ENGINE_SH:-./cmb_time_arrow_MI_scan_axes_full_main.sh}"
