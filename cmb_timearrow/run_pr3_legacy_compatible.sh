#!/usr/bin/env bash
set -euo pipefail

# Legacy-compatible reproduction of the old shell intent, but on the fixed Python.
# Covers:
#   1) all four PR3 full maps,
#   2) SMICA resolution tests,
#   3) SMICA MI-bin sensitivity,
#   4) SMICA alternative-axis test.
#
# Important difference vs. the old shell:
#   - entropy-bins is not separately swept anymore, because the fixed Python
#     treats it as deprecated/compatibility-only.
#   - NSIDE=512 remains supported, but is NOT default here to avoid easy OOM/kill.
#     Re-enable with: export RESOLUTION_NSIDE_LIST="128 256 512"

export PY_SCRIPT="${PY_SCRIPT:-./cmb_time_arrow_MI_scan_axes.py}"
export DATA_FITS_DIR="${DATA_FITS_DIR:-.}"
export MASK_PATH="${MASK_PATH:-${DATA_FITS_DIR%/}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits}"
export MAP_LIST="${MAP_LIST:-smica sevem nilc commander}"
export OUT_BASE="${OUT_BASE:-cmb_timearrow_runs_pr3_legacy_compatible}"

export RUN_MAIN_DATA=1
export RUN_RESOLUTION_TESTS=1
export RUN_MI_BIN_TESTS=1
export RUN_ALT_AXIS_TESTS=1
export RUN_MOCK=0
export RUN_GLOBAL_REEVAL=0

export RESOLUTION_MAPS="${RESOLUTION_MAPS:-smica}"
export RESOLUTION_NSIDE_LIST="${RESOLUTION_NSIDE_LIST:-128 256}"
export MI_TEST_MAPS="${MI_TEST_MAPS:-smica}"
export ALT_AXIS_MAPS="${ALT_AXIS_MAPS:-smica}"

bash "${ENGINE_SH:-./cmb_time_arrow_MI_scan_axes_full_main.sh}"
