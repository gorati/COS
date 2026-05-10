#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/00_check_environment.sh
bash scripts/01_pantheon_axis_scan.sh
bash scripts/02_split_sample_validation.sh
bash scripts/03_axis_recurrence_map.sh
echo "[OK] Pantheon-only + recurrence stages complete. CMB follow-up requires FITS files under data/cmb/."
