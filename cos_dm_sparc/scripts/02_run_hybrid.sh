#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/current/hybrid
python src/cos_dm_sparc/fit.py data/sparc/Rotmod_LTG data/sparc/SPARC_Lelli2016c.mrt --pattern "**/*.dat" --modulate hybrid --loss chi2 --restarts 3 --maxiter 180 --out-prefix results/current/hybrid/cos_dm_hybrid
