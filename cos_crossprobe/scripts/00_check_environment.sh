#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Python: $($PYTHON_BIN --version)"

echo "[INFO] Checking required Python packages..."
"$PYTHON_BIN" - <<'PY'
import importlib.util
required = ["numpy", "pandas", "matplotlib", "scipy", "astropy", "healpy"]
optional = ["dynesty"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing required packages: " + ", ".join(missing))
print("Required packages: OK")
for m in optional:
    print(f"Optional {m}:", "available" if importlib.util.find_spec(m) else "not installed")
PY

echo "[INFO] Checking local data files..."
test -f data/pantheon/Pantheon+SH0ES.dat || { echo "[ERROR] Missing data/pantheon/Pantheon+SH0ES.dat"; exit 1; }
test -f data/pantheon/Pantheon+SH0ES_STAT+SYS.cov || { echo "[ERROR] Missing data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"; exit 1; }
if [[ -f data/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits && -f data/cmb/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits ]]; then
  echo "[INFO] CMB FITS files: present"
else
  echo "[WARN] CMB FITS files not found. Pantheon stages can run; CMB follow-up scripts require files in data/cmb/."
fi

echo "[OK] Environment check complete."
