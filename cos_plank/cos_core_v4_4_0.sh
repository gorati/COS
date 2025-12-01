#!/usr/bin/env bash
set -euo pipefail

# ================================
# CORE v4.3 runset (kód: v4.4.0)
# ================================

# micromamba helye (PATH-ból)
MAMBA_BIN="$(command -v micromamba || true)"
if [[ -z "${MAMBA_BIN}" ]]; then
  echo "ERROR: micromamba not found in PATH. Install or add to PATH." >&2
  exit 1
fi

# --- Elérési utak: állítsd be, ha szükséges ---
SCRIPT="cos_planck_v4_4_0.py"

# Planck fájlok törzskönyvtára (WSL alatt ez Nálad ez volt)
PLANCK="/mnt/d/Projects"

# Térképek/maszkok
MAP_SMICA="$PLANCK/COM_CMB_IQU-smica_2048_R3.00_full.fits"
MAP_NILC="$PLANCK/COM_CMB_IQU-nilc_2048_R3.00_full.fits"
MAP_SEVEM="$PLANCK/COM_CMB_IQU-sevem_2048_R3.00_full.fits"
MAP_COMMANDER="$PLANCK/COM_CMB_IQU-commander_2048_R3.00_full.fits"
MASK_COMMON="$PLANCK/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
MASK_STRICT="$PLANCK/COM_Mask_CMB-common-Mask-Int_2048_R3.00_STRICT_b30.fits"

# Elméleti Cl fájl (ha nincs, a szkript enélkül fut)
CL_FILE="$PLANCK/theory/lcdm_tt_ell_cl.txt"

# --- Futási paraméterek ---
NSIDE=256
THREADS=12
NSIMS=1000
SEED=1234
APOD=10

echo "Using micromamba at: $MAMBA_BIN"
echo "PLANCK dir: $PLANCK"
echo "SCRIPT: $SCRIPT"
echo "=== CORE v4.3 runs (nsims=$NSIMS) start ==="

# CL-fájl opcionális átadása
CL_OPT=()
if [[ -f "$CL_FILE" ]]; then
  CL_OPT=(--cl-file "$CL_FILE")
else
  echo "[Info] CL file not found ($CL_FILE) -> run without --cl-file."
fi

mkdir -p logs

# --- Futattó függvény: képernyõ + log (tee) ---
run_core(){
  local OUT="$1"; local LMAX="$2"; shift 2
  echo "-> RUN $OUT (lmax=$LMAX)"
  "$MAMBA_BIN" run -n cmb python "$SCRIPT" \
    --map "$MAP_SMICA" \
    --mask "$MASK_COMMON" \
    --out "$OUT" \
    --nside "$NSIDE" --lmax "$LMAX" --nsims "$NSIMS" --fwhm-arcmin 0 \
    --seed "$SEED" --threads "$THREADS" \
    --mask-apod-arcmin "$APOD" \
    "${CL_OPT[@]}" \
    "$@" \
    1> >(tee "logs/${OUT}.log") \
    2> >(cat >&2)
}

# -----------------------------
# A-blokk: SMICA + variációk
# -----------------------------

# A1) SMICA LCDM(th) lmax=384
run_core out_smica_ns256_l384_n1000_lcdm_th_apod${APOD}_v4_4_0 384

# A2) SMICA LCDM(th) lmax=128
run_core out_smica_ns256_l128_n1000_lcdm_th_apod${APOD}_v4_4_0 128

# A3) SMICA LCDM(th) lmax=096
run_core out_smica_ns256_l096_n1000_lcdm_th_apod${APOD}_v4_4_0 96

# A4) SMICA COS-IR (exp; L0=30, p=1.5) lmax=384
run_core out_smica_ns256_l384_n1000_cosir_exp_L0_30_p15_th_apod${APOD}_v4_4_0 384 \
  --alt cosir --ir-kind exp --ir-L0 30 --ir-p 1.5

# A5) SMICA COS-IR (exp; L0=32, p=2.0) lmax=384
run_core out_smica_ns256_l384_n1000_cosir_exp_L0_32_p20_th_apod${APOD}_v4_4_0 384 \
  --alt cosir --ir-kind exp --ir-L0 32 --ir-p 2.0

# A6) SMICA COS-IR (exp; L0=30, p=1.5) lmax=128
run_core out_smica_ns256_l128_n1000_cosir_exp_L0_30_p15_th_apod${APOD}_v4_4_0 128 \
  --alt cosir --ir-kind exp --ir-L0 30 --ir-p 1.5

# A7) SMICA konstans dipólus A=0.04, axis=(227,-27) lmax=384
run_core out_smica_ns256_l384_n1000_dipA004_axis227_-27_th_apod${APOD}_v4_4_0 384 \
  --alt dipole --dipA 0.04 --dip-axis "227,-27" --dip-coords gal

# A8) SMICA konstans dipólus A=0.05, axis=(180,-30) lmax=384
run_core out_smica_ns256_l384_n1000_dipA005_axis180_-30_th_apod${APOD}_v4_4_0 384 \
  --alt dipole --dipA 0.05 --dip-axis "180,-30" --dip-coords gal

# A9) SMICA konstans dipólus A=0.04, axis=(227,-27) lmax=128
run_core out_smica_ns256_l128_n1000_dipA004_axis227_-27_th_apod${APOD}_v4_4_0 128 \
  --alt dipole --dipA 0.04 --dip-axis "227,-27" --dip-coords gal

# -----------------------------------
# B-blokk: szigorúbb (b30) maszkkal
# -----------------------------------

# B10) SMICA strict b30 LCDM(th) lmax=384
run_core out_smica_ns256_l384_n1000_lcdm_th_apod${APOD}_strictb30_v4_4_0 384 \
  --mask "$MASK_STRICT"

# B11) SMICA strict b30 COS-IR (exp; L0=30, p=1.5) lmax=384
run_core out_smica_ns256_l384_n1000_cosir_exp_L0_30_p15_th_apod${APOD}_strictb30_v4_4_0 384 \
  --mask "$MASK_STRICT" \
  --alt cosir --ir-kind exp --ir-L0 30 --ir-p 1.5

# -----------------------------
# C-blokk: NILC/SEVEM/COMMANDER
# -----------------------------

# C12) NILC LCDM(th) lmax=384
run_core out_nilc_ns256_l384_n1000_lcdm_th_apod${APOD}_v4_4_0 384 \
  --map "$MAP_NILC"

# C13) SEVEM LCDM(th) lmax=384
run_core out_sevem_ns256_l384_n1000_lcdm_th_apod${APOD}_v4_4_0 384 \
  --map "$MAP_SEVEM"

# C14) COMMANDER LCDM(th) lmax=384
run_core out_commander_ns256_l384_n1000_lcdm_th_apod${APOD}_v4_4_0 384 \
  --map "$MAP_COMMANDER"

# C15) NILC COS-IR (exp; L0=30, p=1.5) lmax=384
run_core out_nilc_ns256_l384_n1000_cosir_exp_L0_30_p15_th_apod${APOD}_v4_4_0 384 \
  --map "$MAP_NILC" \
  --alt cosir --ir-kind exp --ir-L0 30 --ir-p 1.5

# C16) SEVEM COS-IR (exp; L0=30, p=1.5) lmax=384
run_core out_sevem_ns256_l384_n1000_cosir_exp_L0_30_p15_th_apod${APOD}_v4_4_0 384 \
  --map "$MAP_SEVEM" \
  --alt cosir --ir-kind exp --ir-L0 30 --ir-p 1.5

echo "=== CORE v4.4.0 runs finished ==="
