#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Beállítások
# ==============================

# Uchuu EDR VAC v1.0 alap URL (DESI hivatalos)
MOCK_BASE_URL="https://data.desi.lbl.gov/public/edr/vac/edr/uchuu/v1.0"

# Hova töltsük le a mock fájlokat (FITS-ek)
MOCK_DATA_DIR="uchuu_edr_v1_0"

# Hova írjon eredményeket a COS-kód
MOCK_OUT_BASE="results_uchuu_mocks"

# Hány mockot futtassunk tracer/égfélpárra (0..101)
# N_MOCKS=102 esetén mind a 102 lightcone lefut (időigényes!)
N_MOCKS="${N_MOCKS:-20}"

# Python interpreter
PY="${PY:-python}"

mkdir -p "${MOCK_DATA_DIR}"
mkdir -p "${MOCK_OUT_BASE}"

if (( N_MOCKS < 1 || N_MOCKS > 102 )); then
  echo "[HIBA] N_MOCKS=${N_MOCKS} kívül esik az [1, 102] intervallumon."
  exit 1
fi

# ==============================
# Általános letöltő függvény
# ==============================
# Használat: dl_mock RELPATH
#   ahol RELPATH pl. "BGS-BRIGHT_Uchuu/BGS_BRIGHT_N_0_uchuu.dat.fits"
dl_mock() {
  local relpath="$1"
  local url="${MOCK_BASE_URL}/${relpath}"
  local fullpath="${MOCK_DATA_DIR}/${relpath}"

  mkdir -p "$(dirname "${fullpath}")"

  if [ -s "${fullpath}" ]; then
    echo "[*] Már létezik: ${fullpath}"
    return
  fi

  echo "[*] Letöltés: ${url}"
  curl -L -C - -o "${fullpath}" "${url}"

  if [ ! -s "${fullpath}" ]; then
    echo "[HIBA] Üres vagy hiányzó fájl letöltés után: ${fullpath}"
    exit 1
  fi
}

# ==============================
# Egyetlen mock lightcone futtatása
# ==============================
# Használat: run_one_mock TRACER SKY IDX
#   TRACER: BGS / LRG / ELG / QSO
#   SKY:    N / S
#   IDX:    0..101 (lightcone index)
run_one_mock() {
  local tracer="$1"
  local sky="$2"
  local idx="$3"

  local subdir=""
  local data_fname=""
  local ran_fname=""
  local zmin=""
  local zmax=""

  case "${tracer}" in
    BGS)
      subdir="BGS-BRIGHT_Uchuu"
      data_fname="BGS_BRIGHT_${sky}_${idx}_uchuu.dat.fits"
      # BGS-hez csak 18 random van: BGS_BRIGHT_[N/S]_0..17_uchuu.ran.fits
      # A 102 lightcone-hoz ciklikusan osztjuk be a 18 randomot.
      local ran_idx=$((idx % 18))
      ran_fname="BGS_BRIGHT_${sky}_${ran_idx}_uchuu.ran.fits"
      # Z-tartomány a doksi szerint: 0.0–0.6, mi 0.1–0.6-ra vágunk (DR1 BGS-analóg)
      zmin="0.1"
      zmax="0.6"
      ;;
    LRG)
      subdir="LRG_main_Uchuu"
      data_fname="LRG_main_${sky}_${idx}_uchuu.dat.fits"
      # Randomból egy darab/égfél: LRG_main_N_uchuu.ran.fits, LRG_main_S_uchuu.ran.fits
      ran_fname="LRG_main_${sky}_uchuu.ran.fits"
      zmin="0.4"
      zmax="1.1"
      ;;
    ELG)
      subdir="ELG_Uchuu"
      data_fname="ELG_${sky}_${idx}_uchuu.dat.fits"
      # ELG-hez 102 random van: ELG_[N/S]_[i]_uchuu.ran.fits
      ran_fname="ELG_${sky}_${idx}_uchuu.ran.fits"
      # Doksi: 0.88–1.34
      zmin="0.88"
      zmax="1.34"
      ;;
    QSO)
      subdir="QSO_Uchuu"
      data_fname="QSO_${sky}_${idx}_uchuu.dat.fits"
      # Randomból egy darab/égfél: QSO_N_uchuu.ran.fits, QSO_S_uchuu.ran.fits
      ran_fname="QSO_${sky}_uchuu.ran.fits"
      zmin="0.9"
      zmax="2.1"
      ;;
    *)
      echo "[HIBA] Ismeretlen tracer: ${tracer}"
      exit 1
      ;;
  esac

  local rel_data="${subdir}/${data_fname}"
  local rel_ran="${subdir}/${ran_fname}"

  # Letöltés, ha kell
  dl_mock "${rel_data}"
  dl_mock "${rel_ran}"

  local outdir="${MOCK_OUT_BASE}/${tracer}_${sky}/mock_${idx}"
  mkdir -p "${outdir}"

  echo "[*] Futtatás (mock): tracer=${tracer}, sky=${sky}, idx=${idx}, z=[${zmin}, ${zmax}]"

  "${PY}" cos_desi_tests.py \
    --data    "${MOCK_DATA_DIR}/${rel_data}" \
    --randoms "${MOCK_DATA_DIR}/${rel_ran}" \
    --outdir  "${outdir}" \
    --z-min   "${zmin}" --z-max "${zmax}" \
    --r-min   5 --r-max 150 --r-bins 20 \
    --n-axes  256 \
    --ra-col  RA --dec-col DEC --z-col Z \
    --w-cols-data    WEIGHT_FKP \
    --w-cols-random  WEIGHT_FKP \
    --subsample-2pcf 40000 \
    --subsample-hemi 80000 \
    --mst-n-max      5000

  echo "[*] Kész mock: ${tracer}-${sky}, idx=${idx}"
}

# ==============================
# Tracer+égfél mock-sorozat futtatása
# ==============================
run_mock_suite_for_tracer() {
  local tracer="$1"
  local n_mocks="$2"

  for sky in N S; do
    for idx in $(seq 0 $((n_mocks-1))); do
      run_one_mock "${tracer}" "${sky}" "${idx}"
    done
  done
}

# ==============================
# Fő futási logika
# ==============================

echo "[*] Uchuu mock futtatás indul. N_MOCKS tracer/égfélpárra: ${N_MOCKS}"

# BGS
run_mock_suite_for_tracer BGS "${N_MOCKS}"
# LRG
run_mock_suite_for_tracer LRG "${N_MOCKS}"
# ELG
run_mock_suite_for_tracer ELG "${N_MOCKS}"
# QSO
run_mock_suite_for_tracer QSO "${N_MOCKS}"

echo "[OK] Uchuu mock futtatás kész. Eredmények itt: ${MOCK_OUT_BASE}"
