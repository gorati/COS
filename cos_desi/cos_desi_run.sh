#!/usr/bin/env bash
set -euo pipefail

# === Beállítások ===
DATA_DIR="desi_dr1_LSS_v1.5"
OUT_DIR_BASE="results_desi_v1_5"

# Ha külön micromamba env-et használsz, ezt vedd elõ:
# PY="micromamba run -n cmb python"
PY="python"

mkdir -p "${DATA_DIR}"
mkdir -p "${OUT_DIR_BASE}"

# === Letöltõ függvény ===
# Használat: dl URL FILENAME
dl() {
  local url="$1"
  local fn="$2"
  echo "[*] Letoltes: ${fn}"
  curl -L -C - -o "${DATA_DIR}/${fn}" "${url}"
  if [ ! -s "${DATA_DIR}/${fn}" ]; then
    echo "[HIBA] Ures vagy nem letezo fajl: ${fn}"
    exit 1
  fi
}

# === Futattó függvény ===
# Használat: run_one TRACER SKY DATA_FNAME RAN_FNAME ZMIN ZMAX
run_one() {
  local tracer="$1"
  local sky="$2"
  local data_fname="$3"
  local ran_fname="$4"
  local zmin="$5"
  local zmax="$6"
  local outdir="${OUT_DIR_BASE}/${tracer}_${sky}"

  mkdir -p "${outdir}"
  echo "[*] Futtatas: ${tracer}-${sky}  z=[${zmin}, ${zmax}]"

  ${PY} cos_desi_tests.py \
    --data    "${DATA_DIR}/${data_fname}" \
    --randoms "${DATA_DIR}/${ran_fname}" \
    --outdir  "${outdir}" \
    --z-min   "${zmin}" --z-max "${zmax}" \
    --r-min 5 --r-max 150 --r-bins 20 \
    --n-axes 256 \
    --ra-col RA --dec-col DEC --z-col Z \
    --w-cols-data WEIGHT \
    --w-cols-random WEIGHT \
    --subsample-2pcf 40000 \
    --subsample-hemi 80000 \
    --mst-n-max 5000

  echo "[*] Kesz: ${tracer}-${sky}"
}

# === 1) URL-ek és fájlnevek – adat KATALÓGUSOK ===
URL_BGS_NGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/BGS_BRIGHT_NGC_clustering.dat.fits"
URL_BGS_SGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/BGS_BRIGHT_SGC_clustering.dat.fits"
URL_LRG_NGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/LRG_NGC_clustering.dat.fits"
URL_LRG_SGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/LRG_SGC_clustering.dat.fits"
URL_ELG_NGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/ELG_LOPnotqso_NGC_clustering.dat.fits"
URL_ELG_SGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/ELG_LOPnotqso_SGC_clustering.dat.fits"
URL_QSO_NGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/QSO_NGC_clustering.dat.fits"
URL_QSO_SGC="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/QSO_SGC_clustering.dat.fits"

FN_BGS_NGC="BGS_BRIGHT_NGC_clustering.dat.fits"
FN_BGS_SGC="BGS_BRIGHT_SGC_clustering.dat.fits"
FN_LRG_NGC="LRG_NGC_clustering.dat.fits"
FN_LRG_SGC="LRG_SGC_clustering.dat.fits"
FN_ELG_NGC="ELG_LOPnotqso_NGC_clustering.dat.fits"
FN_ELG_SGC="ELG_LOPnotqso_SGC_clustering.dat.fits"
FN_QSO_NGC="QSO_NGC_clustering.dat.fits"
FN_QSO_SGC="QSO_SGC_clustering.dat.fits"

# === 1b) URL-ek és fájlnevek – RANDOM KATALÓGUSOK (RANN=0) ===
URL_BGS_NGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/BGS_BRIGHT_NGC_0_clustering.ran.fits"
URL_BGS_SGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/BGS_BRIGHT_SGC_0_clustering.ran.fits"
URL_LRG_NGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/LRG_NGC_0_clustering.ran.fits"
URL_LRG_SGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/LRG_SGC_0_clustering.ran.fits"
URL_ELG_NGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/ELG_LOPnotqso_NGC_0_clustering.ran.fits"
URL_ELG_SGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/ELG_LOPnotqso_SGC_0_clustering.ran.fits"
URL_QSO_NGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/QSO_NGC_0_clustering.ran.fits"
URL_QSO_SGC_RAN="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/QSO_SGC_0_clustering.ran.fits"

FN_BGS_NGC_RAN="BGS_BRIGHT_NGC_0_clustering.ran.fits"
FN_BGS_SGC_RAN="BGS_BRIGHT_SGC_0_clustering.ran.fits"
FN_LRG_NGC_RAN="LRG_NGC_0_clustering.ran.fits"
FN_LRG_SGC_RAN="LRG_SGC_0_clustering.ran.fits"
FN_ELG_NGC_RAN="ELG_LOPnotqso_NGC_0_clustering.ran.fits"
FN_ELG_SGC_RAN="ELG_LOPnotqso_SGC_0_clustering.ran.fits"
FN_QSO_NGC_RAN="QSO_NGC_0_clustering.ran.fits"
FN_QSO_SGC_RAN="QSO_SGC_0_clustering.ran.fits"

# === 2) Letöltés – adat + random mind a 8 tracerre ===
dl "${URL_BGS_NGC}"     "${FN_BGS_NGC}"
dl "${URL_BGS_SGC}"     "${FN_BGS_SGC}"
dl "${URL_LRG_NGC}"     "${FN_LRG_NGC}"
dl "${URL_LRG_SGC}"     "${FN_LRG_SGC}"
dl "${URL_ELG_NGC}"     "${FN_ELG_NGC}"
dl "${URL_ELG_SGC}"     "${FN_ELG_SGC}"
dl "${URL_QSO_NGC}"     "${FN_QSO_NGC}"
dl "${URL_QSO_SGC}"     "${FN_QSO_SGC}"

dl "${URL_BGS_NGC_RAN}" "${FN_BGS_NGC_RAN}"
dl "${URL_BGS_SGC_RAN}" "${FN_BGS_SGC_RAN}"
dl "${URL_LRG_NGC_RAN}" "${FN_LRG_NGC_RAN}"
dl "${URL_LRG_SGC_RAN}" "${FN_LRG_SGC_RAN}"
dl "${URL_ELG_NGC_RAN}" "${FN_ELG_NGC_RAN}"
dl "${URL_ELG_SGC_RAN}" "${FN_ELG_SGC_RAN}"
dl "${URL_QSO_NGC_RAN}" "${FN_QSO_NGC_RAN}"
dl "${URL_QSO_SGC_RAN}" "${FN_QSO_SGC_RAN}"

# (Opció) SHA256 hash-ek mentése
echo "[*] SHA256 ellenorzes (sha256sum)..."
(
  cd "${DATA_DIR}"
  for f in *.fits; do
    echo "==== ${f} ===="
    sha256sum "${f}"
  done
) > "${DATA_DIR}/SHA256SUMS.txt"
echo "[*] Kesz: ${DATA_DIR}/SHA256SUMS.txt"

# === 3) Futtatás mind a 8 tracerre ===
# Tipikus z-ablakok: BGS: 0.1–0.6, LRG: 0.4–1.1, ELG: 0.6–1.6, QSO: 0.8–2.1
run_one BGS NGC "${FN_BGS_NGC}" "${FN_BGS_NGC_RAN}" 0.1 0.6
run_one BGS SGC "${FN_BGS_SGC}" "${FN_BGS_SGC_RAN}" 0.1 0.6

run_one LRG NGC "${FN_LRG_NGC}" "${FN_LRG_NGC_RAN}" 0.4 1.1
run_one LRG SGC "${FN_LRG_SGC}" "${FN_LRG_SGC_RAN}" 0.4 1.1

run_one ELG NGC "${FN_ELG_NGC}" "${FN_ELG_NGC_RAN}" 0.6 1.6
run_one ELG SGC "${FN_ELG_SGC}" "${FN_ELG_SGC_RAN}" 0.6 1.6

run_one QSO NGC "${FN_QSO_NGC}" "${FN_QSO_NGC_RAN}" 0.8 2.1
run_one QSO SGC "${FN_QSO_SGC}" "${FN_QSO_SGC_RAN}" 0.8 2.1

echo "[OK] Kesz. Eredmenyek itt: ${OUT_DIR_BASE}"
