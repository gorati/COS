#!/bin/bash

MASK="COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
LMAX_GRID="8,16,24,32,48,64,96,128,192,256"

###############################################################################
# 1) MI-alap teszt mind a 4 Planck térképre (SMICA, SEVEM, NILC, COMMANDER)
#    NSIDE=256, 1000 random tengely, 3 különbözõ seed
###############################################################################

for MAP in smica sevem nilc commander; do
  for SEED in 12345 54321 98765; do
    echo "[info] Fut: MAP=${MAP}, NSIDE=256, n_axes=1000, seed=${SEED}"
    python cmb_time_arrow_MI_scan_axes.py \
      --map "COM_CMB_IQU-${MAP}_2048_R3.00_full.fits" \
      --mask "${MASK}" \
      --work-nside 256 \
      --lmax-grid "${LMAX_GRID}" \
      --entropy-bins 64 \
      --mi-bins 32 \
      --n-axes 1000 \
      --seed ${SEED} \
      --cos-axis "0,90" \
      --cos-coords gal \
      --out "cmb_time_arrow_MI_${MAP}_nside256_axes1000_seed${SEED}.json"
  done
done

###############################################################################
# 2) SMICA: felbontás teszt (NSIDE=128,256,512) 2 különbözõ seeddel
###############################################################################

for NSIDE in 128 256 512; do
  for SEED in 12345 98765; do
    echo "[info] Fut: SMICA, NSIDE=${NSIDE}, n_axes=1000, seed=${SEED}"
    python cmb_time_arrow_MI_scan_axes.py \
      --map "COM_CMB_IQU-smica_2048_R3.00_full.fits" \
      --mask "${MASK}" \
      --work-nside ${NSIDE} \
      --lmax-grid "${LMAX_GRID}" \
      --entropy-bins 64 \
      --mi-bins 32 \
      --n-axes 1000 \
      --seed ${SEED} \
      --cos-axis "0,90" \
      --cos-coords gal \
      --out "cmb_time_arrow_MI_smica_nside${NSIDE}_axes1000_seed${SEED}.json"
  done
done

###############################################################################
# 3) SMICA: binszám teszt (hisztogram diszkretizáció érzékenysége)
#    entropy-bins {32,64,96}, mi-bins {16,32,48}
###############################################################################

for ENT in 32 64 96; do
  for MI in 16 32 48; do
    echo "[info] Fut: SMICA, NSIDE=256, entropy_bins=${ENT}, mi_bins=${MI}"
    python cmb_time_arrow_MI_scan_axes.py \
      --map "COM_CMB_IQU-smica_2048_R3.00_full.fits" \
      --mask "${MASK}" \
      --work-nside 256 \
      --lmax-grid "${LMAX_GRID}" \
      --entropy-bins ${ENT} \
      --mi-bins ${MI} \
      --n-axes 1000 \
      --seed 12345 \
      --cos-axis "0,90" \
      --cos-coords gal \
      --out "cmb_time_arrow_MI_smica_nside256_ent${ENT}_mi${MI}_axes1000_seed12345.json"
  done
done

###############################################################################
# 4) SMICA: alternatív "COS-tengely" teszt (pl. 227,-27) 3 seeddel
#    — ez azt nézi, hogy tengelyválasztásra mennyire stabil az idõnyíl-jel
###############################################################################

for SEED in 12345 54321 98765; do
  echo "[info] Fut: SMICA, NSIDE=256, axis=227,-27, seed=${SEED}"
  python cmb_time_arrow_MI_scan_axes.py \
    --map "COM_CMB_IQU-smica_2048_R3.00_full.fits" \
    --mask "${MASK}" \
    --work-nside 256 \
    --lmax-grid "${LMAX_GRID}" \
    --entropy-bins 64 \
    --mi-bins 32 \
    --n-axes 1000 \
    --seed ${SEED} \
    --cos-axis "227,-27" \
    --cos-coords gal \
    --out "cmb_time_arrow_MI_smica_nside256_axes1000_seed${SEED}_axis227-27.json"
done
