#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Parameterized main pipeline for cmb_time_arrow_MI_scan_axes.py
#
# Design goals:
#   - one generic engine shell,
#   - configurable sections,
#   - wrappers can enable only the needed workflow,
#   - supports PR3 multi-map runs and SEVEM+mock main run,
#   - keeps NSIDE=512 optional instead of hard-disabled.
###############################################################################

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${PY_SCRIPT:-./cmb_time_arrow_MI_scan_axes.py}"

# Input directories
DATA_FITS_DIR="${DATA_FITS_DIR:-.}"
MASK_PATH="${MASK_PATH:-${DATA_FITS_DIR%/}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits}"

# Default PR3 map list
MAP_LIST="${MAP_LIST:-smica sevem nilc commander}"
MAP_FIELD="${MAP_FIELD:-0}"
MASK_FIELD="${MASK_FIELD:-0}"

# Output layout
OUT_BASE="${OUT_BASE:-cmb_timearrow_runs}"
LOG_DIR="${LOG_DIR:-${OUT_BASE}/logs}"
mkdir -p "${LOG_DIR}"

# Generic scan parameters
WORK_NSIDE_DEFAULT="${WORK_NSIDE_DEFAULT:-256}"
LMAX_GRID="${LMAX_GRID:-8,16,24,32,48,64,96,128,192,256}"
ENTROPY_BINS_DEFAULT="${ENTROPY_BINS_DEFAULT:-64}"
MI_BINS_DEFAULT="${MI_BINS_DEFAULT:-32}"
N_AXES_DEFAULT="${N_AXES_DEFAULT:-1000}"
COS_AXIS_DEFAULT="${COS_AXIS_DEFAULT:-0,90}"
ALT_COS_AXIS="${ALT_COS_AXIS:-227,-27}"
GLOBAL_NULL_MIN="${GLOBAL_NULL_MIN:-30}"

# Seeds
DATA_SEEDS="${DATA_SEEDS:-12345 54321 98765}"
RESOLUTION_SEEDS="${RESOLUTION_SEEDS:-12345 98765}"
ALT_AXIS_SEEDS="${ALT_AXIS_SEEDS:-12345 54321 98765}"
MI_TEST_SEED="${MI_TEST_SEED:-12345}"

# Which sections to run
RUN_MAIN_DATA="${RUN_MAIN_DATA:-1}"
RUN_RESOLUTION_TESTS="${RUN_RESOLUTION_TESTS:-0}"
RUN_MI_BIN_TESTS="${RUN_MI_BIN_TESTS:-0}"
RUN_ALT_AXIS_TESTS="${RUN_ALT_AXIS_TESTS:-0}"
RUN_MOCK="${RUN_MOCK:-0}"
RUN_GLOBAL_REEVAL="${RUN_GLOBAL_REEVAL:-0}"

# Resolution tests
RESOLUTION_MAPS="${RESOLUTION_MAPS:-smica}"
RESOLUTION_NSIDE_LIST="${RESOLUTION_NSIDE_LIST:-128 256 512}"

# MI-bin tests (entropy-bins kept only for CLI compatibility; not swept by default)
MI_TEST_MAPS="${MI_TEST_MAPS:-smica}"
MI_BIN_LIST="${MI_BIN_LIST:-16 32 48}"

# Alternative-axis tests
ALT_AXIS_MAPS="${ALT_AXIS_MAPS:-smica}"

# Mock / global settings
MOCK_LIMIT="${MOCK_LIMIT:-300}"
MOCK_SEED_BASE="${MOCK_SEED_BASE:-900000}"
# Base directory for mocks. Preferred layout for multi-map runs:
#   mock_fits/sevem/*.fits
#   mock_fits/smica/*.fits
#   ...
MOCK_FITS_ROOT="${MOCK_FITS_ROOT:-./mock_fits}"

log()  { printf '[info] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
err()  { printf '[error] %s\n' "$*" >&2; exit 1; }

[[ -f "${PY_SCRIPT}" ]] || err "Nem találom a Python scriptet: ${PY_SCRIPT}"

run_scan() {
  "${PYTHON_BIN}" "${PY_SCRIPT}" "$@"
}

resolve_map_path() {
  local map_name="$1"
  local candidates=(
    "${DATA_FITS_DIR%/}/COM_CMB_IQU-${map_name}_2048_R3.01_full.fits"
    "${DATA_FITS_DIR%/}/COM_CMB_IQU-${map_name}_2048_R3.00_full.fits"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -f "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  find "${DATA_FITS_DIR}" -maxdepth 2 -type f \( -iname "*.fits" -o -iname "*.fits.gz" \) \
    \( -iname "*${map_name}*full*.fits*" -o -iname "*${map_name}*.fits*" \) | sort | head -n 1
}

resolve_mask_path() {
  if [[ -n "${MASK_PATH}" && -f "${MASK_PATH}" ]]; then
    printf '%s\n' "${MASK_PATH}"
    return 0
  fi
  find "${DATA_FITS_DIR}" -maxdepth 2 -type f \( -iname '*.fits' -o -iname '*.fits.gz' \) \
    \( -iname '*Mask*CMB*common*Int*R3*.fits*' -o -iname '*mask*common*int*.fits*' \) | sort | head -n 1
}

map_data_tag() {
  local map_name="$1"
  printf 'pr3_%s\n' "${map_name}"
}

common_args_for_map() {
  local data_tag="$1"
  local mask_resolved="$2"
  local work_nside="$3"
  local mi_bins="$4"

  local -a args=(
    --map-field "${MAP_FIELD}"
    --work-nside "${work_nside}"
    --lmax-grid "${LMAX_GRID}"
    --entropy-bins "${ENTROPY_BINS_DEFAULT}"
    --mi-bins "${mi_bins}"
    --n-axes "${N_AXES_DEFAULT}"
    --cos-coords gal
    --dataset-tag "${data_tag}"
    --require-global-null-min "${GLOBAL_NULL_MIN}"
  )
  if [[ -n "${mask_resolved}" ]]; then
    args+=(--mask "${mask_resolved}" --mask-field "${MASK_FIELD}")
  fi
  printf '%s\n' "${args[@]}"
}

collect_mock_maps_for_map() {
  local map_name="$1"
  local -n _out_ref=$2
  _out_ref=()

  local preferred_dir="${MOCK_FITS_ROOT%/}/${map_name}"
  local search_dirs=()
  if [[ -d "${preferred_dir}" ]]; then
    search_dirs+=("${preferred_dir}")
  fi
  if [[ -d "${MOCK_FITS_ROOT}" ]]; then
    search_dirs+=("${MOCK_FITS_ROOT}")
  fi

  local dir path base
  for dir in "${search_dirs[@]}"; do
    while IFS= read -r path; do
      [[ -n "${path}" ]] || continue
      base="$(basename "${path}")"
      case "${base}" in
        *sim*|*SIM*|*mock*|*MOCK*|*mc*|*MC*)
          if [[ "${dir}" == "${MOCK_FITS_ROOT}" && "${base,,}" != *"${map_name}"* ]]; then
            # Root-level shared mock dir: accept all files for single-map runs,
            # but in multi-map runs prefer per-map dirs or name-tagged files.
            if [[ "${MAP_LIST}" == *" "* && "${MAP_LIST}" != "${map_name}" ]]; then
              continue
            fi
          fi
          _out_ref+=("${path}")
          ;;
      esac
    done < <(find "${dir}" -maxdepth 3 -type f \( -iname '*.fits' -o -iname '*.fits.gz' \) | sort)
    if [[ ${#_out_ref[@]} -gt 0 ]]; then
      break
    fi
  done

  if [[ ${#_out_ref[@]} -eq 0 ]]; then
    return 1
  fi
  if [[ "${MOCK_LIMIT}" =~ ^[0-9]+$ ]] && [[ "${MOCK_LIMIT}" -gt 0 ]] && [[ ${#_out_ref[@]} -gt ${MOCK_LIMIT} ]]; then
    _out_ref=("${_out_ref[@]:0:${MOCK_LIMIT}}")
  fi
  return 0
}

run_main_data_for_map() {
  local map_name="$1"
  local map_path="$2"
  local mask_resolved="$3"
  local data_tag="$4"
  local map_out_dir="${OUT_BASE}/${map_name}/data"
  mkdir -p "${map_out_dir}"

  local -a common_args
  mapfile -t common_args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${WORK_NSIDE_DEFAULT}" "${MI_BINS_DEFAULT}")

  local seed out_json
  for seed in ${DATA_SEEDS}; do
    out_json="${map_out_dir}/cmb_time_arrow_MI_${data_tag}_nside${WORK_NSIDE_DEFAULT}_axes${N_AXES_DEFAULT}_seed${seed}.json"
    log "Main run: map=${map_name}, seed=${seed}"
    run_scan \
      --mode data \
      --map "${map_path}" \
      "${common_args[@]}" \
      --seed "${seed}" \
      --cos-axis "${COS_AXIS_DEFAULT}" \
      --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
  done
}

run_resolution_tests_for_map() {
  local map_name="$1"
  local map_path="$2"
  local mask_resolved="$3"
  local data_tag="$4"
  local map_out_dir="${OUT_BASE}/${map_name}/resolution"
  mkdir -p "${map_out_dir}"

  local nside seed out_json
  for nside in ${RESOLUTION_NSIDE_LIST}; do
    for seed in ${RESOLUTION_SEEDS}; do
      out_json="${map_out_dir}/cmb_time_arrow_MI_${data_tag}_nside${nside}_axes${N_AXES_DEFAULT}_seed${seed}.json"
      log "Resolution test: map=${map_name}, nside=${nside}, seed=${seed}"
      local -a args
      mapfile -t args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${nside}" "${MI_BINS_DEFAULT}")
      run_scan \
        --mode data \
        --map "${map_path}" \
        "${args[@]}" \
        --seed "${seed}" \
        --cos-axis "${COS_AXIS_DEFAULT}" \
        --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
    done
  done
}

run_mi_bin_tests_for_map() {
  local map_name="$1"
  local map_path="$2"
  local mask_resolved="$3"
  local data_tag="$4"
  local map_out_dir="${OUT_BASE}/${map_name}/mi_bins"
  mkdir -p "${map_out_dir}"

  local mib out_json
  for mib in ${MI_BIN_LIST}; do
    out_json="${map_out_dir}/cmb_time_arrow_MI_${data_tag}_nside${WORK_NSIDE_DEFAULT}_mi${mib}_axes${N_AXES_DEFAULT}_seed${MI_TEST_SEED}.json"
    log "MI-bin test: map=${map_name}, mi_bins=${mib}"
    local -a args
    mapfile -t args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${WORK_NSIDE_DEFAULT}" "${mib}")
    run_scan \
      --mode data \
      --map "${map_path}" \
      "${args[@]}" \
      --seed "${MI_TEST_SEED}" \
      --cos-axis "${COS_AXIS_DEFAULT}" \
      --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
  done
}

run_alt_axis_tests_for_map() {
  local map_name="$1"
  local map_path="$2"
  local mask_resolved="$3"
  local data_tag="$4"
  local map_out_dir="${OUT_BASE}/${map_name}/alt_axis"
  mkdir -p "${map_out_dir}"

  local seed out_json
  for seed in ${ALT_AXIS_SEEDS}; do
    out_json="${map_out_dir}/cmb_time_arrow_MI_${data_tag}_nside${WORK_NSIDE_DEFAULT}_axes${N_AXES_DEFAULT}_seed${seed}_axis227-27.json"
    log "Alt-axis test: map=${map_name}, axis=${ALT_COS_AXIS}, seed=${seed}"
    local -a args
    mapfile -t args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${WORK_NSIDE_DEFAULT}" "${MI_BINS_DEFAULT}")
    run_scan \
      --mode data \
      --map "${map_path}" \
      "${args[@]}" \
      --seed "${seed}" \
      --cos-axis "${ALT_COS_AXIS}" \
      --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
  done
}

run_mocks_for_map() {
  local map_name="$1"
  local mask_resolved="$2"
  local data_tag="$3"
  local map_mock_dir="${OUT_BASE}/${map_name}/mock"
  mkdir -p "${map_mock_dir}"

  declare -a mock_maps
  if ! collect_mock_maps_for_map "${map_name}" mock_maps; then
    warn "Nem találtam mock FITS-eket ehhez a maphoz: ${map_name}"
    return 1
  fi
  log "Talált mock FITS-ek (${map_name}): ${#mock_maps[@]}"

  local -a args
  mapfile -t args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${WORK_NSIDE_DEFAULT}" "${MI_BINS_DEFAULT}")

  local idx=0 map_path base stem out_json seed
  for map_path in "${mock_maps[@]}"; do
    idx=$((idx + 1))
    base="$(basename "${map_path}")"
    stem="${base%.*}"
    stem="${stem%.fits}"
    out_json="${map_mock_dir}/${stem}.json"
    seed=$((MOCK_SEED_BASE + idx))
    log "Mock run ${idx}/${#mock_maps[@]} (${map_name}): ${base}"
    run_scan \
      --mode mock \
      --map "${map_path}" \
      "${args[@]}" \
      --seed "${seed}" \
      --cos-axis "${COS_AXIS_DEFAULT}" \
      --mock-id "${map_name}_mock_${idx}" \
      --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
  done
}

run_global_reeval_for_map() {
  local map_name="$1"
  local map_path="$2"
  local mask_resolved="$3"
  local data_tag="$4"
  local map_mock_dir="${OUT_BASE}/${map_name}/mock"
  local map_global_dir="${OUT_BASE}/${map_name}/global"
  mkdir -p "${map_global_dir}"

  shopt -s nullglob
  local mock_jsons=("${map_mock_dir}"/*.json)
  shopt -u nullglob

  if [[ ${#mock_jsons[@]} -eq 0 ]]; then
    warn "Nincs mock JSON ehhez a maphoz: ${map_name}; globális p-értéket kihagyom."
    return 1
  fi
  if [[ ${#mock_jsons[@]} -lt ${GLOBAL_NULL_MIN} ]]; then
    warn "Csak ${#mock_jsons[@]} mock JSON van (${map_name}); a globális p-érték bizonytalan lehet."
  fi

  local -a args
  mapfile -t args < <(common_args_for_map "${data_tag}" "${mask_resolved}" "${WORK_NSIDE_DEFAULT}" "${MI_BINS_DEFAULT}")

  local seed out_json
  for seed in ${DATA_SEEDS}; do
    out_json="${map_global_dir}/cmb_time_arrow_MI_${data_tag}_nside${WORK_NSIDE_DEFAULT}_axes${N_AXES_DEFAULT}_seed${seed}_global.json"
    log "Global re-eval: map=${map_name}, seed=${seed}"
    run_scan \
      --mode data \
      --map "${map_path}" \
      "${args[@]}" \
      --seed "${seed}" \
      --cos-axis "${COS_AXIS_DEFAULT}" \
      --global-null-dir "${map_mock_dir}" \
      --out "${out_json}" | tee "${LOG_DIR}/$(basename "${out_json%.json}").log"
  done
}

main() {
  local mask_resolved
  mask_resolved="$(resolve_mask_path || true)"
  if [[ -n "${mask_resolved}" ]]; then
    log "Mask: ${mask_resolved}"
  else
    warn "Nem találtam maszkot. Maszk nélküli futás lesz."
  fi

  local map_name map_path data_tag
  for map_name in ${MAP_LIST}; do
    map_path="$(resolve_map_path "${map_name}" || true)"
    [[ -n "${map_path}" ]] || err "Nem találom a mapet ehhez: ${map_name}"
    data_tag="$(map_data_tag "${map_name}")"
    log "Map: ${map_name} -> ${map_path}"

    if [[ "${RUN_MAIN_DATA}" == "1" ]]; then
      run_main_data_for_map "${map_name}" "${map_path}" "${mask_resolved}" "${data_tag}"
    fi
    if [[ "${RUN_RESOLUTION_TESTS}" == "1" && " ${RESOLUTION_MAPS} " == *" ${map_name} "* ]]; then
      run_resolution_tests_for_map "${map_name}" "${map_path}" "${mask_resolved}" "${data_tag}"
    fi
    if [[ "${RUN_MI_BIN_TESTS}" == "1" && " ${MI_TEST_MAPS} " == *" ${map_name} "* ]]; then
      run_mi_bin_tests_for_map "${map_name}" "${map_path}" "${mask_resolved}" "${data_tag}"
    fi
    if [[ "${RUN_ALT_AXIS_TESTS}" == "1" && " ${ALT_AXIS_MAPS} " == *" ${map_name} "* ]]; then
      run_alt_axis_tests_for_map "${map_name}" "${map_path}" "${mask_resolved}" "${data_tag}"
    fi
    if [[ "${RUN_MOCK}" == "1" ]]; then
      run_mocks_for_map "${map_name}" "${mask_resolved}" "${data_tag}" || true
    fi
    if [[ "${RUN_GLOBAL_REEVAL}" == "1" ]]; then
      run_global_reeval_for_map "${map_name}" "${map_path}" "${mask_resolved}" "${data_tag}" || true
    fi
  done

  log "Kész. OUT_BASE=${OUT_BASE}"
}

main "$@"
