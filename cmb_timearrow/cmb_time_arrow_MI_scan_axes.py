#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cmb_time_arrow_MI_scan_axes.py

Axis-scan “time-arrow test” on a CMB map based on mutual information (MI)
(CPU- and memory-efficient version).

Idea:

- Read a CMB map (e.g. Planck SMICA) and an optional mask.
- Optionally degrade to a smaller NSIDE (e.g. 256).
- Apply low-pass filtering up to several ℓ_max values to build a scale sequence:
    X_0(ℓ_max[0]), X_1(ℓ_max[1]), ..., X_{k-1}(ℓ_max[k-1]).
- For adjacent scale pairs:
    (ℓ_i, ℓ_{i+1}) for each i=0..k-2.
- Estimate MI(X_i, X_{i+1}) from value pairs on the same pixels:
    * globally (full sky),
    * per hemisphere (A, B) for every axis.
- Define:
    ΔMI_i = | MI_A(ℓ_i, ℓ_{i+1}) - MI_B(ℓ_i, ℓ_{i+1}) |
  → this is a scale-indexed sequence (i=0..k-2).
- Compute a monotonicity score for this sequence (−1..+1).
- Repeat for many random axes and, optionally, for a COS axis.
- Evaluate where the absolute monotonicity of the COS axis (|mono|) falls in the
  percentile distribution of absolute monotonicities from the random axes.

Output:

- JSON containing:
  - lmax_grid and lmax_pairs (e.g. [ [8,16], [16,24], ... ])
  - MI_full_pairs: global MI(ℓ_i, ℓ_{i+1})
  - monotonicity_MI_full_pairs
  - per axis:
    * lon, lat, is_cos_axis
    * delta_MI_pairs (ΔMI_i sequence)
    * monotonicity_delta_MI
    * abs_monotonicity_delta_MI
  - cos_axis block:
    * monotonicity_delta_MI
    * abs_monotonicity_delta_MI
    * percentile_abs_vs_random
  - time_arrow_MI_assessment block:
    * local axis-rank diagnostic (not a global p-value)
  - time_arrow_MI_global_assessment block (optional):
    * global p-values estimated from mock simulations
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np

try:
    import glob
except Exception:
    glob = None

try:
    import healpy as hp
except Exception:
    hp = None

try:
    from astropy.io import fits
except Exception:
    fits = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def require_healpy() -> None:
    if hp is None:
        raise RuntimeError(
            "The 'healpy' package is required to run this script. "
            "Install with:  pip install healpy"
        )


def _robust_read_healpix_map(path: str, field: int = 0, label: str = "map") -> np.ndarray:
    """Read a HEALPix FITS map with robust HDU fallback."""
    require_healpy()

    attempts = [(1, field), (0, field)]
    last_err = None
    for hdu, fld in attempts:
        try:
            print(f"[info] {label.capitalize()} loading: {path} [field={fld}, hdu={hdu}]", flush=True)
            return hp.read_map(path, field=fld, hdu=hdu, dtype=float)
        except Exception as e:
            last_err = e

    if fits is not None:
        try:
            with fits.open(path, memmap=True) as hdul:
                candidates = []
                for idx, hdu_obj in enumerate(hdul):
                    xtension = hdu_obj.header.get("XTENSION", "PRIMARY")
                    if xtension in ("BINTABLE", "IMAGE", "PRIMARY"):
                        candidates.append(idx)
                for hdu in candidates:
                    try:
                        print(f"[warn] Fallback loading: {path} [field={field}, hdu={hdu}]", flush=True)
                        return hp.read_map(path, field=field, hdu=hdu, dtype=float)
                    except Exception as e:
                        last_err = e
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to read the {label} file with healpy: {path}. "
        f"Last error: {last_err}"
    )


def load_map(path: str, field: int = 0) -> Tuple[np.ndarray, int]:
    """Read a CMB map from FITS and determine NSIDE."""
    m = _robust_read_healpix_map(path, field=field, label="map")
    nside = hp.get_nside(m)
    print(f"[info] Original NSIDE={nside}, npix={hp.nside2npix(nside)}", flush=True)
    return m, nside


def load_mask(path: str, field: int = 0) -> np.ndarray:
    """Read a mask from FITS with robust HDU fallback."""
    m = _robust_read_healpix_map(path, field=field, label="mask")
    print(f"[info] Original mask NSIDE={hp.get_nside(m)}", flush=True)
    return m


def degrade_to_work_nside(
    m: np.ndarray,
    mask: Optional[np.ndarray],
    work_nside: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Align the map (and optional mask) to work_nside.

    Rules:
    - a mapet nem degrade upward;
    - if the map is already at work_nside, still align the mask separately if
      it is on a different NSIDE;
    - if the mask is at finer resolution, degrade it to work_nside;
    - if the mask is coarser, do not upscale it; leave it unchanged and raise
      an explicit error later if a size mismatch remains.
    """
    require_healpy()
    nside_orig = hp.get_nside(m)

    # Map handling
    if nside_orig == work_nside:
        print(f"[info] NSIDE is already {nside_orig}; not degrading the map.", flush=True)
        m_d = m
    elif nside_orig < work_nside:
        print(
            f"[warn] Original NSIDE={nside_orig} < work_nside={work_nside}, "
            "a mapet nem degrade upward.",
            flush=True,
        )
        m_d = m
    else:
        print(f"[info] Map degradation NSIDE={nside_orig} → NSIDE={work_nside}", flush=True)
        m_d = hp.ud_grade(m, nside_out=work_nside, pess=False)

    # Handle the mask separately
    mask_d = None
    if mask is not None:
        mask_nside = hp.get_nside(mask)
        target_nside = hp.get_nside(m_d)
        if mask_nside == target_nside:
            print(f"[info] Mask NSIDE is already {mask_nside}; not degrading.", flush=True)
            mask_d = mask
        elif mask_nside > target_nside:
            print(
                f"[info] Mask degradation NSIDE={mask_nside} → NSIDE={target_nside}",
                flush=True,
            )
            mask_d = hp.ud_grade(mask, nside_out=target_nside, pess=False)
            mask_d = np.clip(mask_d, 0.0, 1.0)
        else:
            print(
                f"[warn] A mask NSIDE={mask_nside} < map NSIDE={target_nside}; "
                "the mask will not be upscaled.",
                flush=True,
            )
            mask_d = mask

    return m_d, mask_d


def standardize_map(m: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Z-score normalization: (T - mean) / std, using only mask-allowed pixels."""
    if mask is not None:
        good = mask > 0.5
        vals = m[good]
    else:
        vals = m
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("No valid pixels remain in the map after masking.")
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    if not np.isfinite(sigma) or sigma == 0.0:
        print("[warn] Map standard deviation is zero or non-finite; subtracting only the mean.", flush=True)
        return m - mu
    print(f"[info] Standardization: mean={mu:.4e}, std={sigma:.4e}", flush=True)
    return (m - mu) / sigma


def lowpass_map(m: np.ndarray, lmax: int) -> np.ndarray:
    """Low-pass: map2alm lmax-ig, majd alm2map ugyanarra az NSIDE-ra."""
    require_healpy()
    nside = hp.get_nside(m)
    print(f"[info] Low-pass filtering: ℓ_max={lmax}", flush=True)
    alm = hp.map2alm(m, lmax=lmax)
    m_lp = hp.alm2map(alm, nside=nside, lmax=lmax)
    return m_lp


def precompute_pix_vectors(nside: int) -> np.ndarray:
    """Pixel direction vectors for a given NSIDE, shape=(3,npix)."""
    require_healpy()
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix, dtype=int)
    vx, vy, vz = hp.pix2vec(nside, ipix)
    return np.vstack((vx, vy, vz))


def build_hemi_masks_for_axis(
    nside: int,
    axis_lon: float,
    axis_lat: float,
    axis_coords: str,
    pix_vecs: np.ndarray,
    base_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hemisphere masks (A, B) for a given axis.

    Coordinate system currently supported:
      - 'gal' (galactic), assuming the map uses the same system.
    """
    require_healpy()
    axis_coords = axis_coords.lower()
    if axis_coords not in ("gal",):
        raise ValueError("axis_coords can only be 'gal' in this version.")

    theta = np.radians(90.0 - axis_lat)
    phi = np.radians(axis_lon)
    n_axis = np.asarray(hp.ang2vec(theta, phi))

    dotp = n_axis[0] * pix_vecs[0] + n_axis[1] * pix_vecs[1] + n_axis[2] * pix_vecs[2]
    hemiA = (dotp >= 0.0).astype(float)
    hemiB = (dotp < 0.0).astype(float)

    if base_mask is not None:
        bm = (base_mask > 0.5).astype(float)
        hemiA *= bm
        hemiB *= bm

    return hemiA, hemiB


def masked_pixels(m: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Return masked pixels as a 1D vector."""
    if mask is None:
        return m.reshape(-1)
    good = mask > 0.5
    return m[good].reshape(-1)


def discrete_entropy(x: np.ndarray, bins: int = 64) -> float:
    """Discrete Shannon entropy from a 1D histogram of pixel values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    hist, edges = np.histogram(x, bins=bins, density=False)
    if not np.any(hist > 0):
        return float("nan")
    p = hist.astype(float) / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mutual_information_2d(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Bivariate mutual information (MI) from a 2D histogram.

    I(X;Y) = sum_{i,j} p(x_i, y_j) log( p(x_i, y_j) / (p(x_i) p(y_j)) )
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return float("nan")

    hist2d, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    if not np.any(hist2d > 0):
        return float("nan")

    pxy = hist2d.astype(float) / np.sum(hist2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # points where pxy>0
    mask = pxy > 0
    pxy_nz = pxy[mask]

    px_exp = px[:, np.newaxis]
    py_exp = py[np.newaxis, :]
    denom = px_exp * py_exp
    denom_nz = denom[mask]

    # Keep only entries where denom>0
    valid = denom_nz > 0
    if not np.any(valid):
        return float("nan")

    pxy_v = pxy_nz[valid]
    denom_v = denom_nz[valid]

    return float(np.sum(pxy_v * np.log(pxy_v / denom_v)))


def monotonicity_score(values: np.ndarray) -> float:
    """
    Time-arrow-like directionality score (−1..+1):
      +1 ≈ increasing trend, −1 ≈ decreasing, 0 ≈ no clear trend.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan")
    idx = np.arange(n, dtype=float)
    v_mean = float(np.mean(v))
    i_mean = float(np.mean(idx))
    num = float(np.sum((idx - i_mean) * (v - v_mean)))
    den = float(np.sqrt(np.sum((idx - i_mean) ** 2) * np.sum((v - v_mean) ** 2)))
    if den == 0.0:
        return float("nan")
    return num / den


def sample_random_axes_gal(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n random axes in galactic coordinates (uniform on the sphere).
    Returns: shape=(n,2): lon[deg], lat[deg]
    """
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    lat = np.arcsin(u)
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)
    return np.vstack((lon_deg, lat_deg)).T


def interpret_time_arrow_mi_result(
    cos_abs_mono: Optional[float],
    rnd_abs_monos: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Heuristic local axis-rank interpretation for the COS-axis MI time-arrow indicator.

    Here we inspect absolute monotonicity (|mono|):

      - large |mono| → strongly ordered behavior,
      - small |mono| → noisier, less trend-like behavior.

    Relative to the distribution of absolute monotonicities for random axes,
    determine the percentile position of the COS axis.

    Returns:
      {
        "level": "none" | "weak" | "strong" | "undetermined",
        "message": str,
        "cos_abs_mono": float or None,
        "percentile_abs_vs_random": float or None
      }
    """
    if cos_abs_mono is None or not np.isfinite(cos_abs_mono):
        return {
            "level": "undetermined",
            "message": (
                "The MI-based time-arrow statistic for the COS axis could not be evaluated reliably in this run "
                "(missing or non-finite value)."
            ),
            "is_global_pvalue": False,
            "cos_abs_mono": None,
            "percentile_abs_vs_random": None,
        }

    if rnd_abs_monos is None:
        return {
            "level": "undetermined",
            "message": (
                "No MI time-arrow values are available for random axes, so the COS result "
                "cannot be compared against them."
            ),
            "is_global_pvalue": False,
            "cos_abs_mono": float(cos_abs_mono),
            "percentile_abs_vs_random": None,
        }

    rnd_abs = np.asarray(rnd_abs_monos, dtype=float)
    rnd_abs = rnd_abs[np.isfinite(rnd_abs)]
    if rnd_abs.size == 0:
        return {
            "level": "undetermined",
            "message": (
                "The MI time-arrow values for the random axes were invalid, "
                "so the percentile position of the COS axis cannot be determined."
            ),
            "is_global_pvalue": False,
            "cos_abs_mono": float(cos_abs_mono),
            "percentile_abs_vs_random": None,
        }

    # Percentilis: P(|mono_random| <= |mono_COS|)
    count_le = int(np.count_nonzero(rnd_abs <= cos_abs_mono))
    percentile = (count_le + 0.5) / (rnd_abs.size + 1.0)

    # Heuristic thresholds:
    # - strong: abs-mono percentile > 0.95
    # - weak:   0.80 < percentile ≤ 0.95
    # - none:   percentile ≤ 0.80
    if percentile > 0.95:
        level = "strong"
        msg = (
            "A strong, unusual MI-based time-arrow-like pattern is visible in this statistic: "
            "the absolute monotonicity of the COS axis exceeds that of the overwhelming majority of random axes (>95%)."
        )
    elif percentile > 0.80:
        level = "weak"
        msg = (
            "A weak or moderate MI-based time-arrow-like deviation is visible in this statistic: "
            "the absolute monotonicity of the COS axis lies within the upper ~20% of random axes."
        )
    else:
        level = "none"
        msg = (
            "No strong, significant time-arrow-like deviation was found in this MI-based statistic: "
            "the absolute monotonicity of the COS axis is not unusual relative to the distribution of random axes."
        )

    return {
        "level": level,
        "message": msg + " This is a local axis-rank diagnostic on the same map, not a global p-value.",
        "cos_abs_mono": float(cos_abs_mono),
        "percentile_abs_vs_random": float(percentile),
        "is_global_pvalue": False,
    }




def empirical_tail_pvalue_ge(obs: Optional[float], null: Optional[np.ndarray]) -> Optional[float]:
    """Empirical upper-tail p-value: P_null(X >= obs)."""
    if obs is None or not np.isfinite(obs):
        return None
    if null is None:
        return None
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    count_ge = int(np.count_nonzero(arr >= float(obs)))
    return float((count_ge + 1.0) / (arr.size + 1.0))

def empirical_percentile_le(obs: Optional[float], null: Optional[np.ndarray]) -> Optional[float]:
    """Empirikus percentile: P_null(X <= obs)."""
    if obs is None or not np.isfinite(obs):
        return None
    if null is None:
        return None
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    count_le = int(np.count_nonzero(arr <= float(obs)))
    return float((count_le + 0.5) / (arr.size + 1.0))


def safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_mock_result_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal validation for the look-elsewhere null distribution."""
    errors: List[str] = []
    scan_max = _extract_random_scan_max_from_result(data)
    cos_abs = _extract_cos_abs_from_result(data)
    if scan_max is None or not np.isfinite(scan_max):
        errors.append('missing_random_scan_max')
    mode = data.get('mode')
    if mode is not None and mode not in ('data', 'mock', 'postprocess'):
        errors.append('invalid_mode')
    return {
        'ok_for_scan_null': len(errors) == 0 and scan_max is not None and np.isfinite(scan_max),
        'has_cos_abs': cos_abs is not None and np.isfinite(cos_abs),
        'scan_max': float(scan_max) if scan_max is not None and np.isfinite(scan_max) else None,
        'cos_abs': float(cos_abs) if cos_abs is not None and np.isfinite(cos_abs) else None,
        'errors': errors,
    }


def _extract_random_scan_max_from_result(data: Dict[str, Any]) -> Optional[float]:
    if isinstance(data.get('random_axes_abs_monotonicity_max'), (int, float)):
        val = float(data['random_axes_abs_monotonicity_max'])
        return val if np.isfinite(val) else None
    axes = data.get('axes')
    if isinstance(axes, list):
        vals = []
        for ax in axes:
            try:
                if not bool(ax.get('is_cos_axis', False)):
                    v = float(ax.get('abs_monotonicity_delta_MI', float('nan')))
                    if np.isfinite(v):
                        vals.append(v)
            except Exception:
                pass
        if vals:
            return float(np.max(vals))
    return None


def _extract_cos_abs_from_result(data: Dict[str, Any]) -> Optional[float]:
    ta = data.get('time_arrow_MI_assessment')
    if isinstance(ta, dict):
        val = ta.get('cos_abs_mono')
        if isinstance(val, (int, float)):
            val = float(val)
            return val if np.isfinite(val) else None
    axes = data.get('axes')
    if isinstance(axes, list):
        for ax in axes:
            try:
                if bool(ax.get('is_cos_axis', False)):
                    v = float(ax.get('abs_monotonicity_delta_MI', float('nan')))
                    if np.isfinite(v):
                        return v
            except Exception:
                pass
    return None


def load_global_null_results(json_paths: Sequence[str]) -> Dict[str, Any]:
    mock_scan_max: List[float] = []
    mock_cos_abs: List[float] = []
    invalid_files: List[str] = []
    scan_only_files: List[str] = []
    total_files = 0

    for path_str in json_paths:
        total_files += 1
        path = Path(path_str)
        data = safe_json_load(path)
        if data is None:
            invalid_files.append(str(path))
            continue
        info = validate_mock_result_dict(data)
        if info['ok_for_scan_null']:
            mock_scan_max.append(float(info['scan_max']))
        else:
            invalid_files.append(str(path))
            continue
        if info['has_cos_abs']:
            mock_cos_abs.append(float(info['cos_abs']))
        else:
            scan_only_files.append(str(path))

    return {
        'mock_scan_max': np.asarray(mock_scan_max, dtype=float),
        'mock_cos_abs': np.asarray(mock_cos_abs, dtype=float),
        'n_total_files': int(total_files),
        'n_invalid_files': int(len(invalid_files)),
        'invalid_files': invalid_files,
        'n_scan_only_files': int(len(scan_only_files)),
        'scan_only_files': scan_only_files,
    }


def build_global_assessment(
    current_scan_max_random: Optional[float],
    current_cos_abs_mono: Optional[float],
    mock_null: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if mock_null is None:
        return {
            'available': False,
            'message': (
                'A global p-value cannot be computed because no null distribution from mock simulations '
                'was provided. The local axis-rank diagnostic does not by itself replace the '
                'look-elsewhere correction.'
            ),
            'scan_global_pvalue_vs_mock_max': None,
            'scan_global_percentile_vs_mock_max': None,
            'cos_fixed_axis_pvalue_vs_mock_cos': None,
            'cos_fixed_axis_percentile_vs_mock_cos': None,
            'n_mock_scan': 0,
            'n_mock_cos': 0,
            'n_total_files': 0,
            'n_invalid_files': 0,
            'n_scan_only_files': 0,
            'is_global_pvalue': True,
        }

    scan_null = np.asarray(mock_null.get('mock_scan_max', []), dtype=float)
    scan_null = scan_null[np.isfinite(scan_null)]
    cos_null = np.asarray(mock_null.get('mock_cos_abs', []), dtype=float)
    cos_null = cos_null[np.isfinite(cos_null)]

    scan_p = empirical_tail_pvalue_ge(current_scan_max_random, scan_null)
    scan_pct = empirical_percentile_le(current_scan_max_random, scan_null)
    cos_p = empirical_tail_pvalue_ge(current_cos_abs_mono, cos_null)
    cos_pct = empirical_percentile_le(current_cos_abs_mono, cos_null)

    msg_parts = []
    if scan_p is not None:
        msg_parts.append(
            f'The global p-value for the axis-scan maximum is p≈{scan_p:.4f} '
            f'(null: {scan_null.size} mock maxima).'
        )
    if cos_p is not None:
        msg_parts.append(
            f'The empirical p-value for the fixed COS axis is p≈{cos_p:.4f} '
            f'(null: {cos_null.size} mock COS values).'
        )
    if mock_null.get('n_invalid_files', 0) > 0:
        msg_parts.append(
            f"{mock_null.get('n_invalid_files', 0)} mock files were skipped because they were not usable for the null distribution."
        )
    if mock_null.get('n_scan_only_files', 0) > 0:
        msg_parts.append(
            f"{mock_null.get('n_scan_only_files', 0)} mock files only provided a scan maximum, not a fixed COS-axis value."
        )
    if not msg_parts:
        msg_parts.append(
            'Mock files were provided, but no usable global null distribution could be extracted from them.'
        )

    return {
        'available': True,
        'message': ' '.join(msg_parts),
        'scan_global_pvalue_vs_mock_max': scan_p,
        'scan_global_percentile_vs_mock_max': scan_pct,
        'cos_fixed_axis_pvalue_vs_mock_cos': cos_p,
        'cos_fixed_axis_percentile_vs_mock_cos': cos_pct,
        'n_mock_scan': int(scan_null.size),
        'n_mock_cos': int(cos_null.size),
        'n_total_files': int(mock_null.get('n_total_files', 0)),
        'n_invalid_files': int(mock_null.get('n_invalid_files', 0)),
        'n_scan_only_files': int(mock_null.get('n_scan_only_files', 0)),
        'is_global_pvalue': True,
    }


def find_json_files_from_args(global_null_dir: Optional[str], global_null_jsonl: Optional[str]) -> List[str]:
    paths: List[str] = []
    if global_null_dir:
        d = Path(global_null_dir)
        if d.exists() and d.is_dir():
            paths.extend(sorted(str(x) for x in d.glob('*.json')))
    if global_null_jsonl:
        p = Path(global_null_jsonl)
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                paths.append(line)
    # deduplicate preserving order
    seen = set()
    out = []
    for x in paths:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


@dataclass
class AxisMIResult:
    index: int
    lon_deg: float
    lat_deg: float
    is_cos_axis: bool
    delta_MI_pairs: List[float]
    monotonicity_delta_MI: float
    abs_monotonicity_delta_MI: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CMB MI-based time-arrow test: axis-scan using the monotonicity of ΔMI across ℓ-pairs."
    )
    ap.add_argument("--map", required=True, help="CMB map FITS (e.g. SMICA / SEVEM / NPIPE)")
    ap.add_argument("--map-field", type=int, default=0, help="Which FITS field to read from the map (default: 0, typically T/I).")
    ap.add_argument("--mask", default=None, help="Optional mask FITS")
    ap.add_argument("--mask-field", type=int, default=0, help="Which FITS field to read from the mask (default: 0).")
    ap.add_argument(
        "--work-nside",
        type=int,
        default=256,
        help="Working NSIDE (the map is degraded to this value). Default: 256",
    )
    ap.add_argument(
        "--lmax-grid",
        type=str,
        default="8,16,24,32,48,64,96,128,192,256",
        help="Comma-separated ℓ_max values, e.g. '8,16,24,32,48,64,96,128,192,256'",
    )
    ap.add_argument(
        "--entropy-bins",
        type=int,
        default=64,
        help="DEPRECATED: accepted for compatibility; it does not affect the primary MI statistic in the current version (default: 64)",
    )
    ap.add_argument(
        "--mi-bins",
        type=int,
        default=32,
        help="Number of 2D histogram bins used to compute MI (default: 32)",
    )
    ap.add_argument(
        "--n-axes",
        type=int,
        default=100,
        help="Number of random axes (default: 100)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random generator seed",
    )
    ap.add_argument(
        "--cos-axis",
        type=str,
        default=None,
        help="COS-axis lon,lat in degrees, e.g. '227,-27' (galactic).",
    )
    ap.add_argument(
        "--cos-coords",
        type=str,
        default="gal",
        help="Coordinate system of the COS axis (currently only 'gal').",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="data",
        choices=["data", "mock"],
        help="Run mode: real data or mock simulation. Also stored in the JSON output.",
    )
    ap.add_argument(
        "--mock-id",
        type=str,
        default=None,
        help="Optional mock identifier for the output JSON.",
    )
    ap.add_argument(
        "--global-null-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing JSON outputs from previously run mock simulations. "
            "If provided, the script also estimates a global p-value for the look-elsewhere correction."
        ),
    )
    ap.add_argument(
        "--global-null-jsonl",
        type=str,
        default=None,
        help=(
            "Optional text file with one mock JSON path per line. "
            "Can be used alongside or instead of --global-null-dir."
        ),
    )
    ap.add_argument(
        "--require-global-null-min",
        type=int,
        default=30,
        help="Minimum number of usable mock files required before reporting the global p-value (default: 30).",
    )
    ap.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Optional short tag for the run (e.g. PR4_NPIPE_SEVEM).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="cmb_time_arrow_MI_scan_axes.json",
        help="Output JSON filename",
    )
    args = ap.parse_args()

    # ℓ_max list
    try:
        l_grid: Sequence[int] = [
            int(x) for x in args.lmax_grid.replace(";", ",").split(",") if x.strip() != ""
        ]
    except Exception as e:
        raise SystemExit(f"Invalid --lmax-grid format: {args.lmax_grid!r}  ({e})")

    if len(l_grid) < 3:
        raise SystemExit("At least three ℓ_max values are required so that ℓ-pairs exist.")

    print(f"[info] ℓ_max grid: {l_grid}", flush=True)
    print(f"[info] work_nside={args.work_nside}", flush=True)
    if ('--entropy-bins' in sys.argv) or (args.entropy_bins != 64):
        print(
            f"[warn] --entropy-bins={args.entropy_bins} is a compatibility/deprecated parameter in the current version; "
            "it does not affect the primary MI statistic.",
            flush=True,
        )
    else:
        print("[info] --entropy-bins accepted for compatibility, but not used.", flush=True)
    print(f"[info] MI bin count (2D): {args.mi_bins}", flush=True)
    print(f"[info] Random axes count: {args.n_axes}, seed={args.seed}", flush=True)
    print(f"[info] Mode: {args.mode}", flush=True)
    print(f"[info] map_field={args.map_field}", flush=True)
    if args.mask is not None:
        print(f"[info] mask_field={args.mask_field}", flush=True)
    if args.dataset_tag:
        print(f"[info] dataset_tag={args.dataset_tag}", flush=True)
    if args.mode == "mock" and args.cos_axis is None:
        print(
            "[warn] In mock mode, --cos-axis is not provided; the fixed-COS global null will only be usable for the scan maximum.",
            flush=True,
        )

    rng = np.random.default_rng(args.seed)

    # Map + mask loading
    m_raw, nside_orig = load_map(args.map, field=args.map_field)
    mask_raw = None
    if args.mask is not None:
        mask_raw = load_mask(args.mask, field=args.mask_field)

    # Degrade to work_nside
    m_work, mask_work = degrade_to_work_nside(m_raw, mask_raw, args.work_nside)
    nside = hp.get_nside(m_work)
    print(f"[info] Working NSIDE={nside}, npix={hp.nside2npix(nside)}", flush=True)

    # Normalize mask to the range 0..1
    if mask_work is not None:
        mask_work = np.clip(mask_work, 0.0, 1.0)
        if mask_work.shape != m_work.shape:
            raise RuntimeError(
                f"Mask/map size mismatch: map npix={m_work.size}, mask npix={mask_work.size}. "
                "Different NSIDE values likely remained after preprocessing."
            )

    # Standardized map
    m_std = standardize_map(m_work, mask_work)

    # Pixel direction vectors
    print("[info] Precomputing pixel direction vectors...", flush=True)
    pix_vecs = precompute_pix_vectors(nside)
    print("[info] Pixel direction vectors ready.", flush=True)

    # Axis list: COS + random
    random_axes = sample_random_axes_gal(args.n_axes, rng)
    axes_list: List[Tuple[float, float, bool]] = []

    cos_axis_lon: Optional[float] = None
    cos_axis_lat: Optional[float] = None
    if args.cos_axis is not None:
        try:
            lon_str, lat_str = args.cos_axis.split(",")
            cos_axis_lon = float(lon_str)
            cos_axis_lat = float(lat_str)
        except Exception as e:
            raise SystemExit(f"Invalid --cos-axis format: {args.cos_axis!r}  ({e})")
        print(
            f"[info] COS axis: lon={cos_axis_lon}, lat={cos_axis_lat} [{args.cos_coords}]",
            flush=True,
        )
        axes_list.append((cos_axis_lon, cos_axis_lat, True))

    for i in range(random_axes.shape[0]):
        lon_deg, lat_deg = random_axes[i]
        axes_list.append((float(lon_deg), float(lat_deg), False))

    n_axes_total = len(axes_list)
    print(f"[info] Total axes (COS + random): {n_axes_total}", flush=True)

    # Low-pass maps for every ℓ_max
    maps_lp: List[np.ndarray] = []
    for lmax in l_grid:
        m_lp = lowpass_map(m_std, lmax=lmax)
        maps_lp.append(m_lp)

    # ℓ-pairs (indices and actual values)
    l_pairs: List[Tuple[int, int]] = []
    for i in range(len(l_grid) - 1):
        l_pairs.append((l_grid[i], l_grid[i + 1]))
    print(f"[info] ℓ-pairs for MI: {l_pairs}", flush=True)

    # Global MI for ℓ-pairs (full sky, after masking)
    MI_full_pairs: List[float] = []
    for (l1, l2), i in zip(l_pairs, range(len(l_pairs))):
        m1 = maps_lp[i]
        m2 = maps_lp[i + 1]
        pix1 = masked_pixels(m1, mask_work)
        pix2 = masked_pixels(m2, mask_work)
        MIg = mutual_information_2d(pix1, pix2, bins=args.mi_bins)
        MI_full_pairs.append(MIg)
        print(f"[info] MI_full(ℓ={l1},{l2}) ≈ {MIg:.4f}", flush=True)

    MI_full_arr = np.asarray(MI_full_pairs, dtype=float)
    mono_MI_full = monotonicity_score(MI_full_arr)
    print(f"[info] Monotonicity MI_full_pairs ≈ {mono_MI_full:.3f}", flush=True)

    # Hemisphere masks for every axis (same NSIDE, independent of ℓ)
    hemi_masks: List[Tuple[np.ndarray, np.ndarray]] = []
    for axis_idx, (lon_deg, lat_deg, is_cos) in enumerate(axes_list):
        hemiA, hemiB = build_hemi_masks_for_axis(
            nside=nside,
            axis_lon=lon_deg,
            axis_lat=lat_deg,
            axis_coords="gal",
            pix_vecs=pix_vecs,
            base_mask=mask_work,
        )
        hemi_masks.append((hemiA, hemiB))
        tag = "COS" if is_cos else "rnd"
        print(
            f"[info] Hemisphere masks built for axis [axis {axis_idx} | {tag}] "
            f"lon={lon_deg:.2f}, lat={lat_deg:.2f}",
            flush=True,
        )

    # ΔMI(ℓ-pairs) for every axis
    delta_MI_per_axis: List[List[float]] = [[] for _ in range(n_axes_total)]
    axis_results: List[AxisMIResult] = []

    for axis_idx, ((lon_deg, lat_deg, is_cos), (hemiA, hemiB)) in enumerate(
        zip(axes_list, hemi_masks)
    ):
        dMI_list: List[float] = []
        for pair_idx, (l1, l2) in enumerate(l_pairs):
            m1 = maps_lp[pair_idx]
            m2 = maps_lp[pair_idx + 1]

            pix1_A = masked_pixels(m1, hemiA)
            pix2_A = masked_pixels(m2, hemiA)
            pix1_B = masked_pixels(m1, hemiB)
            pix2_B = masked_pixels(m2, hemiB)

            MI_A = mutual_information_2d(pix1_A, pix2_A, bins=args.mi_bins)
            MI_B = mutual_information_2d(pix1_B, pix2_B, bins=args.mi_bins)

            if np.isfinite(MI_A) and np.isfinite(MI_B):
                dMI = abs(MI_A - MI_B)
            else:
                dMI = float("nan")

            dMI_list.append(dMI)

        dMI_arr = np.asarray(dMI_list, dtype=float)
        mono_dMI = monotonicity_score(dMI_arr)
        abs_mono_dMI = float(abs(mono_dMI)) if np.isfinite(mono_dMI) else float("nan")

        axis_results.append(
            AxisMIResult(
                index=axis_idx,
                lon_deg=float(lon_deg),
                lat_deg=float(lat_deg),
                is_cos_axis=bool(is_cos),
                delta_MI_pairs=[float(x) for x in dMI_arr.tolist()],
                monotonicity_delta_MI=float(mono_dMI),
                abs_monotonicity_delta_MI=abs_mono_dMI,
            )
        )

        tag = "COS" if is_cos else "rnd"
        print(
            f"[axis {axis_idx:3d} | {tag}] lon={lon_deg:7.2f}, lat={lat_deg:7.2f}, "
            f"mono(ΔMI)≈{mono_dMI:.3f}, |mono(ΔMI)|≈{abs_mono_dMI:.3f}",
            flush=True,
        )

    # COS-axis absolute monotonicity and random-axis distribution
    cos_abs_mono: Optional[float] = None
    rnd_abs_monos: Optional[np.ndarray] = None

    cos_minos = [
        ar.abs_monotonicity_delta_MI
        for ar in axis_results
        if ar.is_cos_axis and np.isfinite(ar.abs_monotonicity_delta_MI)
    ]
    if len(cos_minos) > 0:
        cos_abs_mono = float(cos_minos[0])

    rnd_abs_list = [
        ar.abs_monotonicity_delta_MI
        for ar in axis_results
        if (not ar.is_cos_axis) and np.isfinite(ar.abs_monotonicity_delta_MI)
    ]
    if len(rnd_abs_list) > 0:
        rnd_abs_monos = np.asarray(rnd_abs_list, dtype=float)

    random_axes_abs_monotonicity_max: Optional[float] = None
    if rnd_abs_monos is not None and rnd_abs_monos.size > 0:
        random_axes_abs_monotonicity_max = float(np.max(rnd_abs_monos))

    scan_abs_monotonicity_max_all_axes: Optional[float] = None
    all_abs = [
        ar.abs_monotonicity_delta_MI
        for ar in axis_results
        if np.isfinite(ar.abs_monotonicity_delta_MI)
    ]
    if all_abs:
        scan_abs_monotonicity_max_all_axes = float(np.max(all_abs))

    time_arrow_assessment = interpret_time_arrow_mi_result(cos_abs_mono, rnd_abs_monos)

    null_paths = find_json_files_from_args(args.global_null_dir, args.global_null_jsonl)
    mock_null = load_global_null_results(null_paths) if len(null_paths) > 0 else None
    if mock_null is not None and int(mock_null.get('n_invalid_files', 0)) > 0:
        print(
            f"[warn] {mock_null.get('n_invalid_files', 0)} mock JSON files were excluded from the global null for validation reasons.",
            flush=True,
        )
    if mock_null is not None:
        n_scan = int(np.asarray(mock_null.get('mock_scan_max', np.asarray([]))).size)
        if n_scan < int(args.require_global_null_min):
            print(
                f"[warn] The global null contains only {n_scan} usable mock files; this is smaller than the recommended minimum of {args.require_global_null_min}.",
                flush=True,
            )
    global_assessment = build_global_assessment(
        current_scan_max_random=random_axes_abs_monotonicity_max,
        current_cos_abs_mono=cos_abs_mono,
        mock_null=mock_null,
    )

    if time_arrow_assessment["percentile_abs_vs_random"] is not None:
        print(
            f"[info] COS |mono(ΔMI)| ≈ {time_arrow_assessment['cos_abs_mono']:.3f}, "
            f"percentile(abs-mono) ≈ "
            f"{100.0 * time_arrow_assessment['percentile_abs_vs_random']:.1f}%",
            flush=True,
        )

    print(
        f"[info] MI time-arrow assessment: {time_arrow_assessment['message']} "
        f"(szint={time_arrow_assessment['level']})",
        flush=True,
    )
    print(f"[info] Global assessment: {global_assessment['message']}", flush=True)

    # Write JSON
    out_data: Dict[str, Any] = {
        "mode": args.mode,
        "mock_id": args.mock_id,
        "dataset_tag": args.dataset_tag,
        "map": args.map,
        "map_field": int(args.map_field),
        "mask": args.mask,
        "mask_field": int(args.mask_field) if args.mask is not None else None,
        "nside_orig": int(nside_orig),
        "nside_work": int(nside),
        "lmax_grid": [int(x) for x in l_grid],
        "lmax_pairs": [[int(a), int(b)] for (a, b) in l_pairs],
        "entropy_bins": int(args.entropy_bins),
        "entropy_bins_used": False,
        "deprecated_args": {
            "entropy_bins": {
                "present": ("--entropy-bins" in sys.argv) or (args.entropy_bins != 64),
                "used_in_primary_statistic": False,
                "message": "The primary MI statistic in this pipeline uses only the --mi-bins parameter. --entropy-bins is retained for compatibility.",
            }
        },
        "mi_bins": int(args.mi_bins),
        "n_axes_total": int(n_axes_total),
        "n_axes_random": int(args.n_axes),
        "seed": int(args.seed),
        "MI_full_pairs": MI_full_arr.tolist(),
        "monotonicity_MI_full_pairs": float(mono_MI_full),
        "axes": [asdict(ar) for ar in axis_results],
        "random_axes_abs_monotonicity_max": random_axes_abs_monotonicity_max,
        "scan_abs_monotonicity_max_all_axes": scan_abs_monotonicity_max_all_axes,
        "global_null_sources": null_paths,
        "global_null_min_recommended": int(args.require_global_null_min),
        "cos_axis": {
            "lon_deg": float(cos_axis_lon) if cos_axis_lon is not None else None,
            "lat_deg": float(cos_axis_lat) if cos_axis_lat is not None else None,
            "coords": args.cos_coords,
        },
        "time_arrow_MI_assessment": time_arrow_assessment,
        "time_arrow_MI_global_assessment": global_assessment,
        "global_null_validation": None if mock_null is None else {
            "n_total_files": int(mock_null.get("n_total_files", 0)),
            "n_invalid_files": int(mock_null.get("n_invalid_files", 0)),
            "invalid_files": mock_null.get("invalid_files", []),
            "n_scan_only_files": int(mock_null.get("n_scan_only_files", 0)),
            "scan_only_files": mock_null.get("scan_only_files", []),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[info] Output saved: {out_path}", flush=True)
    if global_assessment.get('available', False) and global_assessment.get('scan_global_pvalue_vs_mock_max') is not None:
        print(f"[info] Global p(scan max) ≈ {global_assessment['scan_global_pvalue_vs_mock_max']:.4f}", flush=True)


if __name__ == "__main__":
    main()
