#!/usr/bin/env python3
"""
Cross-probe fixed-axis CMB MI analysis driven by a Pantheon+ discovery axis.

What this script does
---------------------
1. Reads a Pantheon+ axis-scan JSON produced by ``cos_pantheon_axis_scan.py``
   and selects the best axis according to a chosen Pantheon statistic
   (default: ``abs_delta_q0``).
2. Converts that axis from ICRS to Galactic coordinates.
3. On a CMB temperature map, evaluates the hemispherical asymmetry of the
   cross-scale mutual information sequence

       ΔMI_i = | MI_A(ℓ_i, ℓ_{i+1}) - MI_B(ℓ_i, ℓ_{i+1}) |,

   along this *fixed* axis only.
4. Supports two MI estimators:
   - ``hist`` : 2D histogram estimator
   - ``knn``  : Kraskov–Stögbauer–Grassberger style kNN estimator
5. Optionally builds null distributions from phase-randomized maps that keep the
   CMB power spectrum amplitude approximately fixed while destroying phase
   coherence.
6. Optionally performs a Bayes model comparison on the fixed-axis ΔMI curve
   (constant / linear / negative-slope linear / quadratic).

Important methodological note
-----------------------------
This script implements a *cross-probe follow-up* strategy, not a magic
"smoking gun" machine. If the Pantheon axis is itself selected by scanning many
axes, then the Pantheon side carries its own trials factor. The point is that
CMB is no longer re-scanned for discovery; it is tested conditionally on an
axis learned from an independent probe. That is stronger than a pure CMB
axis-scan, but it is still not the same thing as a globally pre-registered,
zero-tuning discovery claim.

Practical note on kNN MI
------------------------
A full-sky NSIDE=256 hemisphere contains far too many pixels for an exact kNN MI
estimate to be cheap. Therefore the kNN estimator here uses a configurable
subsample of paired pixels per hemisphere and per ℓ-pair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import healpy as hp
except Exception:
    hp = None

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception:
    SkyCoord = None
    u = None

try:
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    from scipy.stats import norm
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required. Install with: pip install scipy") from exc

# dynesty is optional; when unavailable and --run-bayes is requested, Bayes fitting is skipped with a warning.
try:
    import dynesty
except Exception:
    dynesty = None

# matplotlib is optional; we only use it when plotting is requested.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def log(msg: str) -> None:
    print(f"[info] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)


def require_healpy() -> None:
    if hp is None:
        raise RuntimeError("healpy is required. Install with: pip install healpy")



@dataclass
class AxisChoice:
    source: str
    pantheon_statistic: Optional[str]
    icrs_ra_deg: float
    icrs_dec_deg: float
    gal_lon_deg: float
    gal_lat_deg: float
    pantheon_axis_kind: Optional[str] = None
    pantheon_axis_id: Optional[int] = None
    pantheon_axis_value: Optional[float] = None


@dataclass
class FixedAxisResult:
    lmax_grid: List[int]
    lmax_pairs: List[List[int]]
    delta_mi_pairs: List[float]
    mi_hemi_a_pairs: List[float]
    mi_hemi_b_pairs: List[float]
    monotonicity_delta_mi: float
    abs_monotonicity_delta_mi: float
    linear_slope_delta_mi: float


@dataclass
class NullSummary:
    n_valid: int
    null_mode: str
    p_mono_decreasing: Optional[float]
    p_abs_mono: Optional[float]
    p_slope_decreasing: Optional[float]
    obs_mono: float
    obs_abs_mono: float
    obs_slope: float
    null_mono_mean: Optional[float]
    null_abs_mono_mean: Optional[float]
    null_slope_mean: Optional[float]


# -----------------------------------------------------------------------------
# Coordinates and axis handling
# -----------------------------------------------------------------------------

def require_astropy_for_coords() -> None:
    if SkyCoord is None or u is None:
        raise RuntimeError(
            "astropy is required for ICRS/Galactic coordinate conversion. "
            "Install with: pip install astropy"
        )


def icrs_to_gal(ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    require_astropy_for_coords()
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g = c.galactic
    return float(g.l.deg), float(g.b.deg)


def gal_to_icrs(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    require_astropy_for_coords()
    c = SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame="galactic")
    i = c.icrs
    return float(i.ra.deg), float(i.dec.deg)


def load_axis_from_pantheon_json(path: str, statistic: str) -> AxisChoice:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    axes = data.get("axis_results")
    if not isinstance(axes, list) or len(axes) == 0:
        raise RuntimeError(f"No usable 'axis_results' list in Pantheon JSON: {path}")

    best_entry = None
    best_value = -np.inf
    for entry in axes:
        try:
            value = float(entry[statistic])
        except Exception:
            continue
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_entry = entry

    if best_entry is None:
        raise RuntimeError(f"Could not find a finite Pantheon axis for statistic '{statistic}'.")

    ra_deg = float(best_entry["lon_deg"])
    dec_deg = float(best_entry["lat_deg"])
    gal_lon, gal_lat = icrs_to_gal(ra_deg, dec_deg)

    return AxisChoice(
        source="pantheon_best_axis",
        pantheon_statistic=statistic,
        icrs_ra_deg=ra_deg,
        icrs_dec_deg=dec_deg,
        gal_lon_deg=gal_lon,
        gal_lat_deg=gal_lat,
        pantheon_axis_kind=best_entry.get("kind"),
        pantheon_axis_id=int(best_entry["axis_id"]) if "axis_id" in best_entry else None,
        pantheon_axis_value=float(best_value),
    )


def load_manual_axis(lon_deg: float, lat_deg: float, coords: str) -> AxisChoice:
    coords = coords.lower()
    if coords == "gal":
        gal_lon, gal_lat = float(lon_deg), float(lat_deg)
        ra_deg, dec_deg = gal_to_icrs(gal_lon, gal_lat)
    elif coords == "icrs":
        ra_deg, dec_deg = float(lon_deg), float(lat_deg)
        gal_lon, gal_lat = icrs_to_gal(ra_deg, dec_deg)
    else:
        raise ValueError("axis coordinates must be 'gal' or 'icrs'.")

    return AxisChoice(
        source="manual_axis",
        pantheon_statistic=None,
        icrs_ra_deg=ra_deg,
        icrs_dec_deg=dec_deg,
        gal_lon_deg=gal_lon,
        gal_lat_deg=gal_lat,
    )


# -----------------------------------------------------------------------------
# Map I/O and preprocessing
# -----------------------------------------------------------------------------

def robust_read_map(path: str, field: int = 0, label: str = "map") -> np.ndarray:
    require_healpy()
    attempts = [(1, field), (0, field)]
    last_err: Optional[Exception] = None
    for hdu, fld in attempts:
        try:
            log(f"Loading {label}: {path} [field={fld}, hdu={hdu}]")
            return hp.read_map(path, field=fld, hdu=hdu, dtype=float)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read {label} file: {path}. Last error: {last_err}")


def degrade_to_work_nside(
    m: np.ndarray,
    mask: Optional[np.ndarray],
    work_nside: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    require_healpy()
    nside_map = hp.get_nside(m)
    if nside_map > work_nside:
        log(f"Degrading map NSIDE {nside_map} -> {work_nside}")
        m_out = hp.ud_grade(m, nside_out=work_nside, pess=False)
    elif nside_map == work_nside:
        m_out = m
    else:
        warn(f"Map NSIDE={nside_map} < requested work_nside={work_nside}; keeping native map NSIDE.")
        m_out = m

    mask_out = None
    if mask is not None:
        target_nside = hp.get_nside(m_out)
        nside_mask = hp.get_nside(mask)
        if nside_mask > target_nside:
            log(f"Degrading mask NSIDE {nside_mask} -> {target_nside}")
            mask_out = hp.ud_grade(mask, nside_out=target_nside, pess=False)
            mask_out = np.clip(mask_out, 0.0, 1.0)
        elif nside_mask == target_nside:
            mask_out = np.clip(mask, 0.0, 1.0)
        else:
            raise RuntimeError(
                f"Mask NSIDE={nside_mask} is coarser than map NSIDE={target_nside}; not upscaling automatically."
            )
    return m_out, mask_out


def standardize_map(m: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        good = np.isfinite(m)
    else:
        good = (mask > 0.5) & np.isfinite(m)
    vals = m[good]
    if vals.size == 0:
        raise RuntimeError("No valid map pixels remain after masking.")
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("Map standard deviation is non-finite or zero.")
    log(f"Standardizing map on allowed pixels: mean={mu:.4e}, std={sigma:.4e}")
    return (m - mu) / sigma


def build_lowpass_maps_from_alm(alm_full: np.ndarray, l_grid: Sequence[int], nside: int) -> List[np.ndarray]:
    require_healpy()
    lmax_full = hp.Alm.getlmax(alm_full.size)
    out: List[np.ndarray] = []
    for lmax in l_grid:
        fl = np.zeros(lmax_full + 1, dtype=float)
        fl[: min(lmax, lmax_full) + 1] = 1.0
        alm_lp = hp.almxfl(alm_full, fl)
        m_lp = hp.alm2map(alm_lp, nside=nside, lmax=lmax_full, verbose=False)
        out.append(np.asarray(m_lp, dtype=float))
    return out


def phase_randomize_alm(alm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    require_healpy()
    out = np.array(alm, dtype=np.complex128, copy=True)
    lmax = hp.Alm.getlmax(out.size)
    ell, emm = hp.Alm.getlm(lmax)
    mask = emm > 0
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(mask)))
    out[mask] *= np.exp(1j * phases)
    # m=0 coefficients remain untouched to preserve the real-valued constraint.
    return out


# -----------------------------------------------------------------------------
# Hemisphere masks and MI estimators
# -----------------------------------------------------------------------------

def precompute_pix_vectors(nside: int) -> np.ndarray:
    require_healpy()
    ipix = np.arange(hp.nside2npix(nside), dtype=int)
    vx, vy, vz = hp.pix2vec(nside, ipix)
    return np.vstack((vx, vy, vz))


def hemisphere_pixel_indices(
    nside: int,
    lon_deg: float,
    lat_deg: float,
    pix_vecs: np.ndarray,
    base_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    require_healpy()
    theta = np.radians(90.0 - lat_deg)
    phi = np.radians(lon_deg)
    axis_vec = np.asarray(hp.ang2vec(theta, phi), dtype=float)
    dots = axis_vec[0] * pix_vecs[0] + axis_vec[1] * pix_vecs[1] + axis_vec[2] * pix_vecs[2]
    hemi_a = dots >= 0.0
    hemi_b = ~hemi_a
    if base_mask is not None:
        good = base_mask > 0.5
        hemi_a &= good
        hemi_b &= good
    idx_a = np.flatnonzero(hemi_a)
    idx_b = np.flatnonzero(hemi_b)
    return idx_a, idx_b


def sample_pairs(
    m1: np.ndarray,
    m2: np.ndarray,
    idx: np.ndarray,
    sample_size: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    use_idx = idx
    if sample_size is not None and sample_size > 0 and idx.size > sample_size:
        use_idx = rng.choice(idx, size=sample_size, replace=False)
    x = np.asarray(m1[use_idx], dtype=float)
    y = np.asarray(m2[use_idx], dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def mutual_information_hist2d(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    hist2d, _, _ = np.histogram2d(x, y, bins=bins, density=False)
    if not np.any(hist2d > 0):
        return float("nan")
    pxy = hist2d.astype(float) / np.sum(hist2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    valid = nz & (denom > 0)
    return float(np.sum(pxy[valid] * np.log(pxy[valid] / denom[valid])))


def _query_counts(tree: cKDTree, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    try:
        counts = tree.query_ball_point(points, r=radii, p=np.inf, return_length=True)
        return np.asarray(counts, dtype=int) - 1
    except TypeError:
        neighborhoods = tree.query_ball_point(points, r=radii, p=np.inf)
        return np.array([len(v) - 1 for v in neighborhoods], dtype=int)


def mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
    jitter_frac: float = 1e-10,
) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    n = x.size
    if n <= max(k + 1, 4):
        return float("nan")

    # Tiny jitter to avoid exact ties.
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return 0.0
    x = x + rng.normal(scale=jitter_frac * sx, size=n)
    y = y + rng.normal(scale=jitter_frac * sy, size=n)

    pts_x = x[:, None]
    pts_y = y[:, None]
    pts_xy = np.column_stack([x, y])

    tree_xy = cKDTree(pts_xy)
    dists, _ = tree_xy.query(pts_xy, k=k + 1, p=np.inf)
    eps = np.nextafter(dists[:, -1], 0.0)

    tree_x = cKDTree(pts_x)
    tree_y = cKDTree(pts_y)

    nx = _query_counts(tree_x, pts_x, eps)
    ny = _query_counts(tree_y, pts_y, eps)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def estimate_mi(
    x: np.ndarray,
    y: np.ndarray,
    estimator: str,
    bins: int,
    knn_k: int,
    rng: np.random.Generator,
) -> float:
    estimator = estimator.lower()
    if estimator == "hist":
        return mutual_information_hist2d(x, y, bins=bins)
    if estimator == "knn":
        return mutual_information_knn(x, y, k=knn_k, rng=rng)
    raise ValueError(f"Unknown MI estimator: {estimator}")


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def monotonicity_score(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    good = np.isfinite(v)
    v = v[good]
    if v.size < 2:
        return float("nan")
    idx = np.arange(v.size, dtype=float)
    v_mean = float(np.mean(v))
    i_mean = float(np.mean(idx))
    num = float(np.sum((idx - i_mean) * (v - v_mean)))
    den = float(np.sqrt(np.sum((idx - i_mean) ** 2) * np.sum((v - v_mean) ** 2)))
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def linear_slope(values: Sequence[float], x: Optional[np.ndarray] = None) -> float:
    y = np.asarray(values, dtype=float)
    if x is None:
        x = np.arange(y.size, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[good]
    y = y[good]
    if y.size < 2:
        return float("nan")
    A = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[1])


def empirical_p_lower(obs: float, null: np.ndarray) -> Optional[float]:
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(obs):
        return None
    return float((np.sum(arr <= obs) + 1.0) / (arr.size + 1.0))


def empirical_p_upper(obs: float, null: np.ndarray) -> Optional[float]:
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(obs):
        return None
    return float((np.sum(arr >= obs) + 1.0) / (arr.size + 1.0))


def fixed_axis_delta_mi(
    maps_lp: Sequence[np.ndarray],
    l_grid: Sequence[int],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    estimator: str,
    mi_bins: int,
    knn_k: int,
    sample_size: Optional[int],
    rng: np.random.Generator,
) -> FixedAxisResult:
    delta: List[float] = []
    mi_a_list: List[float] = []
    mi_b_list: List[float] = []
    pairs: List[List[int]] = []

    for i in range(len(l_grid) - 1):
        m1 = maps_lp[i]
        m2 = maps_lp[i + 1]
        x_a, y_a = sample_pairs(m1, m2, idx_a, sample_size, rng)
        x_b, y_b = sample_pairs(m1, m2, idx_b, sample_size, rng)

        mi_a = estimate_mi(x_a, y_a, estimator=estimator, bins=mi_bins, knn_k=knn_k, rng=rng)
        mi_b = estimate_mi(x_b, y_b, estimator=estimator, bins=mi_bins, knn_k=knn_k, rng=rng)
        dmi = abs(mi_a - mi_b) if np.isfinite(mi_a) and np.isfinite(mi_b) else float("nan")

        pairs.append([int(l_grid[i]), int(l_grid[i + 1])])
        mi_a_list.append(float(mi_a))
        mi_b_list.append(float(mi_b))
        delta.append(float(dmi))

    x_index = np.arange(len(delta), dtype=float)
    mono = monotonicity_score(delta)
    slope = linear_slope(delta, x=x_index)
    return FixedAxisResult(
        lmax_grid=[int(v) for v in l_grid],
        lmax_pairs=pairs,
        delta_mi_pairs=[float(v) for v in delta],
        mi_hemi_a_pairs=mi_a_list,
        mi_hemi_b_pairs=mi_b_list,
        monotonicity_delta_mi=float(mono),
        abs_monotonicity_delta_mi=float(abs(mono)) if np.isfinite(mono) else float("nan"),
        linear_slope_delta_mi=float(slope),
    )


def summarize_null(obs: FixedAxisResult, null_results: Sequence[FixedAxisResult], null_mode: str) -> NullSummary:
    monos = np.array([r.monotonicity_delta_mi for r in null_results], dtype=float)
    abs_monos = np.abs(monos)
    slopes = np.array([r.linear_slope_delta_mi for r in null_results], dtype=float)
    return NullSummary(
        n_valid=int(np.sum(np.isfinite(monos))),
        null_mode=null_mode,
        p_mono_decreasing=empirical_p_lower(obs.monotonicity_delta_mi, monos),
        p_abs_mono=empirical_p_upper(obs.abs_monotonicity_delta_mi, abs_monos),
        p_slope_decreasing=empirical_p_lower(obs.linear_slope_delta_mi, slopes),
        obs_mono=float(obs.monotonicity_delta_mi),
        obs_abs_mono=float(obs.abs_monotonicity_delta_mi),
        obs_slope=float(obs.linear_slope_delta_mi),
        null_mono_mean=float(np.nanmean(monos)) if np.any(np.isfinite(monos)) else None,
        null_abs_mono_mean=float(np.nanmean(abs_monos)) if np.any(np.isfinite(abs_monos)) else None,
        null_slope_mean=float(np.nanmean(slopes)) if np.any(np.isfinite(slopes)) else None,
    )


# -----------------------------------------------------------------------------
# Bayes fitting
# -----------------------------------------------------------------------------

def build_bayes_errors(
    obs_delta: np.ndarray,
    null_curves: Optional[np.ndarray],
    error_floor: float,
) -> Tuple[np.ndarray, str]:
    obs_delta = np.asarray(obs_delta, dtype=float)
    if null_curves is not None and null_curves.size > 0:
        std = np.nanstd(null_curves, axis=0, ddof=1 if null_curves.shape[0] > 1 else 0)
        std = np.where(np.isfinite(std) & (std > error_floor), std, error_floor)
        return std, "per-scale std from null curves"
    return np.full_like(obs_delta, fill_value=error_floor, dtype=float), "constant fallback error floor"


@dataclass
class BayesianModelResult:
    logZ: float
    logZerr: float
    param_means: List[float]
    param_stds: List[float]


class BoxPrior:
    def __init__(self, bounds: List[Tuple[float, float]]):
        self.bounds = bounds

    def __call__(self, uvec: np.ndarray) -> np.ndarray:
        uvec = np.asarray(uvec, dtype=float)
        out = np.empty_like(uvec)
        for i, (lo, hi) in enumerate(self.bounds):
            out[i] = lo + (hi - lo) * uvec[i]
        return out


class BayesianCurveModels:
    def __init__(self, x: np.ndarray, y: np.ndarray, yerr: np.ndarray):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.yerr = np.asarray(yerr, dtype=float)
        self.y_span = max(float(np.nanmax(y) - np.nanmin(y)), float(np.nanstd(y)), 1e-3)
        self.mu_lo = float(np.nanmin(y) - 2.0 * self.y_span)
        self.mu_hi = float(np.nanmax(y) + 2.0 * self.y_span)
        self.coeff = 4.0 * self.y_span
        sigma_lo = max(np.nanmedian(yerr) * 0.1, 1e-6)
        sigma_hi = max(10.0 * self.y_span, np.nanmedian(yerr) * 10.0, 1e-4)
        self.logsig_bounds = (math.log10(sigma_lo), math.log10(sigma_hi))

    def loglike_const(self, params: np.ndarray) -> float:
        mu_c, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma = np.sqrt(self.yerr ** 2 + sigma_int ** 2)
        return float(np.sum(norm.logpdf(self.y, loc=mu_c, scale=sigma)))

    def loglike_linear(self, params: np.ndarray) -> float:
        mu_l, slope, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma = np.sqrt(self.yerr ** 2 + sigma_int ** 2)
        model = mu_l + slope * self.x
        return float(np.sum(norm.logpdf(self.y, loc=model, scale=sigma)))

    def loglike_quadratic(self, params: np.ndarray) -> float:
        mu_q, a, b, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma = np.sqrt(self.yerr ** 2 + sigma_int ** 2)
        model = mu_q + a * self.x + b * self.x ** 2
        return float(np.sum(norm.logpdf(self.y, loc=model, scale=sigma)))

    def prior_const(self) -> BoxPrior:
        return BoxPrior([
            (self.mu_lo, self.mu_hi),
            self.logsig_bounds,
        ])

    def prior_linear_symmetric(self) -> BoxPrior:
        return BoxPrior([
            (self.mu_lo, self.mu_hi),
            (-self.coeff, self.coeff),
            self.logsig_bounds,
        ])

    def prior_linear_negative(self) -> BoxPrior:
        return BoxPrior([
            (self.mu_lo, self.mu_hi),
            (-self.coeff, 0.0),
            self.logsig_bounds,
        ])

    def prior_quadratic(self) -> BoxPrior:
        return BoxPrior([
            (self.mu_lo, self.mu_hi),
            (-self.coeff, self.coeff),
            (-self.coeff, self.coeff),
            self.logsig_bounds,
        ])


def run_nested(loglike, prior, ndim: int, nlive: int, dlogz: float, label: str) -> BayesianModelResult:
    if dynesty is None:
        raise RuntimeError("dynesty is not installed; Bayesian fitting is unavailable.")
    log(f"Nested sampling: {label} (ndim={ndim}, nlive={nlive}, dlogz={dlogz})")
    sampler = dynesty.NestedSampler(loglike, prior, ndim, nlive=nlive, bound="multi", sample="rwalk")
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    logz = float(res.logz[-1])
    logzerr = float(res.logzerr[-1])
    weights = np.exp(res.logwt - res.logz[-1])
    weights = weights / np.sum(weights)
    means = np.sum(res.samples * weights[:, None], axis=0)
    vars_ = np.sum((res.samples - means[None, :]) ** 2 * weights[:, None], axis=0)
    return BayesianModelResult(
        logZ=logz,
        logZerr=logzerr,
        param_means=[float(v) for v in means],
        param_stds=[float(v) for v in np.sqrt(np.maximum(vars_, 0.0))],
    )


def bayes_compare_curve(
    obs: FixedAxisResult,
    null_curves: Optional[np.ndarray],
    nlive: int,
    dlogz: float,
    error_floor: float,
) -> Dict[str, Any]:
    y = np.asarray(obs.delta_mi_pairs, dtype=float)
    if y.size < 3 or not np.all(np.isfinite(y)):
        raise RuntimeError("Not enough finite ΔMI points for Bayes fitting.")

    x_raw = np.asarray([pair[1] for pair in obs.lmax_pairs], dtype=float)
    x = (x_raw - np.mean(x_raw)) / max(np.std(x_raw), 1.0)
    yerr, err_note = build_bayes_errors(y, null_curves, error_floor=error_floor)
    models = BayesianCurveModels(x=x, y=y, yerr=yerr)

    const_res = run_nested(models.loglike_const, models.prior_const(), 2, nlive, dlogz, "constant")
    lin_sym_res = run_nested(models.loglike_linear, models.prior_linear_symmetric(), 3, nlive, dlogz, "linear_symmetric")
    lin_neg_res = run_nested(models.loglike_linear, models.prior_linear_negative(), 3, nlive, dlogz, "linear_negative")
    quad_res = run_nested(models.loglike_quadratic, models.prior_quadratic(), 4, nlive, dlogz, "quadratic")

    out = {
        "x_upper_lmax": [int(v) for v in x_raw],
        "x_standardized": [float(v) for v in x],
        "y_delta_mi": [float(v) for v in y],
        "y_error": [float(v) for v in yerr],
        "error_model": err_note,
        "models": {
            "constant": asdict(const_res),
            "linear_symmetric": asdict(lin_sym_res),
            "linear_negative": asdict(lin_neg_res),
            "quadratic": asdict(quad_res),
        },
        "bayes_factors": {
            "delta_logZ_linear_symmetric_minus_constant": float(lin_sym_res.logZ - const_res.logZ),
            "delta_logZ_linear_negative_minus_constant": float(lin_neg_res.logZ - const_res.logZ),
            "delta_logZ_quadratic_minus_constant": float(quad_res.logZ - const_res.logZ),
        },
    }
    return out


# -----------------------------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------------------------

def maybe_plot(
    outpath: str,
    obs: FixedAxisResult,
    null_curves: Optional[np.ndarray],
) -> None:
    if plt is None:
        warn("matplotlib is not available; skipping plot generation.")
        return
    x = np.array([pair[1] for pair in obs.lmax_pairs], dtype=float)
    y = np.asarray(obs.delta_mi_pairs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="fixed-axis ΔMI")
    if null_curves is not None and null_curves.size > 0:
        mu = np.nanmean(null_curves, axis=0)
        sd = np.nanstd(null_curves, axis=0, ddof=1 if null_curves.shape[0] > 1 else 0)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.25, label="phase-null ±1σ")
        ax.plot(x, mu, linestyle="--", label="phase-null mean")
    ax.set_xlabel(r"upper $\ell_{\max}$ of pair")
    ax.set_ylabel(r"$\Delta MI$")
    ax.set_title("Fixed-axis CMB cross-scale MI asymmetry")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    log(f"Saved plot: {outpath}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pantheon-driven fixed-axis CMB MI analysis with optional kNN MI, phase nulls, and Bayes fitting."
    )
    axis_group = ap.add_mutually_exclusive_group(required=True)
    axis_group.add_argument("--pantheon-json", type=str, default=None, help="Pantheon scan JSON from cos_pantheon_axis_scan.py")
    axis_group.add_argument("--axis-lon", type=float, default=None, help="Manual axis longitude in degrees")

    ap.add_argument("--axis-lat", type=float, default=None, help="Manual axis latitude in degrees (required with --axis-lon)")
    ap.add_argument("--axis-coords", choices=["gal", "icrs"], default="gal", help="Coordinate system for manual axis input")
    ap.add_argument("--pantheon-stat", choices=["abs_delta_q0", "abs_delta_mean_residual"], default="abs_delta_q0", help="Statistic used to choose the best Pantheon axis")

    ap.add_argument("--map", required=True, help="CMB map FITS")
    ap.add_argument("--map-field", type=int, default=0, help="Map FITS field index")
    ap.add_argument("--mask", default=None, help="Optional mask FITS")
    ap.add_argument("--mask-field", type=int, default=0, help="Mask FITS field index")
    ap.add_argument("--work-nside", type=int, default=256, help="Working NSIDE")
    ap.add_argument("--lmax-grid", type=str, default="8,16,24,32,48,64,96,128,192,256", help="Comma-separated ℓ_max grid")

    ap.add_argument("--mi-estimator", choices=["hist", "knn"], default="knn", help="MI estimator")
    ap.add_argument("--mi-bins", type=int, default=32, help="2D histogram bin count for hist MI")
    ap.add_argument("--knn-k", type=int, default=5, help="k for kNN MI")
    ap.add_argument("--mi-sample-size", type=int, default=20000, help="Per-hemisphere paired-pixel subsample size for MI evaluation; 0 means use all")

    ap.add_argument("--n-phase-null", type=int, default=0, help="Number of phase-randomized null maps")
    ap.add_argument("--n-random-axes", type=int, default=0, help="Optional number of random axes on the real map for a local rank diagnostic")
    ap.add_argument("--seed", type=int, default=12345, help="Master RNG seed")

    ap.add_argument("--run-bayes", action="store_true", help="Run Bayes model comparison on the fixed-axis ΔMI curve")
    ap.add_argument("--bayes-nlive", type=int, default=500, help="dynesty nlive")
    ap.add_argument("--bayes-dlogz", type=float, default=0.5, help="dynesty dlogz stopping threshold")
    ap.add_argument("--bayes-error-floor", type=float, default=1e-4, help="Minimum y-error used in Bayes fitting")

    ap.add_argument("--plot", default=None, help="Optional plot output path (PNG/PDF)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    return ap.parse_args()


def parse_l_grid(text: str) -> List[int]:
    try:
        vals = [int(x) for x in text.replace(";", ",").split(",") if x.strip()]
    except Exception as exc:
        raise SystemExit(f"Invalid --lmax-grid: {text!r} ({exc})")
    if len(vals) < 3:
        raise SystemExit("At least three ℓ_max values are required.")
    if sorted(vals) != vals:
        raise SystemExit("--lmax-grid must be monotonically increasing.")
    return vals


def main() -> None:
    args = parse_args()
    require_healpy()
    rng = np.random.default_rng(args.seed)

    if args.axis_lon is not None and args.axis_lat is None:
        raise SystemExit("--axis-lat is required when --axis-lon is used.")

    l_grid = parse_l_grid(args.lmax_grid)
    log(f"ℓ-grid: {l_grid}")
    log(f"MI estimator: {args.mi_estimator}")
    if args.mi_estimator == "knn":
        log(f"kNN settings: k={args.knn_k}, sample_size={args.mi_sample_size}")
    else:
        log(f"Histogram MI settings: bins={args.mi_bins}, sample_size={args.mi_sample_size}")

    if args.pantheon_json is not None:
        axis = load_axis_from_pantheon_json(args.pantheon_json, statistic=args.pantheon_stat)
        log(
            f"Selected Pantheon axis by {args.pantheon_stat}: "
            f"ICRS(RA,Dec)=({axis.icrs_ra_deg:.3f},{axis.icrs_dec_deg:.3f}), "
            f"Gal(l,b)=({axis.gal_lon_deg:.3f},{axis.gal_lat_deg:.3f})"
        )
    else:
        axis = load_manual_axis(args.axis_lon, args.axis_lat, coords=args.axis_coords)
        log(
            f"Using manual axis: ICRS(RA,Dec)=({axis.icrs_ra_deg:.3f},{axis.icrs_dec_deg:.3f}), "
            f"Gal(l,b)=({axis.gal_lon_deg:.3f},{axis.gal_lat_deg:.3f})"
        )

    m_raw = robust_read_map(args.map, field=args.map_field, label="map")
    mask_raw = robust_read_map(args.mask, field=args.mask_field, label="mask") if args.mask else None
    m_work, mask_work = degrade_to_work_nside(m_raw, mask_raw, work_nside=args.work_nside)
    nside = hp.get_nside(m_work)
    log(f"Working NSIDE={nside}, npix={hp.nside2npix(nside)}")

    m_std = standardize_map(m_work, mask_work)
    lmax_full = max(l_grid)
    log(f"Computing alm up to lmax={lmax_full}")
    alm_full = hp.map2alm(m_std, lmax=lmax_full)
    maps_lp = build_lowpass_maps_from_alm(alm_full, l_grid=l_grid, nside=nside)

    pix_vecs = precompute_pix_vectors(nside)
    idx_a, idx_b = hemisphere_pixel_indices(
        nside=nside,
        lon_deg=axis.gal_lon_deg,
        lat_deg=axis.gal_lat_deg,
        pix_vecs=pix_vecs,
        base_mask=mask_work,
    )
    log(f"Fixed-axis hemisphere sizes after mask: A={idx_a.size}, B={idx_b.size}")
    sample_size = None if args.mi_sample_size <= 0 else int(args.mi_sample_size)

    obs = fixed_axis_delta_mi(
        maps_lp=maps_lp,
        l_grid=l_grid,
        idx_a=idx_a,
        idx_b=idx_b,
        estimator=args.mi_estimator,
        mi_bins=args.mi_bins,
        knn_k=args.knn_k,
        sample_size=sample_size,
        rng=rng,
    )
    log(
        f"Observed fixed-axis statistics: mono={obs.monotonicity_delta_mi:.4f}, "
        f"|mono|={obs.abs_monotonicity_delta_mi:.4f}, slope={obs.linear_slope_delta_mi:.4e}"
    )

    phase_null_results: List[FixedAxisResult] = []
    phase_null_curves: Optional[np.ndarray] = None
    null_summary: Optional[NullSummary] = None
    if args.n_phase_null > 0:
        log(f"Running {args.n_phase_null} phase-randomized null realizations")
        for i in range(args.n_phase_null):
            alm_rand = phase_randomize_alm(alm_full, rng=rng)
            maps_rand = build_lowpass_maps_from_alm(alm_rand, l_grid=l_grid, nside=nside)
            res = fixed_axis_delta_mi(
                maps_lp=maps_rand,
                l_grid=l_grid,
                idx_a=idx_a,
                idx_b=idx_b,
                estimator=args.mi_estimator,
                mi_bins=args.mi_bins,
                knn_k=args.knn_k,
                sample_size=sample_size,
                rng=rng,
            )
            phase_null_results.append(res)
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == args.n_phase_null:
                log(f"Phase null {i + 1}/{args.n_phase_null}")
        phase_null_curves = np.array([r.delta_mi_pairs for r in phase_null_results], dtype=float)
        null_summary = summarize_null(obs, phase_null_results, null_mode="phase_randomized")
        log(
            f"Phase-null p-values: p(mono_decreasing)={null_summary.p_mono_decreasing}, "
            f"p(|mono|)={null_summary.p_abs_mono}, p(slope_decreasing)={null_summary.p_slope_decreasing}"
        )

    random_axis_rank: Optional[Dict[str, Any]] = None
    if args.n_random_axes > 0:
        log(f"Running local rank diagnostic with {args.n_random_axes} random axes on the real map")
        lon = rng.uniform(0.0, 360.0, size=args.n_random_axes)
        uvals = rng.uniform(-1.0, 1.0, size=args.n_random_axes)
        lat = np.degrees(np.arcsin(uvals))
        rnd_abs = []
        for i in range(args.n_random_axes):
            ridx_a, ridx_b = hemisphere_pixel_indices(
                nside=nside,
                lon_deg=float(lon[i]),
                lat_deg=float(lat[i]),
                pix_vecs=pix_vecs,
                base_mask=mask_work,
            )
            rres = fixed_axis_delta_mi(
                maps_lp=maps_lp,
                l_grid=l_grid,
                idx_a=ridx_a,
                idx_b=ridx_b,
                estimator=args.mi_estimator,
                mi_bins=args.mi_bins,
                knn_k=args.knn_k,
                sample_size=sample_size,
                rng=rng,
            )
            rnd_abs.append(rres.abs_monotonicity_delta_mi)
        rnd_abs = np.array(rnd_abs, dtype=float)
        rnd_abs = rnd_abs[np.isfinite(rnd_abs)]
        percentile = None
        if rnd_abs.size > 0 and np.isfinite(obs.abs_monotonicity_delta_mi):
            percentile = float((np.sum(rnd_abs <= obs.abs_monotonicity_delta_mi) + 0.5) / (rnd_abs.size + 1.0))
        random_axis_rank = {
            "n_random_axes": int(rnd_abs.size),
            "percentile_abs_vs_random_axes": percentile,
            "random_abs_mono_mean": float(np.mean(rnd_abs)) if rnd_abs.size > 0 else None,
            "random_abs_mono_max": float(np.max(rnd_abs)) if rnd_abs.size > 0 else None,
        }
        log(f"Local rank percentile of fixed axis vs random axes: {percentile}")

    bayes = None
    if args.run_bayes:
        if dynesty is None:
            warn("dynesty is not installed; skipping Bayesian model comparison. Install with: pip install dynesty")
        else:
            bayes = bayes_compare_curve(
                obs=obs,
                null_curves=phase_null_curves,
                nlive=args.bayes_nlive,
                dlogz=args.bayes_dlogz,
                error_floor=args.bayes_error_floor,
            )
            log(
                "Bayes comparison complete: "
                f"ΔlogZ(linear_negative-constant)={bayes['bayes_factors']['delta_logZ_linear_negative_minus_constant']:.3f}"
            )

    if args.plot:
        maybe_plot(args.plot, obs=obs, null_curves=phase_null_curves)

    out = {
        "metadata": {
            "map_path": args.map,
            "mask_path": args.mask,
            "map_field": int(args.map_field),
            "mask_field": int(args.mask_field),
            "work_nside": int(nside),
            "lmax_grid": [int(v) for v in l_grid],
            "mi_estimator": args.mi_estimator,
            "mi_bins": int(args.mi_bins),
            "knn_k": int(args.knn_k),
            "mi_sample_size": int(args.mi_sample_size),
            "n_phase_null": int(args.n_phase_null),
            "n_random_axes": int(args.n_random_axes),
            "seed": int(args.seed),
            "method_note": (
                "Pantheon best-axis -> fixed-axis CMB follow-up. "
                "Cross-probe conditioning reduces CMB look-elsewhere relative to a pure axis scan, "
                "but does not erase Pantheon-side axis-selection uncertainty or trials."
            ),
        },
        "chosen_axis": asdict(axis),
        "fixed_axis_result": asdict(obs),
        "phase_randomized_null": asdict(null_summary) if null_summary is not None else None,
        "random_axis_rank": random_axis_rank,
        "bayes": bayes,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"Saved output: {args.out}")


if __name__ == "__main__":
    main()
