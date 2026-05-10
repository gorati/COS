#!/usr/bin/env python3
"""
COS Pantheon+ axis scan (optimized v2)

What changed vs. the first version
----------------------------------
- The baseline cosmographic fit is computed once and reused.
- For the primary statistic ``abs_delta_mean_residual`` the hemisphere statistic is
  represented as a linear form in the residual vector. This makes the main axis
  scan and the residual-shuffle null test much faster.
- A slower sky-scramble null mode is still available for methodological checks,
  but on a desktop machine it can take many hours with 1000 axes x 300 nulls.
- Optional secondary ``delta_q0`` calculations can be disabled (default) because
  they are not needed when the primary statistic is residual-based.

The fast path preserves the same *data statistic* for ``abs_delta_mean_residual``
while changing only how it is computed numerically. The residual-shuffle null is a
valid null model, but it is not identical to the slower sky-scramble null.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required. Install it with: pip install pandas") from exc


C_KM_S = 299792.458
DEFAULT_H0 = 70.0


def log(msg: str) -> None:
    print(f"[info] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}")


@dataclass
class FitResult:
    q0: float
    M: float
    chi2: float
    ndof: int


@dataclass
class AxisResult:
    axis_id: int
    kind: str
    lon_deg: float
    lat_deg: float
    n_pos: int
    n_neg: int
    delta_mean_residual: float
    abs_delta_mean_residual: float
    delta_q0: float
    abs_delta_q0: float


@dataclass
class NullSummary:
    n_valid: int
    statistic_name: str
    null_mode: str
    cos_value: float
    data_scan_max_value: Optional[float]
    global_max_p_value: Optional[float]
    fixed_cos_p_value: Optional[float]


class AnalysisError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Coordinate helpers
# -----------------------------------------------------------------------------

def unit_vector_from_lonlat_deg(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    clat = np.cos(lat)
    return np.array([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)], dtype=float)


def vectors_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cdec = np.cos(dec)
    return np.column_stack((cdec * np.cos(ra), cdec * np.sin(ra), np.sin(dec)))


def galactic_to_icrs_lonlat(l_deg: float, b_deg: float) -> Tuple[float, float]:
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError as exc:
        raise AnalysisError(
            "Galactic axis input requested, but astropy is not installed. "
            "Install astropy or provide axis coordinates in ICRS."
        ) from exc

    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    icrs = c.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_pantheon_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data table not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    if df.empty:
        raise AnalysisError(f"Loaded table is empty: {path}")
    required = ["zHD", "m_b_corr", "RA", "DEC"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise AnalysisError(
            f"Required columns missing from {path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def load_covariance_matrix(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Covariance file not found: {path}")
    raw = np.loadtxt(path)
    raw = np.asarray(raw, dtype=float).ravel()
    if raw.size < 2:
        raise AnalysisError(f"Covariance file is too short: {path}")
    n = int(round(raw[0]))
    remainder = raw[1:]
    if remainder.size == n * n:
        return remainder.reshape((n, n))
    raw2d = np.loadtxt(path, ndmin=2)
    if raw2d.ndim == 2 and raw2d.shape[0] == raw2d.shape[1]:
        return np.asarray(raw2d, dtype=float)
    raise AnalysisError(
        f"Could not parse covariance matrix from {path}. "
        f"Expected either [n, flattened matrix] or a square text matrix."
    )


# -----------------------------------------------------------------------------
# Selection and model
# -----------------------------------------------------------------------------

def make_selection_mask(
    df: pd.DataFrame,
    zmin: float,
    zmax: float,
    include_calibrators: bool,
    used_in_sh0es_only: bool,
) -> np.ndarray:
    mask = np.isfinite(df["zHD"].to_numpy(dtype=float))
    mask &= np.isfinite(df["m_b_corr"].to_numpy(dtype=float))
    mask &= np.isfinite(df["RA"].to_numpy(dtype=float))
    mask &= np.isfinite(df["DEC"].to_numpy(dtype=float))
    mask &= (df["zHD"].to_numpy(dtype=float) > zmin)
    mask &= (df["zHD"].to_numpy(dtype=float) < zmax)

    if not include_calibrators and "IS_CALIBRATOR" in df.columns:
        is_cal = df["IS_CALIBRATOR"].to_numpy()
        mask &= (is_cal == 0) | (is_cal == False)  # noqa: E712

    if used_in_sh0es_only and "USED_IN_SH0ES_HF" in df.columns:
        used = df["USED_IN_SH0ES_HF"].to_numpy()
        mask &= (used == 1) | (used == True)  # noqa: E712

    return mask.astype(bool)


def subselect_covariance(cov: np.ndarray, keep: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(keep)
    if cov.shape[0] != cov.shape[1]:
        raise AnalysisError("Covariance matrix is not square.")
    if idx.size == 0:
        raise AnalysisError("Selection mask is empty.")
    if cov.shape[0] < idx.max() + 1:
        raise AnalysisError("Covariance matrix is smaller than the selected data vector length.")
    return cov[np.ix_(idx, idx)]


def mu_cosmography_lowz(z: np.ndarray, q0: float, H0: float = DEFAULT_H0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    dl_mpc = (C_KM_S / H0) * (z + 0.5 * (1.0 - q0) * z * z)
    if np.any(dl_mpc <= 0):
        raise AnalysisError("Non-positive luminosity distance encountered.")
    return 5.0 * np.log10(dl_mpc) + 25.0


def best_M_given_mu(mu_model: np.ndarray, m_obs: np.ndarray, inv_cov: np.ndarray) -> float:
    ones = np.ones_like(mu_model)
    rhs = ones @ inv_cov @ (m_obs - mu_model)
    denom = ones @ inv_cov @ ones
    return float(rhs / denom)


def chi2_for_q0(q0: float, z: np.ndarray, m_obs: np.ndarray, inv_cov: np.ndarray) -> Tuple[float, float]:
    mu = mu_cosmography_lowz(z, q0=q0)
    M = best_M_given_mu(mu, m_obs, inv_cov)
    resid = m_obs - (mu + M)
    chi2 = float(resid @ inv_cov @ resid)
    return chi2, M


def fit_cosmography_grid(
    z: np.ndarray,
    m_obs: np.ndarray,
    cov: np.ndarray,
    q0_min: float = -2.0,
    q0_max: float = 1.0,
    n_grid: int = 1201,
) -> FitResult:
    if z.size < 8:
        raise AnalysisError("Too few data points for a stable fit.")
    inv_cov = np.linalg.inv(cov)
    qgrid = np.linspace(q0_min, q0_max, n_grid)
    chi2_vals = np.empty_like(qgrid)
    M_vals = np.empty_like(qgrid)
    for i, q0 in enumerate(qgrid):
        chi2_vals[i], M_vals[i] = chi2_for_q0(q0, z, m_obs, inv_cov)
    i_best = int(np.argmin(chi2_vals))
    ndof = int(z.size - 2)
    return FitResult(
        q0=float(qgrid[i_best]),
        M=float(M_vals[i_best]),
        chi2=float(chi2_vals[i_best]),
        ndof=ndof,
    )


def weighted_mean_from_cov(values: np.ndarray, cov: np.ndarray) -> float:
    ones = np.ones(len(values), dtype=float)
    sol = np.linalg.solve(cov, ones)
    den = float(ones @ sol)
    if den == 0.0 or not np.isfinite(den):
        raise AnalysisError("Degenerate weighted-mean denominator encountered.")
    return float((sol @ values) / den)


def weighted_mean_linear_weights(cov: np.ndarray) -> np.ndarray:
    ones = np.ones(cov.shape[0], dtype=float)
    sol = np.linalg.solve(cov, ones)
    den = float(ones @ sol)
    if den == 0.0 or not np.isfinite(den):
        raise AnalysisError("Degenerate weighted-mean denominator encountered.")
    return sol / den


# -----------------------------------------------------------------------------
# Axis helpers
# -----------------------------------------------------------------------------

def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    r = np.sqrt(1.0 - u * u)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), u))


def lonlat_from_unit_vector(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = map(float, v)
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


def build_axis_set(cos_axis_vec: np.ndarray, n_random_axes: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    random_axes = random_unit_vectors(n_random_axes, rng)
    axes = np.vstack([cos_axis_vec.reshape(1, 3), random_axes])
    kinds = ["cos"] + ["random"] * n_random_axes
    return axes, kinds


def membership_matrix(obj_vecs: np.ndarray, axes: np.ndarray) -> np.ndarray:
    dots = obj_vecs @ axes.T
    return dots >= 0.0


def compute_residuals(z: np.ndarray, m_obs: np.ndarray, fit: FitResult) -> np.ndarray:
    return m_obs - (mu_cosmography_lowz(z, q0=fit.q0) + fit.M)


def prepare_residual_mean_operator(
    cov: np.ndarray,
    pos_matrix: np.ndarray,
    min_hemi_size: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build linear operators c_i such that delta_i = c_i @ residuals."""
    n_obj, n_axes = pos_matrix.shape
    coeff = np.full((n_axes, n_obj), np.nan, dtype=float)
    n_pos_arr = np.zeros(n_axes, dtype=int)
    n_neg_arr = np.zeros(n_axes, dtype=int)
    valid = np.zeros(n_axes, dtype=bool)

    log(f"Precomputing hemisphere weights for {n_axes} axes")
    for i in range(n_axes):
        pos = pos_matrix[:, i]
        neg = ~pos
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        n_pos_arr[i] = n_pos
        n_neg_arr[i] = n_neg
        if n_pos < min_hemi_size or n_neg < min_hemi_size:
            continue
        cov_pos = cov[np.ix_(pos, pos)]
        cov_neg = cov[np.ix_(neg, neg)]
        w_pos = weighted_mean_linear_weights(cov_pos)
        w_neg = weighted_mean_linear_weights(cov_neg)
        row = np.zeros(n_obj, dtype=float)
        row[pos] = w_pos
        row[neg] = -w_neg
        coeff[i, :] = row
        valid[i] = True
        if (i + 1) % progress_every == 0 or i == 0 or (i + 1) == n_axes:
            log(f"Prepared axis weights {i + 1}/{n_axes}")
    return coeff, valid, n_pos_arr, n_neg_arr


def residual_stat_from_coeff(coeff: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    return coeff @ residuals


def compute_delta_q0_for_axes(
    z: np.ndarray,
    m_obs: np.ndarray,
    cov: np.ndarray,
    pos_matrix: np.ndarray,
    valid_axes: np.ndarray,
    progress_every: int,
) -> np.ndarray:
    n_axes = pos_matrix.shape[1]
    out = np.full(n_axes, np.nan, dtype=float)
    log("Computing hemisphere delta_q0 values (slow diagnostic)")
    for i in range(n_axes):
        if not valid_axes[i]:
            continue
        pos = pos_matrix[:, i]
        neg = ~pos
        cov_pos = cov[np.ix_(pos, pos)]
        cov_neg = cov[np.ix_(neg, neg)]
        fit_pos = fit_cosmography_grid(z[pos], m_obs[pos], cov_pos)
        fit_neg = fit_cosmography_grid(z[neg], m_obs[neg], cov_neg)
        out[i] = float(fit_pos.q0 - fit_neg.q0)
        if (i + 1) % progress_every == 0 or i == 0 or (i + 1) == n_axes:
            log(f"Computed delta_q0 for axis {i + 1}/{n_axes}")
    return out


def build_axis_results(
    axes: np.ndarray,
    kinds: Sequence[str],
    n_pos_arr: np.ndarray,
    n_neg_arr: np.ndarray,
    delta_mean: np.ndarray,
    delta_q0: Optional[np.ndarray],
) -> List[AxisResult]:
    results: List[AxisResult] = []
    for i, (axis_vec, kind) in enumerate(zip(axes, kinds)):
        lon_deg, lat_deg = lonlat_from_unit_vector(axis_vec)
        dmean = float(delta_mean[i]) if np.isfinite(delta_mean[i]) else np.nan
        dq0 = np.nan
        if delta_q0 is not None and np.isfinite(delta_q0[i]):
            dq0 = float(delta_q0[i])
        results.append(
            AxisResult(
                axis_id=i,
                kind=kind,
                lon_deg=lon_deg,
                lat_deg=lat_deg,
                n_pos=int(n_pos_arr[i]),
                n_neg=int(n_neg_arr[i]),
                delta_mean_residual=dmean,
                abs_delta_mean_residual=abs(dmean) if np.isfinite(dmean) else np.nan,
                delta_q0=dq0,
                abs_delta_q0=abs(dq0) if np.isfinite(dq0) else np.nan,
            )
        )
    return results


# -----------------------------------------------------------------------------
# Summaries and null tests
# -----------------------------------------------------------------------------

def empirical_p_geq(null_values: np.ndarray, value: float) -> Optional[float]:
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0 or not np.isfinite(value):
        return None
    return float((np.sum(valid >= value) + 1.0) / (valid.size + 1.0))


def summarize_cos_vs_random(results: Sequence[AxisResult], stat_name: str) -> Dict[str, float]:
    cos_res = next(r for r in results if r.kind == "cos")
    rnd = np.array([getattr(r, stat_name) for r in results if r.kind == "random"], dtype=float)
    rnd = rnd[np.isfinite(rnd)]
    cos_val = float(getattr(cos_res, stat_name))
    percentile = float((np.sum(rnd <= cos_val) / rnd.size) * 100.0) if rnd.size > 0 else np.nan
    return {
        "cos_value": cos_val,
        "random_count": int(rnd.size),
        "percentile_vs_random": percentile,
        "random_max": float(np.max(rnd)) if rnd.size > 0 else np.nan,
        "random_mean": float(np.mean(rnd)) if rnd.size > 0 else np.nan,
    }


def residual_shuffle_null(
    residuals: np.ndarray,
    coeff: np.ndarray,
    valid_axes: np.ndarray,
    stat_name: str,
    n_null: int,
    null_seed: int,
    progress_every: int,
) -> NullSummary:
    rng = np.random.default_rng(null_seed)
    data_vals = residual_stat_from_coeff(coeff, residuals)
    data_abs = np.abs(data_vals)
    cos_value = float(data_abs[0])

    fixed_cos_null = np.empty(n_null, dtype=float)
    global_max_null = np.empty(n_null, dtype=float)
    idx = np.arange(residuals.size)
    valid_rows = valid_axes.copy()
    data_scan_max_value = float(np.nanmax(data_abs[valid_rows]))

    for i in range(n_null):
        perm = rng.permutation(idx)
        vals = residual_stat_from_coeff(coeff, residuals[perm])
        vals_abs = np.abs(vals)
        fixed_cos_null[i] = vals_abs[0]
        global_max_null[i] = float(np.nanmax(vals_abs[valid_rows]))
        if (i + 1) % progress_every == 0 or i == 0 or (i + 1) == n_null:
            log(f"Residual-shuffle null {i + 1}/{n_null}")

    return NullSummary(
        n_valid=int(n_null),
        statistic_name=stat_name,
        null_mode="residual-shuffle",
        cos_value=cos_value,
        data_scan_max_value=data_scan_max_value,
        global_max_p_value=empirical_p_geq(global_max_null, data_scan_max_value),
        fixed_cos_p_value=empirical_p_geq(fixed_cos_null, cos_value),
    )


def sky_scramble_null(
    residuals: np.ndarray,
    axes: np.ndarray,
    obj_vecs: np.ndarray,
    cov: np.ndarray,
    stat_name: str,
    n_null: int,
    null_seed: int,
    min_hemi_size: int,
    progress_every: int,
) -> NullSummary:
    rng = np.random.default_rng(null_seed)
    base_pos = membership_matrix(obj_vecs, axes)
    coeff0, valid0, *_ = prepare_residual_mean_operator(cov, base_pos, min_hemi_size, progress_every=max(progress_every * 10, 1000))
    data_vals = residual_stat_from_coeff(coeff0, residuals)
    data_abs = np.abs(data_vals)
    cos_value = float(data_abs[0])
    data_scan_max_value = float(np.nanmax(data_abs[valid0]))

    fixed_cos_null: List[float] = []
    global_max_null: List[float] = []
    idx = np.arange(residuals.size)

    warn("Sky-scramble null is exact but slow with the full covariance matrix.")
    for i in range(n_null):
        perm = rng.permutation(idx)
        pos_perm = base_pos[perm, :]
        coeff_i, valid_i, *_ = prepare_residual_mean_operator(
            cov, pos_perm, min_hemi_size, progress_every=max(progress_every * 10, 1000)
        )
        vals = residual_stat_from_coeff(coeff_i, residuals)
        vals_abs = np.abs(vals)
        fixed_cos_null.append(float(vals_abs[0]))
        global_max_null.append(float(np.nanmax(vals_abs[valid_i])))
        if (i + 1) % progress_every == 0 or i == 0 or (i + 1) == n_null:
            log(f"Sky-scramble null {i + 1}/{n_null}")

    fixed_cos_null_arr = np.array(fixed_cos_null, dtype=float)
    global_max_null_arr = np.array(global_max_null, dtype=float)
    return NullSummary(
        n_valid=int(global_max_null_arr.size),
        statistic_name=stat_name,
        null_mode="sky-scramble",
        cos_value=cos_value,
        data_scan_max_value=data_scan_max_value,
        global_max_p_value=empirical_p_geq(global_max_null_arr, data_scan_max_value),
        fixed_cos_p_value=empirical_p_geq(fixed_cos_null_arr, cos_value),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COS axis scan on Pantheon+ supernova data (optimized v2)")
    p.add_argument("--data", required=True, help="Path to Pantheon+ .dat table")
    p.add_argument("--cov", required=True, help="Path to Pantheon+ covariance file")
    p.add_argument("--out", required=True, help="Output JSON path")

    p.add_argument("--zmin", type=float, default=0.01, help="Minimum zHD for the baseline sample")
    p.add_argument("--zmax", type=float, default=0.10, help="Maximum zHD for the baseline sample")
    p.add_argument("--include-calibrators", action="store_true", help="Include SH0ES calibrators in the selected sample")
    p.add_argument("--used-in-sh0es-only", action="store_true", help="Restrict to rows flagged as USED_IN_SH0ES_HF when that column exists")

    p.add_argument("--cos-lon", type=float, default=0.0, help="COS axis longitude / RA in degrees")
    p.add_argument("--cos-lat", type=float, default=90.0, help="COS axis latitude / Dec in degrees")
    p.add_argument("--cos-coords", choices=["icrs", "gal"], default="gal", help="Coordinate system of the COS axis input")

    p.add_argument("--n-random-axes", type=int, default=1000, help="Number of random axes in the scan")
    p.add_argument("--axis-seed", type=int, default=12345, help="Random seed for the axis set")
    p.add_argument("--min-hemi-size", type=int, default=20, help="Minimum number of SNe per hemisphere")

    p.add_argument(
        "--statistic",
        choices=["abs_delta_mean_residual", "abs_delta_q0"],
        default="abs_delta_mean_residual",
        help="Primary axis statistic for ranking and null calibration",
    )
    p.add_argument(
        "--compute-secondary-delta-q0",
        action="store_true",
        help="Also compute per-axis delta_q0 diagnostics (slow).",
    )

    p.add_argument("--run-null", action="store_true", help="Run null realizations")
    p.add_argument(
        "--null-mode",
        choices=["residual-shuffle", "sky-scramble"],
        default="residual-shuffle",
        help="Null mode. residual-shuffle is much faster; sky-scramble is slower but more literal.",
    )
    p.add_argument("--run-sky-scramble-null", action="store_true", help="Backward-compatible alias for --run-null --null-mode sky-scramble")
    p.add_argument("--n-null", type=int, default=300, help="Number of null realizations")
    p.add_argument("--null-seed", type=int, default=24680, help="Random seed for the null realizations")
    p.add_argument("--progress-every", type=int, default=10, help="Progress print cadence for null realizations")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_sky_scramble_null:
        args.run_null = True
        args.null_mode = "sky-scramble"

    log(f"Loading table: {args.data}")
    df = load_pantheon_table(args.data)

    log(f"Loading covariance: {args.cov}")
    cov_full = load_covariance_matrix(args.cov)
    if cov_full.shape[0] != len(df):
        raise AnalysisError(f"Covariance size {cov_full.shape} does not match table length {len(df)}.")

    sel = make_selection_mask(
        df=df,
        zmin=args.zmin,
        zmax=args.zmax,
        include_calibrators=args.include_calibrators,
        used_in_sh0es_only=args.used_in_sh0es_only,
    )
    n_sel = int(np.sum(sel))
    if n_sel < 50:
        raise AnalysisError(f"Selected sample is too small: {n_sel}")
    log(f"Selected sample size: {n_sel}")

    df_sel = df.loc[sel].reset_index(drop=True)
    cov_sel = subselect_covariance(cov_full, sel)
    z = df_sel["zHD"].to_numpy(dtype=float)
    m_obs = df_sel["m_b_corr"].to_numpy(dtype=float)
    ra = df_sel["RA"].to_numpy(dtype=float)
    dec = df_sel["DEC"].to_numpy(dtype=float)
    obj_vecs = vectors_from_radec_deg(ra, dec)

    if args.cos_coords == "gal":
        cos_ra, cos_dec = galactic_to_icrs_lonlat(args.cos_lon, args.cos_lat)
        log(f"COS axis converted from Galactic to ICRS: RA={cos_ra:.3f}, Dec={cos_dec:.3f}")
    else:
        cos_ra, cos_dec = args.cos_lon, args.cos_lat
        log(f"COS axis in ICRS: RA={cos_ra:.3f}, Dec={cos_dec:.3f}")

    cos_axis_vec = unit_vector_from_lonlat_deg(cos_ra, cos_dec)
    baseline_fit = fit_cosmography_grid(z, m_obs, cov_sel)
    log(
        "Baseline fit: "
        f"q0={baseline_fit.q0:.4f}, M={baseline_fit.M:.4f}, "
        f"chi2/ndof={baseline_fit.chi2:.2f}/{baseline_fit.ndof}"
    )
    residuals = compute_residuals(z, m_obs, baseline_fit)

    axes, kinds = build_axis_set(cos_axis_vec, args.n_random_axes, args.axis_seed)
    pos_matrix = membership_matrix(obj_vecs, axes)

    coeff, valid_axes, n_pos_arr, n_neg_arr = prepare_residual_mean_operator(
        cov_sel, pos_matrix, args.min_hemi_size, progress_every=max(args.progress_every * 5, 25)
    )
    delta_mean = residual_stat_from_coeff(coeff, residuals)

    delta_q0 = None
    need_delta_q0 = args.compute_secondary_delta_q0 or args.statistic == "abs_delta_q0"
    if need_delta_q0:
        delta_q0 = compute_delta_q0_for_axes(
            z, m_obs, cov_sel, pos_matrix, valid_axes, progress_every=max(args.progress_every * 5, 25)
        )
    else:
        log("Skipping per-axis delta_q0 diagnostics for speed")

    axis_results = build_axis_results(axes, kinds, n_pos_arr, n_neg_arr, delta_mean, delta_q0)
    ranking = summarize_cos_vs_random(axis_results, args.statistic)
    log(
        f"COS {args.statistic}={ranking['cos_value']:.6g}, "
        f"percentile vs random={ranking['percentile_vs_random']:.2f}%"
    )

    null_summary = None
    if args.run_null:
        if args.null_mode == "residual-shuffle" and args.statistic != "abs_delta_mean_residual":
            raise AnalysisError(
                "Fast residual-shuffle null is currently implemented only for the residual-based primary statistic. "
                "Use --statistic abs_delta_mean_residual or switch to --null-mode sky-scramble."
            )
        log(f"Running {args.null_mode} null test")
        if args.null_mode == "residual-shuffle":
            null_summary = residual_shuffle_null(
                residuals=residuals,
                coeff=coeff,
                valid_axes=valid_axes,
                stat_name=args.statistic,
                n_null=args.n_null,
                null_seed=args.null_seed,
                progress_every=args.progress_every,
            )
        else:
            null_summary = sky_scramble_null(
                residuals=residuals,
                axes=axes,
                obj_vecs=obj_vecs,
                cov=cov_sel,
                stat_name=args.statistic,
                n_null=args.n_null,
                null_seed=args.null_seed,
                min_hemi_size=args.min_hemi_size,
                progress_every=args.progress_every,
            )
        log(
            "Null summary: "
            f"mode={null_summary.null_mode}, "
            f"global_max_p={null_summary.global_max_p_value}, "
            f"fixed_cos_p={null_summary.fixed_cos_p_value}, "
            f"valid={null_summary.n_valid}"
        )

    output = {
        "metadata": {
            "data_path": args.data,
            "cov_path": args.cov,
            "zmin": args.zmin,
            "zmax": args.zmax,
            "include_calibrators": bool(args.include_calibrators),
            "used_in_sh0es_only": bool(args.used_in_sh0es_only),
            "sample_size": n_sel,
            "n_random_axes": args.n_random_axes,
            "axis_seed": args.axis_seed,
            "min_hemi_size": args.min_hemi_size,
            "primary_statistic": args.statistic,
            "compute_secondary_delta_q0": bool(args.compute_secondary_delta_q0),
            "cos_input_coords": args.cos_coords,
            "cos_input_lon_deg": args.cos_lon,
            "cos_input_lat_deg": args.cos_lat,
            "cos_axis_icrs_ra_deg": cos_ra,
            "cos_axis_icrs_dec_deg": cos_dec,
            "null_ran": bool(args.run_null),
            "null_mode": args.null_mode if args.run_null else None,
            "n_null": args.n_null if args.run_null else 0,
            "progress_every": args.progress_every,
        },
        "baseline_fit": asdict(baseline_fit),
        "cos_vs_random_summary": ranking,
        "axis_results": [asdict(r) for r in axis_results],
        "null_summary": asdict(null_summary) if null_summary is not None else None,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log(f"Saved output: {args.out}")


if __name__ == "__main__":
    main()
