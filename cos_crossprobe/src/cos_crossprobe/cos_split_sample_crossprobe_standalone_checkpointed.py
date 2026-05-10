#!/usr/bin/env python3
"""
Split-sample Pantheon axis validation + optional fixed-axis CMB follow-up.

Why this script exists
----------------------
The earlier Pantheon->CMB pipeline selected the Pantheon axis on the full SN sample
and then tested that same axis on CMB. That reduces the *CMB-side* look-elsewhere
penalty, but it still leaves Pantheon-side axis-selection uncertainty.

This script tightens the design:

1. Split the Pantheon sample repeatedly into train/validation subsets.
2. On each train split, scan the axis set and select the best axis.
3. On the corresponding validation split, test that *fixed* train-selected axis:
   - out-of-sample percentile versus the validation random-axis distribution,
   - optional fixed-axis sky-scramble null p-value.
4. Aggregate the split results and choose a validated axis.
5. Optionally carry that validated axis into the fixed-axis CMB MI pipeline.

The goal is not to manufacture a detection. The goal is to turn the Pantheon axis
choice into an out-of-sample, auditable selection step before the CMB follow-up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[info] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)


class AnalysisError(RuntimeError):
    pass


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"Could not load module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class SplitValidationResult:
    split_id: int
    n_train: int
    n_validation: int
    train_best_axis_id: int
    train_best_axis_kind: str
    train_best_lon_deg: float
    train_best_lat_deg: float
    train_best_stat_value: float
    train_best_signed_value: float
    validation_same_axis_stat_value: float
    validation_same_axis_signed_value: float
    validation_percentile_vs_random: Optional[float]
    validation_fixed_axis_null_p: Optional[float]
    validation_random_mean: Optional[float]
    validation_random_max: Optional[float]
    train_cos_stat_value: Optional[float]
    validation_cos_stat_value: Optional[float]
    sign_consistent_train_validation: Optional[bool]


@dataclass
class AggregatedAxisChoice:
    source: str
    axis_id: Optional[int]
    kind: Optional[str]
    lon_deg: float
    lat_deg: float
    score: float
    n_supporting_splits: int


C_KM_S = 299792.458
DEFAULT_H0 = 70.0


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


def unit_vector_from_lonlat_deg(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    clat = np.cos(lat)
    return np.array([
        clat * np.cos(lon),
        clat * np.sin(lon),
        np.sin(lat),
    ], dtype=float)


def lonlat_from_unit_vector(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = map(float, v)
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


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
        mask &= (is_cal == 0) | (is_cal == False)

    if used_in_sh0es_only and "USED_IN_SH0ES_HF" in df.columns:
        used = df["USED_IN_SH0ES_HF"].to_numpy()
        mask &= (used == 1) | (used == True)

    return mask.astype(bool)


def subselect_covariance(cov: np.ndarray, keep: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(keep)
    if cov.shape[0] != cov.shape[1]:
        raise AnalysisError("Covariance matrix is not square.")
    if idx.size == 0:
        raise AnalysisError("Empty covariance sub-selection.")
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


def weighted_mean(values: np.ndarray, cov: np.ndarray) -> float:
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(values), dtype=float)
    num = ones @ inv_cov @ values
    den = ones @ inv_cov @ ones
    return float(num / den)


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    r = np.sqrt(1.0 - u * u)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), u))


def split_by_axis(obj_vecs: np.ndarray, axis_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dots = obj_vecs @ axis_vec
    pos = dots >= 0.0
    neg = ~pos
    return pos, neg


def compute_residuals(z: np.ndarray, m_obs: np.ndarray, fit: FitResult) -> np.ndarray:
    return m_obs - (mu_cosmography_lowz(z, q0=fit.q0) + fit.M)


def axis_statistic_for_sample(
    z: np.ndarray,
    m_obs: np.ndarray,
    cov: np.ndarray,
    obj_vecs: np.ndarray,
    axis_vec: np.ndarray,
    axis_id: int,
    kind: str,
    min_hemi_size: int,
) -> AxisResult:
    pos, neg = split_by_axis(obj_vecs, axis_vec)
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    lon_deg, lat_deg = lonlat_from_unit_vector(axis_vec)

    if n_pos < min_hemi_size or n_neg < min_hemi_size:
        return AxisResult(
            axis_id=axis_id,
            kind=kind,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            n_pos=n_pos,
            n_neg=n_neg,
            delta_mean_residual=np.nan,
            abs_delta_mean_residual=np.nan,
            delta_q0=np.nan,
            abs_delta_q0=np.nan,
        )

    global_fit = fit_cosmography_grid(z, m_obs, cov)
    resid = compute_residuals(z, m_obs, global_fit)

    cov_pos = cov[np.ix_(pos, pos)]
    cov_neg = cov[np.ix_(neg, neg)]
    mean_pos = weighted_mean(resid[pos], cov_pos)
    mean_neg = weighted_mean(resid[neg], cov_neg)
    delta_mean = float(mean_pos - mean_neg)

    fit_pos = fit_cosmography_grid(z[pos], m_obs[pos], cov_pos)
    fit_neg = fit_cosmography_grid(z[neg], m_obs[neg], cov_neg)
    delta_q0 = float(fit_pos.q0 - fit_neg.q0)

    return AxisResult(
        axis_id=axis_id,
        kind=kind,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        n_pos=n_pos,
        n_neg=n_neg,
        delta_mean_residual=delta_mean,
        abs_delta_mean_residual=abs(delta_mean),
        delta_q0=delta_q0,
        abs_delta_q0=abs(delta_q0),
    )


def run_axis_scan(
    z: np.ndarray,
    m_obs: np.ndarray,
    cov: np.ndarray,
    obj_vecs: np.ndarray,
    cos_axis_vec: np.ndarray,
    n_random_axes: int,
    seed: int,
    min_hemi_size: int,
) -> List[AxisResult]:
    rng = np.random.default_rng(seed)
    random_axes = random_unit_vectors(n_random_axes, rng)
    axes = [np.asarray(cos_axis_vec, dtype=float)] + [np.asarray(random_axes[i], dtype=float) for i in range(n_random_axes)]
    kinds = ["cos"] + ["random"] * n_random_axes
    results = []
    for axis_id, (avec, kind) in enumerate(zip(axes, kinds)):
        results.append(
            axis_statistic_for_sample(
                z=z,
                m_obs=m_obs,
                cov=cov,
                obj_vecs=obj_vecs,
                axis_vec=np.asarray(avec, dtype=float),
                axis_id=axis_id,
                kind=kind,
                min_hemi_size=min_hemi_size,
            )
        )
    return results


def empirical_p_geq(null_values: np.ndarray, value: float) -> Optional[float]:
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0 or not np.isfinite(value):
        return None
    return float((np.sum(valid >= value) + 1.0) / (valid.size + 1.0))


class _PantheonAPI:
    pass


def make_internal_pantheon_api():
    p = _PantheonAPI()
    p.load_pantheon_table = load_pantheon_table
    p.load_covariance_matrix = load_covariance_matrix
    p.make_selection_mask = make_selection_mask
    p.subselect_covariance = subselect_covariance
    p.vectors_from_radec_deg = vectors_from_radec_deg
    p.galactic_to_icrs_lonlat = galactic_to_icrs_lonlat
    p.unit_vector_from_lonlat_deg = unit_vector_from_lonlat_deg
    p.lonlat_from_unit_vector = lonlat_from_unit_vector
    p.fit_cosmography_grid = fit_cosmography_grid
    p.axis_statistic_for_sample = axis_statistic_for_sample
    p.run_axis_scan = run_axis_scan
    p.empirical_p_geq = empirical_p_geq
    p.random_unit_vectors = random_unit_vectors
    return p


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------


def parse_l_grid(text: str) -> List[int]:
    vals = [int(x) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(vals) < 3:
        raise AnalysisError("At least three lmax values are required.")
    if sorted(vals) != vals:
        raise AnalysisError("lmax grid must be monotonically increasing.")
    return vals


def subset_cov(cov: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return cov[np.ix_(idx, idx)]


def make_quantile_bins(z: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.zeros(len(z), dtype=int)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(z, qs)
    # Avoid empty-width bins from tied quantiles.
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(len(z), dtype=int)
    bins = np.digitize(z, edges[1:-1], right=False)
    return bins.astype(int)


def random_train_validation_split(
    n: int,
    validation_fraction: float,
    rng: np.random.Generator,
    strata: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.05 < validation_fraction < 0.95):
        raise AnalysisError("validation_fraction must lie between 0.05 and 0.95.")
    all_idx = np.arange(n, dtype=int)

    if strata is None:
        perm = rng.permutation(all_idx)
        n_val = max(1, int(round(validation_fraction * n)))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return train_idx, val_idx

    strata = np.asarray(strata, dtype=int)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for s in np.unique(strata):
        idx = all_idx[strata == s]
        if idx.size == 0:
            continue
        perm = rng.permutation(idx)
        n_val_s = int(round(validation_fraction * idx.size))
        n_val_s = min(max(n_val_s, 1), max(idx.size - 1, 1)) if idx.size >= 2 else idx.size
        val_parts.append(np.sort(perm[:n_val_s]))
        train_parts.append(np.sort(perm[n_val_s:]))

    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    if train_idx.size == 0 or val_idx.size == 0:
        raise AnalysisError("Split failed: empty train or validation set.")
    return train_idx, val_idx


# -----------------------------------------------------------------------------
# Pantheon split-sample logic
# -----------------------------------------------------------------------------


def get_axis_values(results: Sequence[Any], stat_name: str) -> np.ndarray:
    return np.array([float(getattr(r, stat_name)) for r in results], dtype=float)


def get_signed_stat_name(abs_name: str) -> str:
    if abs_name == "abs_delta_q0":
        return "delta_q0"
    if abs_name == "abs_delta_mean_residual":
        return "delta_mean_residual"
    raise AnalysisError(f"Unsupported statistic: {abs_name}")


def fixed_axis_sky_scramble_pvalue(
    pscan,
    z: np.ndarray,
    m_obs: np.ndarray,
    cov: np.ndarray,
    obj_vecs: np.ndarray,
    axis_vec: np.ndarray,
    axis_kind: str,
    axis_id: int,
    min_hemi_size: int,
    stat_name: str,
    n_null: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], np.ndarray]:
    obs = pscan.axis_statistic_for_sample(
        z=z,
        m_obs=m_obs,
        cov=cov,
        obj_vecs=obj_vecs,
        axis_vec=axis_vec,
        axis_id=axis_id,
        kind=axis_kind,
        min_hemi_size=min_hemi_size,
    )
    obs_val = float(getattr(obs, stat_name))
    if not np.isfinite(obs_val):
        return None, np.array([], dtype=float)

    idx = np.arange(len(z), dtype=int)
    null_vals: List[float] = []
    for i in range(n_null):
        perm = rng.permutation(idx)
        res = pscan.axis_statistic_for_sample(
            z=z,
            m_obs=m_obs,
            cov=cov,
            obj_vecs=obj_vecs[perm],
            axis_vec=axis_vec,
            axis_id=axis_id,
            kind=axis_kind,
            min_hemi_size=min_hemi_size,
        )
        val = float(getattr(res, stat_name))
        if np.isfinite(val):
            null_vals.append(val)
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == n_null:
            log(f"Validation fixed-axis sky-scramble null {i + 1}/{n_null}")

    arr = np.array(null_vals, dtype=float)
    pval = pscan.empirical_p_geq(arr, obs_val)
    return pval, arr


def weighted_percentile_score(percentile: Optional[float], pval: Optional[float], stat: float) -> float:
    pct = 0.0 if percentile is None or not np.isfinite(percentile) else float(percentile)
    if pval is None or not np.isfinite(pval) or pval <= 0.0:
        p_term = 0.0
    else:
        p_term = max(0.0, -math.log10(pval))
    stat_term = 0.0 if not np.isfinite(stat) else float(stat)
    return pct + p_term + 0.05 * stat_term


def angular_distance_mod_sign_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    x = float(np.clip(abs(np.dot(v1, v2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(x)))


def aggregate_validated_axes(
    split_results: Sequence[SplitValidationResult],
    pscan,
) -> Tuple[Optional[AggregatedAxisChoice], Optional[AggregatedAxisChoice], List[Dict[str, Any]]]:
    if not split_results:
        return None, None, []

    leaderboard: List[Dict[str, Any]] = []
    best_split: Optional[SplitValidationResult] = None
    best_score = -np.inf

    for s in split_results:
        score = weighted_percentile_score(
            s.validation_percentile_vs_random,
            s.validation_fixed_axis_null_p,
            s.validation_same_axis_stat_value,
        )
        leaderboard.append(
            {
                "split_id": s.split_id,
                "axis_id": s.train_best_axis_id,
                "kind": s.train_best_axis_kind,
                "lon_deg": s.train_best_lon_deg,
                "lat_deg": s.train_best_lat_deg,
                "validation_percentile_vs_random": s.validation_percentile_vs_random,
                "validation_fixed_axis_null_p": s.validation_fixed_axis_null_p,
                "validation_same_axis_stat_value": s.validation_same_axis_stat_value,
                "aggregate_score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_split = s

    if best_split is None:
        return None, None, leaderboard

    best_validated = AggregatedAxisChoice(
        source="best_validated_split",
        axis_id=int(best_split.train_best_axis_id),
        kind=str(best_split.train_best_axis_kind),
        lon_deg=float(best_split.train_best_lon_deg),
        lat_deg=float(best_split.train_best_lat_deg),
        score=float(best_score),
        n_supporting_splits=1,
    )

    ref_vec = pscan.unit_vector_from_lonlat_deg(best_split.train_best_lon_deg, best_split.train_best_lat_deg)
    vec_sum = np.zeros(3, dtype=float)
    weight_sum = 0.0
    n_support = 0
    for s in split_results:
        vec = pscan.unit_vector_from_lonlat_deg(s.train_best_lon_deg, s.train_best_lat_deg)
        if np.dot(vec, ref_vec) < 0.0:
            vec = -vec
        weight = weighted_percentile_score(
            s.validation_percentile_vs_random,
            s.validation_fixed_axis_null_p,
            s.validation_same_axis_stat_value,
        )
        if weight <= 0.0:
            continue
        vec_sum += weight * vec
        weight_sum += weight
        n_support += 1

    consensus = None
    if weight_sum > 0.0 and np.linalg.norm(vec_sum) > 0.0:
        v = vec_sum / np.linalg.norm(vec_sum)
        lon_deg, lat_deg = pscan.lonlat_from_unit_vector(v)
        consensus = AggregatedAxisChoice(
            source="weighted_consensus_over_splits",
            axis_id=None,
            kind="consensus",
            lon_deg=float(lon_deg),
            lat_deg=float(lat_deg),
            score=float(weight_sum / max(n_support, 1)),
            n_supporting_splits=int(n_support),
        )

    leaderboard.sort(key=lambda d: d["aggregate_score"], reverse=True)
    return best_validated, consensus, leaderboard


def make_checkpoint_path(args: argparse.Namespace) -> str:
    if getattr(args, "checkpoint_out", None):
        return str(args.checkpoint_out)
    return str(args.out) + ".checkpoint.json"


def write_pantheon_checkpoint(
    args: argparse.Namespace,
    baseline_fit: Any,
    split_results: Sequence[SplitValidationResult],
    axis_support_rows: Sequence[Dict[str, Any]],
    cos_ra: float,
    cos_dec: float,
    n_sel: int,
) -> None:
    val_p = np.array(
        [s.validation_fixed_axis_null_p for s in split_results if s.validation_fixed_axis_null_p is not None and np.isfinite(s.validation_fixed_axis_null_p)],
        dtype=float,
    )
    val_pct = np.array(
        [s.validation_percentile_vs_random for s in split_results if s.validation_percentile_vs_random is not None and np.isfinite(s.validation_percentile_vs_random)],
        dtype=float,
    )
    payload = {
        "checkpoint": True,
        "complete": False,
        "pantheon_split_validation": {
            "metadata": {
                "data_path": args.data,
                "cov_path": args.cov,
                "zmin": float(args.zmin),
                "zmax": float(args.zmax),
                "include_calibrators": bool(args.include_calibrators),
                "used_in_sh0es_only": bool(args.used_in_sh0es_only),
                "sample_size": int(n_sel),
                "n_splits_requested": int(args.n_splits),
                "n_splits_completed": int(len(split_results)),
                "validation_fraction": float(args.validation_fraction),
                "split_seed": int(args.split_seed),
                "stratify_z": bool(args.stratify_z),
                "z_strat_bins": int(args.z_strat_bins),
                "n_random_axes": int(args.n_random_axes),
                "axis_seed": int(args.axis_seed),
                "min_hemi_size": int(args.min_hemi_size),
                "primary_statistic": args.statistic,
                "n_validation_null": int(args.n_validation_null),
                "validation_null_seed": int(args.validation_null_seed),
                "cos_input_coords": args.cos_coords,
                "cos_input_lon_deg": float(args.cos_lon),
                "cos_input_lat_deg": float(args.cos_lat),
                "cos_axis_icrs_ra_deg": float(cos_ra),
                "cos_axis_icrs_dec_deg": float(cos_dec),
            },
            "baseline_fit_full_selected_sample": asdict(baseline_fit),
            "split_results": [asdict(s) for s in split_results],
            "axis_support_rows": list(axis_support_rows),
            "aggregate_summary_partial": {
                "n_valid_splits": int(len(split_results)),
                "median_validation_percentile_vs_random": float(np.median(val_pct)) if val_pct.size > 0 else None,
                "mean_validation_percentile_vs_random": float(np.mean(val_pct)) if val_pct.size > 0 else None,
                "median_validation_fixed_axis_null_p": float(np.median(val_p)) if val_p.size > 0 else None,
                "mean_validation_fixed_axis_null_p": float(np.mean(val_p)) if val_p.size > 0 else None,
            },
        },
        "cmb_followup": None,
    }

    cp_path = make_checkpoint_path(args)
    cp_file = Path(cp_path)
    cp_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp_file.with_suffix(cp_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(cp_file)




def run_split_sample_pantheon(args: argparse.Namespace, pscan) -> Dict[str, Any]:
    log(f"Loading Pantheon table: {args.data}")
    df = pscan.load_pantheon_table(args.data)

    log(f"Loading covariance: {args.cov}")
    cov_full = pscan.load_covariance_matrix(args.cov)
    if cov_full.shape[0] != len(df):
        raise AnalysisError(
            f"Covariance size {cov_full.shape} does not match table length {len(df)}."
        )

    sel = pscan.make_selection_mask(
        df=df,
        zmin=args.zmin,
        zmax=args.zmax,
        include_calibrators=args.include_calibrators,
        used_in_sh0es_only=args.used_in_sh0es_only,
    )
    n_sel = int(np.sum(sel))
    if n_sel < 100:
        raise AnalysisError(f"Selected sample is too small for split-sample validation: {n_sel}")

    df_sel = df.loc[sel].reset_index(drop=True)
    cov_sel = pscan.subselect_covariance(cov_full, sel)
    z = df_sel["zHD"].to_numpy(dtype=float)
    m_obs = df_sel["m_b_corr"].to_numpy(dtype=float)
    ra = df_sel["RA"].to_numpy(dtype=float)
    dec = df_sel["DEC"].to_numpy(dtype=float)
    obj_vecs = pscan.vectors_from_radec_deg(ra, dec)

    if args.cos_coords == "gal":
        cos_ra, cos_dec = pscan.galactic_to_icrs_lonlat(args.cos_lon, args.cos_lat)
        log(f"COS axis converted from Galactic to ICRS: RA={cos_ra:.3f}, Dec={cos_dec:.3f}")
    else:
        cos_ra, cos_dec = args.cos_lon, args.cos_lat
        log(f"COS axis in ICRS: RA={cos_ra:.3f}, Dec={cos_dec:.3f}")
    cos_axis_vec = pscan.unit_vector_from_lonlat_deg(cos_ra, cos_dec)

    baseline_fit = pscan.fit_cosmography_grid(z, m_obs, cov_sel)
    log(
        "Pantheon baseline fit: "
        f"q0={baseline_fit.q0:.4f}, M={baseline_fit.M:.4f}, chi2/ndof={baseline_fit.chi2:.2f}/{baseline_fit.ndof}"
    )

    split_rng = np.random.default_rng(args.split_seed)
    null_base_seed = int(args.validation_null_seed)
    signed_name = get_signed_stat_name(args.statistic)

    strata = None
    if args.stratify_z:
        strata = make_quantile_bins(z, args.z_strat_bins)
        log(f"Using z-stratified splits with {len(np.unique(strata))} occupied bins")

    split_results: List[SplitValidationResult] = []
    axis_support_rows: List[Dict[str, Any]] = []

    for split_id in range(args.n_splits):
        train_idx, val_idx = random_train_validation_split(
            n=n_sel,
            validation_fraction=args.validation_fraction,
            rng=split_rng,
            strata=strata,
        )

        z_tr = z[train_idx]
        m_tr = m_obs[train_idx]
        vec_tr = obj_vecs[train_idx]
        cov_tr = subset_cov(cov_sel, train_idx)

        z_va = z[val_idx]
        m_va = m_obs[val_idx]
        vec_va = obj_vecs[val_idx]
        cov_va = subset_cov(cov_sel, val_idx)

        train_scan = pscan.run_axis_scan(
            z=z_tr,
            m_obs=m_tr,
            cov=cov_tr,
            obj_vecs=vec_tr,
            cos_axis_vec=cos_axis_vec,
            n_random_axes=args.n_random_axes,
            seed=args.axis_seed,
            min_hemi_size=args.min_hemi_size,
        )
        val_scan = pscan.run_axis_scan(
            z=z_va,
            m_obs=m_va,
            cov=cov_va,
            obj_vecs=vec_va,
            cos_axis_vec=cos_axis_vec,
            n_random_axes=args.n_random_axes,
            seed=args.axis_seed,
            min_hemi_size=args.min_hemi_size,
        )

        train_vals = get_axis_values(train_scan, args.statistic)
        if not np.any(np.isfinite(train_vals)):
            warn(f"Split {split_id}: no finite train-axis statistics; skipping")
            continue

        best_idx = int(np.nanargmax(train_vals))
        best_train = train_scan[best_idx]
        best_val = val_scan[best_idx]
        val_rnd = np.array(
            [float(getattr(r, args.statistic)) for r in val_scan if r.kind == "random"],
            dtype=float,
        )
        val_rnd = val_rnd[np.isfinite(val_rnd)]
        val_pct = None
        if val_rnd.size > 0 and np.isfinite(getattr(best_val, args.statistic)):
            val_pct = float((np.sum(val_rnd <= getattr(best_val, args.statistic)) + 0.5) / (val_rnd.size + 1.0))

        axis_vec = pscan.unit_vector_from_lonlat_deg(best_train.lon_deg, best_train.lat_deg)
        null_p = None
        if args.n_validation_null > 0:
            null_rng = np.random.default_rng(null_base_seed + 100003 * split_id)
            null_p, null_arr = fixed_axis_sky_scramble_pvalue(
                pscan=pscan,
                z=z_va,
                m_obs=m_va,
                cov=cov_va,
                obj_vecs=vec_va,
                axis_vec=axis_vec,
                axis_kind=best_train.kind,
                axis_id=best_train.axis_id,
                min_hemi_size=args.min_hemi_size,
                stat_name=args.statistic,
                n_null=args.n_validation_null,
                rng=null_rng,
            )
        else:
            null_arr = np.array([], dtype=float)

        train_cos = train_scan[0] if train_scan else None
        val_cos = val_scan[0] if val_scan else None
        train_signed = float(getattr(best_train, signed_name)) if np.isfinite(getattr(best_train, signed_name)) else float("nan")
        val_signed = float(getattr(best_val, signed_name)) if np.isfinite(getattr(best_val, signed_name)) else float("nan")
        sign_consistent = None
        if np.isfinite(train_signed) and np.isfinite(val_signed):
            sign_consistent = bool(np.sign(train_signed) == np.sign(val_signed))

        split_res = SplitValidationResult(
            split_id=int(split_id),
            n_train=int(train_idx.size),
            n_validation=int(val_idx.size),
            train_best_axis_id=int(best_train.axis_id),
            train_best_axis_kind=str(best_train.kind),
            train_best_lon_deg=float(best_train.lon_deg),
            train_best_lat_deg=float(best_train.lat_deg),
            train_best_stat_value=float(getattr(best_train, args.statistic)),
            train_best_signed_value=train_signed,
            validation_same_axis_stat_value=float(getattr(best_val, args.statistic)),
            validation_same_axis_signed_value=val_signed,
            validation_percentile_vs_random=val_pct,
            validation_fixed_axis_null_p=null_p,
            validation_random_mean=float(np.mean(val_rnd)) if val_rnd.size > 0 else None,
            validation_random_max=float(np.max(val_rnd)) if val_rnd.size > 0 else None,
            train_cos_stat_value=float(getattr(train_cos, args.statistic)) if train_cos is not None else None,
            validation_cos_stat_value=float(getattr(val_cos, args.statistic)) if val_cos is not None else None,
            sign_consistent_train_validation=sign_consistent,
        )
        split_results.append(split_res)

        axis_support_rows.append(
            {
                "split_id": int(split_id),
                "axis_id": int(best_train.axis_id),
                "kind": str(best_train.kind),
                "lon_deg": float(best_train.lon_deg),
                "lat_deg": float(best_train.lat_deg),
                "train_stat": float(getattr(best_train, args.statistic)),
                "validation_stat": float(getattr(best_val, args.statistic)),
                "validation_percentile_vs_random": val_pct,
                "validation_fixed_axis_null_p": null_p,
                "validation_null_mean": float(np.mean(null_arr)) if null_arr.size > 0 else None,
                "validation_null_max": float(np.max(null_arr)) if null_arr.size > 0 else None,
                "sign_consistent_train_validation": sign_consistent,
            }
        )

        log(
            f"Split {split_id + 1}/{args.n_splits}: train axis id={best_train.axis_id} ({best_train.kind}), "
            f"train {args.statistic}={getattr(best_train, args.statistic):.5g}, "
            f"validation {args.statistic}={getattr(best_val, args.statistic):.5g}, "
            f"validation percentile={val_pct}, validation fixed-axis p={null_p}"
        )

        write_pantheon_checkpoint(
            args=args,
            baseline_fit=baseline_fit,
            split_results=split_results,
            axis_support_rows=axis_support_rows,
            cos_ra=cos_ra,
            cos_dec=cos_dec,
            n_sel=n_sel,
        )

    if not split_results:
        raise AnalysisError("All splits failed; no valid split-sample Pantheon results were produced.")

    best_validated, consensus, leaderboard = aggregate_validated_axes(split_results, pscan)
    if best_validated is None:
        raise AnalysisError("Could not identify a validated Pantheon axis from the split results.")

    val_p = np.array(
        [s.validation_fixed_axis_null_p for s in split_results if s.validation_fixed_axis_null_p is not None and np.isfinite(s.validation_fixed_axis_null_p)],
        dtype=float,
    )
    val_pct = np.array(
        [s.validation_percentile_vs_random for s in split_results if s.validation_percentile_vs_random is not None and np.isfinite(s.validation_percentile_vs_random)],
        dtype=float,
    )

    out = {
        "metadata": {
            "data_path": args.data,
            "cov_path": args.cov,
            "zmin": float(args.zmin),
            "zmax": float(args.zmax),
            "include_calibrators": bool(args.include_calibrators),
            "used_in_sh0es_only": bool(args.used_in_sh0es_only),
            "sample_size": int(n_sel),
            "n_splits": int(args.n_splits),
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
            "stratify_z": bool(args.stratify_z),
            "z_strat_bins": int(args.z_strat_bins),
            "n_random_axes": int(args.n_random_axes),
            "axis_seed": int(args.axis_seed),
            "min_hemi_size": int(args.min_hemi_size),
            "primary_statistic": args.statistic,
            "n_validation_null": int(args.n_validation_null),
            "validation_null_seed": int(args.validation_null_seed),
            "cos_input_coords": args.cos_coords,
            "cos_input_lon_deg": float(args.cos_lon),
            "cos_input_lat_deg": float(args.cos_lat),
            "cos_axis_icrs_ra_deg": float(cos_ra),
            "cos_axis_icrs_dec_deg": float(cos_dec),
            "method_note": (
                "Train/validation split-sample Pantheon axis selection. "
                "The selected axis is chosen on the train subset, then tested as a fixed axis on the validation subset. "
                "This reduces Pantheon-side overfitting relative to a full-sample axis scan, but does not by itself prove a cosmological arrow of time."
            ),
        },
        "baseline_fit_full_selected_sample": asdict(baseline_fit),
        "split_results": [asdict(s) for s in split_results],
        "axis_support_rows": axis_support_rows,
        "leaderboard": leaderboard,
        "aggregate_summary": {
            "n_valid_splits": int(len(split_results)),
            "median_validation_percentile_vs_random": float(np.median(val_pct)) if val_pct.size > 0 else None,
            "mean_validation_percentile_vs_random": float(np.mean(val_pct)) if val_pct.size > 0 else None,
            "median_validation_fixed_axis_null_p": float(np.median(val_p)) if val_p.size > 0 else None,
            "mean_validation_fixed_axis_null_p": float(np.mean(val_p)) if val_p.size > 0 else None,
        },
        "best_validated_axis": asdict(best_validated),
        "consensus_axis": asdict(consensus) if consensus is not None else None,
    }
    return out


# -----------------------------------------------------------------------------
# Optional CMB follow-up
# -----------------------------------------------------------------------------


def run_cmb_followup(
    args: argparse.Namespace,
    cprobe,
    chosen_axis: AggregatedAxisChoice,
) -> Dict[str, Any]:
    if getattr(cprobe, "hp", None) is None:
        raise AnalysisError("healpy is required for the CMB follow-up.")

    rng = np.random.default_rng(args.seed)
    l_grid = parse_l_grid(args.lmax_grid)

    axis = cprobe.load_manual_axis(
        lon_deg=chosen_axis.lon_deg,
        lat_deg=chosen_axis.lat_deg,
        coords="gal",
    )
    log(
        f"CMB follow-up axis from {chosen_axis.source}: "
        f"Gal(l,b)=({axis.gal_lon_deg:.3f},{axis.gal_lat_deg:.3f}), "
        f"ICRS(RA,Dec)=({axis.icrs_ra_deg:.3f},{axis.icrs_dec_deg:.3f})"
    )

    m_raw = cprobe.robust_read_map(args.map, field=args.map_field, label="map")
    mask_raw = cprobe.robust_read_map(args.mask, field=args.mask_field, label="mask") if args.mask else None
    m_work, mask_work = cprobe.degrade_to_work_nside(m_raw, mask_raw, work_nside=args.work_nside)
    nside = cprobe.hp.get_nside(m_work)
    m_std = cprobe.standardize_map(m_work, mask_work)
    lmax_full = max(l_grid)
    log(f"Computing CMB alm up to lmax={lmax_full}")
    alm_full = cprobe.hp.map2alm(m_std, lmax=lmax_full)
    maps_lp = cprobe.build_lowpass_maps_from_alm(alm_full, l_grid=l_grid, nside=nside)

    pix_vecs = cprobe.precompute_pix_vectors(nside)
    idx_a, idx_b = cprobe.hemisphere_pixel_indices(
        nside=nside,
        lon_deg=axis.gal_lon_deg,
        lat_deg=axis.gal_lat_deg,
        pix_vecs=pix_vecs,
        base_mask=mask_work,
    )
    sample_size = None if args.mi_sample_size <= 0 else int(args.mi_sample_size)
    obs = cprobe.fixed_axis_delta_mi(
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

    phase_null_results: List[Any] = []
    phase_null_curves: Optional[np.ndarray] = None
    null_summary = None
    if args.n_phase_null > 0:
        log(f"Running {args.n_phase_null} phase-randomized CMB null realizations")
        for i in range(args.n_phase_null):
            alm_rand = cprobe.phase_randomize_alm(alm_full, rng=rng)
            maps_rand = cprobe.build_lowpass_maps_from_alm(alm_rand, l_grid=l_grid, nside=nside)
            res = cprobe.fixed_axis_delta_mi(
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
                log(f"CMB phase null {i + 1}/{args.n_phase_null}")
        phase_null_curves = np.array([r.delta_mi_pairs for r in phase_null_results], dtype=float)
        null_summary = cprobe.summarize_null(obs, phase_null_results, null_mode="phase_randomized")

    bayes = None
    if args.run_bayes:
        if getattr(cprobe, "dynesty", None) is None:
            warn("dynesty is not installed; skipping Bayes fitting in CMB follow-up.")
        else:
            bayes = cprobe.bayes_compare_curve(
                obs=obs,
                null_curves=phase_null_curves,
                nlive=args.bayes_nlive,
                dlogz=args.bayes_dlogz,
                error_floor=args.bayes_error_floor,
            )

    if args.plot:
        cprobe.maybe_plot(args.plot, obs=obs, null_curves=phase_null_curves)

    return {
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
            "seed": int(args.seed),
        },
        "chosen_axis": asdict(axis),
        "fixed_axis_result": asdict(obs),
        "phase_randomized_null": asdict(null_summary) if null_summary is not None else None,
        "bayes": bayes,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Split-sample Pantheon axis validation with optional fixed-axis CMB follow-up"
    )

    # Module paths
    ap.add_argument("--pantheon-script", default="cos_pantheon_axis_scan.py", help="Ignored in the standalone build; kept only for CLI compatibility")
    ap.add_argument("--crossprobe-script", default="cos_crossprobe_fixed_axis.py", help="Path to cos_crossprobe_fixed_axis.py")

    # Pantheon inputs
    ap.add_argument("--data", required=True, help="Pantheon+ .dat table")
    ap.add_argument("--cov", required=True, help="Pantheon+ covariance file")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--checkpoint-out", default=None, help="Optional checkpoint JSON path written after each completed split")

    # Pantheon sample selection
    ap.add_argument("--zmin", type=float, default=0.01)
    ap.add_argument("--zmax", type=float, default=0.10)
    ap.add_argument("--include-calibrators", action="store_true")
    ap.add_argument("--used-in-sh0es-only", action="store_true")

    # COS / axis scan settings
    ap.add_argument("--cos-lon", type=float, default=0.0, help="COS axis longitude / RA in degrees")
    ap.add_argument("--cos-lat", type=float, default=90.0, help="COS axis latitude / Dec in degrees")
    ap.add_argument("--cos-coords", choices=["icrs", "gal"], default="gal")
    ap.add_argument("--statistic", choices=["abs_delta_mean_residual", "abs_delta_q0"], default="abs_delta_q0")
    ap.add_argument("--n-random-axes", type=int, default=1000)
    ap.add_argument("--axis-seed", type=int, default=12345)
    ap.add_argument("--min-hemi-size", type=int, default=20)

    # Split-sample validation
    ap.add_argument("--n-splits", type=int, default=20, help="Number of random train/validation splits")
    ap.add_argument("--validation-fraction", type=float, default=0.5, help="Fraction of the Pantheon sample used for validation")
    ap.add_argument("--split-seed", type=int, default=20260322)
    ap.add_argument("--stratify-z", action="store_true", help="Stratify random splits by z quantile bins")
    ap.add_argument("--z-strat-bins", type=int, default=5)
    ap.add_argument("--n-validation-null", type=int, default=200, help="Fixed-axis sky-scramble null realizations on the validation subset")
    ap.add_argument("--validation-null-seed", type=int, default=24680)

    # Axis exported to CMB follow-up
    ap.add_argument("--export-axis-source", choices=["best_validated", "consensus"], default="best_validated")

    # Optional CMB follow-up
    ap.add_argument("--run-cmb-followup", action="store_true", help="After Pantheon split validation, run fixed-axis CMB follow-up")
    ap.add_argument("--map", default=None, help="CMB map FITS")
    ap.add_argument("--map-field", type=int, default=0)
    ap.add_argument("--mask", default=None, help="Optional mask FITS")
    ap.add_argument("--mask-field", type=int, default=0)
    ap.add_argument("--work-nside", type=int, default=256)
    ap.add_argument("--lmax-grid", type=str, default="8,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--mi-estimator", choices=["hist", "knn"], default="knn")
    ap.add_argument("--mi-bins", type=int, default=32)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--mi-sample-size", type=int, default=20000)
    ap.add_argument("--n-phase-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345, help="Master seed for the optional CMB phase-null section")
    ap.add_argument("--run-bayes", action="store_true")
    ap.add_argument("--bayes-nlive", type=int, default=500)
    ap.add_argument("--bayes-dlogz", type=float, default=0.5)
    ap.add_argument("--bayes-error-floor", type=float, default=1e-4)
    ap.add_argument("--plot", default=None, help="Optional CMB plot output path")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.pantheon_script and args.pantheon_script != "cos_pantheon_axis_scan.py":
        warn("--pantheon-script is ignored in the standalone build; using the internal Pantheon implementation.")
    pscan = make_internal_pantheon_api()

    out = {
        "pantheon_split_validation": run_split_sample_pantheon(args, pscan),
        "cmb_followup": None,
    }

    best_axis = out["pantheon_split_validation"]["best_validated_axis"]
    consensus_axis = out["pantheon_split_validation"]["consensus_axis"]

    chosen_axis_payload = best_axis
    if args.export_axis_source == "consensus":
        if consensus_axis is None:
            warn("Consensus axis is unavailable; falling back to best_validated axis.")
        else:
            chosen_axis_payload = consensus_axis

    out["exported_axis"] = chosen_axis_payload

    if args.run_cmb_followup:
        if not args.map:
            raise AnalysisError("--map is required when --run-cmb-followup is used.")
        cprobe = load_module_from_path("cos_crossprobe_fixed_axis_mod", args.crossprobe_script)
        chosen_axis = AggregatedAxisChoice(**chosen_axis_payload)
        out["cmb_followup"] = run_cmb_followup(args, cprobe, chosen_axis)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out["checkpoint"] = False
    out["complete"] = True
    out["checkpoint_out"] = make_checkpoint_path(args)
    out_path = Path(args.out)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    tmp.replace(out_path)
    log(f"Saved output: {args.out}")


if __name__ == "__main__":
    main()
