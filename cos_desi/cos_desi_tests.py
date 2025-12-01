#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COS-DESI Analysis Suite (v2, DESI-oriented, cleaned)

Feladat:
- DESI DR1 LSS clustering + random katalógusokból:
  * súlyozott, Landy–Szalay 2PCF normált párokkal,
  * hemiszférikus aszimmetria maszk-korrekcióval (adat - random),
  * filamentaritás proxy (MST teljes hossz) adat vs random.

Nem teljes "DESI publikációs" pipeline, de:
- kozmológiailag értelmezhető 2PCF,
- maszkolt hemiszféra-teszt,
- reprodukálható, paraméterezhető futás.

Most kiegészítve:
- cos_summary.json-ból kinyerhető komprimált skálaszámokkal,
- ensemble p-érték számítással mock-summaries_glob-ra (LCDM mock suite),
- egyszerű, emberi nyelvű LCDM-kompatibilitás / COS-interpretáció kiírással.

A normál futás (adat + random → 2PCF, hemiszféra, MST) CLI-ja nem változott.
"""

import argparse
import os
import time
import json
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as COSMO
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Egyszerű segédfüggvények (I/O, stb.)
# ----------------------------------------------------------------------


@dataclass
class Catalog:
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    w: Optional[np.ndarray] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(rows: Sequence[Sequence[float]], header: Sequence[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ----------------------------------------------------------------------
# Katalógus-beolvasás és egyszerű vágások
# ----------------------------------------------------------------------


def load_fits_catalog(path: str,
                      ra_col: str,
                      dec_col: str,
                      z_col: str,
                      w_cols: Optional[Sequence[str]] = None) -> Catalog:
    """
    FITS katalógus beolvasása (DESI-s LSS clustering jellegű).
    w_cols: lista a szorzandó súlyoszlopok neveivel, pl. ["WEIGHT"] vagy
            ["WEIGHT_FKP", "WEIGHT_SYS", ...]. Ha None, nincs súly.
    """
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        ra = np.asarray(data[ra_col], dtype=float)
        dec = np.asarray(data[dec_col], dtype=float)
        z = np.asarray(data[z_col], dtype=float)

        if w_cols is None or len(w_cols) == 0:
            w = None
        else:
            w_arr = np.ones_like(z, dtype=float)
            for col in w_cols:
                w_arr *= np.asarray(data[col], dtype=float)
            w = w_arr

    return Catalog(ra=ra, dec=dec, z=z, w=w)


def apply_basic_cuts(cat: Catalog, z_min: float, z_max: float) -> Catalog:
    """Egyszerű redshift-szelekció."""
    mask = (cat.z >= z_min) & (cat.z <= z_max)
    ra = cat.ra[mask]
    dec = cat.dec[mask]
    z = cat.z[mask]
    w = None if cat.w is None else cat.w[mask]
    return Catalog(ra=ra, dec=dec, z=z, w=w)


# ----------------------------------------------------------------------
# Geometria: z -> komoving távolság, gömbi -> derékszögű
# ----------------------------------------------------------------------


def comoving_distance_from_z(z: np.ndarray) -> np.ndarray:
    """
    Komoving távolság [Mpc] Planck18 kozmológiával (astropy).
    Egyszerű, vektoros wrapper.
    """
    zz = np.asarray(z, dtype=float)
    return COSMO.comoving_distance(zz).value  # Mpc


def sph_to_cart(ra_deg: np.ndarray, dec_deg: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (RA, DEC, r) -> (x, y, z), RA/DEC fokban, r tetszőleges egységben (itt Mpc).
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = r * cosd * np.cos(ra)
    y = r * cosd * np.sin(ra)
    z = r * np.sin(dec)
    return x, y, z


# ----------------------------------------------------------------------
# Párszámolás + Landy–Szalay 2PCF
# ----------------------------------------------------------------------


def pair_counts_kdtree(pos1: np.ndarray,
                       bins: np.ndarray,
                       pos2: Optional[np.ndarray] = None,
                       w1: Optional[np.ndarray] = None,
                       w2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """
    Súlyozott párok számolása KD-fával:
      - auto: pos1==pos2, vagy
      - cross: pos1 vs pos2.

    Visszatér:
      counts  : length nb array a súlyozott párok összegével.
      norm    : effektív "összes párok súlya" Landy–Szalay normalizáláshoz.
    """
    if pos2 is None:
        pos2 = pos1
        auto = True
    else:
        auto = False

    nb = len(bins) - 1
    r_max = float(bins[-1])

    n1 = pos1.shape[0]
    n2 = pos2.shape[0]

    if w1 is None:
        w1_arr = np.ones(n1, dtype=float)
    else:
        w1_arr = np.asarray(w1, dtype=float)

    if w2 is None:
        w2_arr = np.ones(n2, dtype=float)
    else:
        w2_arr = np.asarray(w2, dtype=float)

    tree2 = cKDTree(pos2)
    counts = np.zeros(nb, dtype=float)

    for i in range(n1):
        idxs = tree2.query_ball_point(pos1[i], r_max)
        if not idxs:
            continue
        w1i = w1_arr[i]
        w2_local = w2_arr[idxs]

        # távolságok
        dr = pos2[idxs] - pos1[i]
        rr = np.linalg.norm(dr, axis=1)
        hist, _ = np.histogram(rr, bins=bins)
        counts += w1i * (hist * w2_local.mean())  # egyszerű approx: átlag súly a binben

    if auto:
        # auto esetben a diag párokat (i,i) nem számoltuk, így a norm:
        n_eff = w1_arr.sum()
        norm = n_eff * (n_eff - 1.0) / 2.0
    else:
        norm = w1_arr.sum() * w2_arr.sum()

    return counts, int(norm)


def ls_norm_factors(wd: Optional[np.ndarray],
                    wr: Optional[np.ndarray],
                    nd: int,
                    nr: int) -> Tuple[float, float, float]:
    """
    Landy–Szalay effektív normalizáló tényezők.
    Egyszerűsített, súlyozott formulák.
    """
    if wd is None:
        sw = float(nd)
        sw2 = float(nd)
        N_dd = nd * (nd - 1) / 2.0
    else:
        wd = np.asarray(wd, dtype=float)
        sw = float(wd.sum())
        sw2 = float(np.sum(wd ** 2))
        N_dd = 0.5 * (sw * sw - sw2)

    if wr is None:
        sr = float(nr)
        sr2 = float(nr)
        N_rr = nr * (nr - 1) / 2.0
    else:
        wr = np.asarray(wr, dtype=float)
        sr = float(wr.sum())
        sr2 = float(np.sum(wr ** 2))
        N_rr = 0.5 * (sr * sr - sr2)

    N_dr = sw * sr

    return N_dd, N_dr, N_rr


def landy_szalay_2pcf(xyzd: np.ndarray,
                      xyzr: np.ndarray,
                      wd: Optional[np.ndarray],
                      wr: Optional[np.ndarray],
                      bins: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Landy–Szalay xi(r) = (DD - 2 DR + RR) / RR, súlyozott párokkal.

    Visszatér:
      xi            : xi(r) (nb-nyi float)
      counts        : dict("DD_w", "DR_w", "RR_w", "DD_norm", "DR_norm", "RR_norm")
      norms         : dict("N_dd", "N_dr", "N_rr") Landy–Szalay effektív normalizálók
    """
    nb = len(bins) - 1

    DD_w, DD_norm_pairs = pair_counts_kdtree(xyzd, bins, pos2=None, w1=wd)
    RR_w, RR_norm_pairs = pair_counts_kdtree(xyzr, bins, pos2=None, w1=wr)
    DR_w, DR_norm_pairs = pair_counts_kdtree(xyzd, bins, pos2=xyzr, w1=wd, w2=wr)

    # Ha valahol nincs pár, ott NaN
    DD_norm = np.full(nb, np.nan)
    RR_norm = np.full(nb, np.nan)
    DR_norm = np.full(nb, np.nan)

    if DD_norm_pairs > 0:
        DD_norm = DD_w / DD_norm_pairs
    if RR_norm_pairs > 0:
        RR_norm = RR_w / RR_norm_pairs
    if DR_norm_pairs > 0:
        DR_norm = DR_w / DR_norm_pairs

    N_dd_eff, N_dr_eff, N_rr_eff = ls_norm_factors(wd, wr, xyzd.shape[0], xyzr.shape[0])

    counts = {
        "DD_w": DD_w,
        "DR_w": DR_w,
        "RR_w": RR_w,
        "DD_norm": DD_norm,
        "DR_norm": DR_norm,
        "RR_norm": RR_norm,
    }
    norms = {
        "N_dd": N_dd_eff,
        "N_dr": N_dr_eff,
        "N_rr": N_rr_eff,
    }

    # xi(r) ~ (DD - 2DR + RR) / RR, a normált párokból
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (DD_norm - 2.0 * DR_norm + RR_norm) / RR_norm
    return xi, counts, norms


# ----------------------------------------------------------------------
# Hemiszférikus aszimmetria (maszk-korrekcióval)
# ----------------------------------------------------------------------


def hemispherical_asymmetry_mask_corrected(
        xyz_data: np.ndarray,
        xyz_rand: np.ndarray,
        w_data: Optional[np.ndarray],
        w_rand: Optional[np.ndarray],
        axes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Hemiszférikus aszimmetria maszk-korrekcióval:

      A_data(n) = (N_plus - N_minus)/(N_plus + N_minus)  (adatból)
      A_rand(n) = hasonló randomból (maszk+selection)
      A_corr(n) = A_data(n) - A_rand(n)

    n: hemiszférikus "pólus" irány (egységvektor).
    """
    if w_data is None:
        w_data = np.ones(xyz_data.shape[0], dtype=float)
    if w_rand is None:
        w_rand = np.ones(xyz_rand.shape[0], dtype=float)

    rhat_d = xyz_data / np.linalg.norm(xyz_data, axis=1, keepdims=True)
    rhat_r = xyz_rand / np.linalg.norm(xyz_rand, axis=1, keepdims=True)

    n_axes = axes.shape[0]
    A_data = np.zeros(n_axes, dtype=float)
    A_rand = np.zeros(n_axes, dtype=float)
    A_corr = np.zeros(n_axes, dtype=float)

    for i, n in enumerate(axes):
        # adat
        cos_theta_d = np.dot(rhat_d, n)
        plus_mask_d = (cos_theta_d >= 0.0)
        minus_mask_d = ~plus_mask_d
        Np_d = float(w_data[plus_mask_d].sum())
        Nm_d = float(w_data[minus_mask_d].sum())
        if Np_d + Nm_d > 0.0:
            A_data[i] = (Np_d - Nm_d) / (Np_d + Nm_d)
        else:
            A_data[i] = np.nan

        # random
        cos_theta_r = np.dot(rhat_r, n)
        plus_mask_r = (cos_theta_r >= 0.0)
        minus_mask_r = ~plus_mask_r
        Np_r = float(w_rand[plus_mask_r].sum())
        Nm_r = float(w_rand[minus_mask_r].sum())
        if Np_r + Nm_r > 0.0:
            A_rand[i] = (Np_r - Nm_r) / (Np_r + Nm_r)
        else:
            A_rand[i] = np.nan

        # korrigált
        A_corr[i] = A_data[i] - A_rand[i]

    # legnagyobb abszolút A_corr indexe
    valid = np.isfinite(A_corr)
    if np.any(valid):
        best_idx = int(np.argmax(np.abs(A_corr[valid])))
    else:
        best_idx = -1

    return A_data, A_rand, A_corr, best_idx


def fibonacci_sphere(n_axes: int,
                     randomize: bool = False,
                     jitter: float = 0.0,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Kvázi-egyenletes pontok az egységgömbön Fibonacci-sphere módszerrel.
    Opcionális kis véletlen jitter is adható.
    """
    if rng is None:
        rng = np.random.default_rng()

    if randomize:
        offset = rng.random() * 2.0
    else:
        offset = 0.0

    indices = np.arange(n_axes, dtype=float) + offset
    phi = (np.pi * (3.0 - np.sqrt(5.0)))  # golden angle

    z = 1.0 - (2.0 * indices + 1.0) / n_axes
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    axes = np.stack([x, y, z], axis=1)
    if jitter > 0.0:
        axes += jitter * rng.normal(size=axes.shape)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes


# ----------------------------------------------------------------------
# MST filamentaritás proxy
# ----------------------------------------------------------------------


def mst_total_length(xyz: np.ndarray) -> float:
    """
    Teljes MST hossz euklideszi térben (scipy minimum_spanning_tree).
    """
    # távolságmátrix
    d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
    mst = minimum_spanning_tree(d2)
    return float(mst.sum())


# ----------------------------------------------------------------------
# Random/alfélék, subsampling
# ----------------------------------------------------------------------


def subsample_indices(n: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """
    Egyszerű véletlen subsample: n-ből n_keep indexet választ (csere nélkül).
    """
    n_keep = min(n, n_keep)
    return rng.choice(n, size=n_keep, replace=False)


# ----------------------------------------------------------------------
# Ensemble analízis: cos_summary.json -> skálaszámok + p-értékek
# ----------------------------------------------------------------------


def extract_scalar_metrics_from_summary(summary: Dict) -> Dict[str, float]:
    """Extract compressed scalar statistics from a cos_summary.json dict.

    Returns a dict with at least:
      - max_abs_A_corr : max_n |A_corr(n)| (hemispherical asymmetry)
      - mst_delta      : MST_total_length_data - MST_total_length_random
      - xi_mean_all    : mean xi(r) over all bins
      - xi_mean_low    : mean xi(r) for 5 <= r <= 50 Mpc
      - xi_mean_high   : mean xi(r) for 50 <= r <= 150 Mpc
    """
    # Hemispherical asymmetry
    hemi = summary.get("hemispherical_asymmetry", {})
    A_corr = np.asarray(hemi.get("A_corr", []), dtype=float)
    if A_corr.size > 0:
        max_abs_A_corr = float(np.nanmax(np.abs(A_corr)))
    else:
        max_abs_A_corr = float("nan")

    # MST filamentarity proxy
    mst = summary.get("mst", {})
    mst_data = float(mst.get("MST_total_length_data", np.nan))
    mst_rand = float(mst.get("MST_total_length_random", np.nan))
    mst_delta = mst_data - mst_rand

    # 2PCF compressed statistics
    two = summary.get("two_point_correlation", {})
    r = np.asarray(two.get("r_centers_Mpc", []), dtype=float)
    xi = np.asarray(two.get("xi_LS", []), dtype=float)
    mask = np.isfinite(r) & np.isfinite(xi)
    if np.any(mask):
        r = r[mask]
        xi = xi[mask]
        xi_mean_all = float(np.mean(xi))
        low_mask = (r >= 5.0) & (r <= 50.0)
        high_mask = (r >= 50.0) & (r <= 150.0)
        xi_mean_low = float(np.mean(xi[low_mask])) if np.any(low_mask) else float("nan")
        xi_mean_high = float(np.mean(xi[high_mask])) if np.any(high_mask) else float("nan")
    else:
        xi_mean_all = float("nan")
        xi_mean_low = float("nan")
        xi_mean_high = float("nan")

    return {
        "max_abs_A_corr": max_abs_A_corr,
        "mst_delta": float(mst_delta),
        "xi_mean_all": xi_mean_all,
        "xi_mean_low": xi_mean_low,
        "xi_mean_high": xi_mean_high,
    }


def _p_value_two_sided(obs: float, samples: np.ndarray, center: Optional[float] = None) -> float:
    """Simple two-sided empirical p-value.

    If center is None, use the median of samples as the reference.
    Uses the absolute deviation |x - center| and a conservative
    (k+1)/(N+1) estimate.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0 or not np.isfinite(obs):
        return float("nan")
    if center is None or not np.isfinite(center):
        center = float(np.median(samples))
    dist_obs = abs(obs - center)
    dist = np.abs(samples - center)
    k = int(np.sum(dist >= dist_obs))
    return float((k + 1) / (samples.size + 1))


def _p_value_upper_tail(obs: float, samples: np.ndarray) -> float:
    """One-sided (upper-tail) empirical p-value with (k+1)/(N+1) convention."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0 or not np.isfinite(obs):
        return float("nan")
    k = int(np.sum(samples >= obs))
    return float((k + 1) / (samples.size + 1))


def run_ensemble_analysis(data_summary_path: str,
                          mock_glob: str,
                          outdir: str,
                          model_label: str = "LCDM") -> Dict:
    """Compute p-values for a data cos_summary.json vs. an ensemble of mock cos_summary.json.

    Parameters
    ----------
    data_summary_path : str
        Path to the cos_summary.json for the *data* run (current tracer/z-bin).
    mock_glob : str
        Glob pattern matching cos_summary.json files for the mocks (same tracer/z-bin).
    outdir : str
        Output directory where ensemble_pvalues.json and ENSEMBLE_SUMMARY.txt will be written.
    model_label : str
        Descriptive label for the mock model (e.g. "LCDM", "LCDM+sys", etc.).
    """
    if not os.path.exists(data_summary_path):
        print("[ensemble] Data summary JSON not found, skipping ensemble analysis.")
        return {}

    with open(data_summary_path, "r", encoding="utf-8") as f:
        data_summary = json.load(f)

    mock_paths = sorted(glob.glob(mock_glob))
    if not mock_paths:
        print("[ensemble] No mock summaries found for glob:", mock_glob)
        return {}

    mock_summaries = []
    for p in mock_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                mock_summaries.append(json.load(f))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[ensemble] Warning: could not read mock summary {p}: {e}")

    if not mock_summaries:
        print("[ensemble] No valid mock summaries could be loaded.")
        return {}

    # Extract scalar metrics
    data_metrics = extract_scalar_metrics_from_summary(data_summary)
    mock_metrics = {k: [] for k in data_metrics.keys()}
    for summ in mock_summaries:
        m = extract_scalar_metrics_from_summary(summ)
        for key, val in m.items():
            mock_metrics[key].append(val)

    mock_metrics_arr = {k: np.asarray(v, dtype=float) for k, v in mock_metrics.items()}

    results_metrics: Dict[str, Dict[str, float]] = {}
    for key, obs_val in data_metrics.items():
        arr = mock_metrics_arr[key]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or not np.isfinite(obs_val):
            results_metrics[key] = {
                "data_value": float(obs_val),
                "mock_mean": float("nan"),
                "mock_std": float("nan"),
                "mock_p68": [float("nan"), float("nan")],
                "p_upper": float("nan"),
                "p_two_sided": float("nan"),
            }
            continue

        mock_mean = float(np.mean(arr))
        mock_std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        p16, p84 = np.percentile(arr, [16.0, 84.0])

        if key == "max_abs_A_corr":
            # large max|A_corr| is the interesting tail
            p_upper = _p_value_upper_tail(obs_val, arr)
            p_two = _p_value_two_sided(obs_val, arr, center=mock_mean)
        else:
            # treat deviations from the mock mean as interesting
            diff_samples = np.abs(arr - mock_mean)
            diff_obs = abs(obs_val - mock_mean)
            p_upper = _p_value_upper_tail(diff_obs, diff_samples)
            p_two = _p_value_two_sided(obs_val, arr, center=mock_mean)

        results_metrics[key] = {
            "data_value": float(obs_val),
            "mock_mean": mock_mean,
            "mock_std": mock_std,
            "mock_p68": [float(p16), float(p84)],
            "p_upper": float(p_upper),
            "p_two_sided": float(p_two),
        }

    ensemble = {
        "model_label": model_label,
        "n_mocks": len(mock_summaries),
        "metrics": results_metrics,
    }

    json_path = os.path.join(outdir, "ensemble_pvalues.json")
    save_json(ensemble, json_path)

    # Human-readable summary with a heuristic interpretation
    txt_path = os.path.join(outdir, "ENSEMBLE_SUMMARY.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("COS-DESI Ensemble p-value summary\n")
        f.write("===================================\n")
        f.write(f"Model label : {model_label}\n")
        f.write(f"# mocks     : {ensemble['n_mocks']}\n\n")

        for key, info in results_metrics.items():
            f.write(f"[{key}]\n")
            f.write(f"  data value        : {info['data_value']:.6e}\n")
            if np.isfinite(info["mock_mean"]):
                f.write(f"  mock mean ± std   : {info['mock_mean']:.6e} ± {info['mock_std']:.6e}\n")
                f.write(
                    f"  mock 68% interval : "
                    f"[{info['mock_p68'][0]:.6e}, {info['mock_p68'][1]:.6e}]\n"
                )
            f.write(f"  p_upper (1-sided) : {info['p_upper']:.3g}\n")
            f.write(f"  p_two_sided       : {info['p_two_sided']:.3g}\n\n")

        # Heuristic interpretation for COS / LCDM compatibility
        f.write("Heuristic interpretation (for LCDM vs COS):\n")
        f.write("  - p_two_sided > 0.05 : compatible with LCDM\n")
        f.write("  - 0.01 < p_two_sided <= 0.05 : mild tension (interesting for COS)\n")
        f.write("  - p_two_sided <= 0.01 : strong tension (robust COS candidate)\n\n")

        for key, info in results_metrics.items():
            p = info["p_two_sided"]
            if not np.isfinite(p):
                continue
            if p <= 0.01:
                level = "strong tension"
            elif p <= 0.05:
                level = "mild tension"
            else:
                level = "compatible"
            f.write(f"  {key}: {level} with {model_label} (p={p:.3g}).\n")

        f.write("\nCOS interpretation guide:\n")
        f.write("  - max_abs_A_corr : large values favour hemispherical anisotropy;\n")
        f.write("                     COS models with strong preferred directions are supported\n")
        f.write("                     if this is in strong/mild tension with LCDM.\n")
        f.write("  - mst_delta      : significant deviation from 0 indicates different filamentarity\n")
        f.write("                     than random; large |mst_delta| may support COS filament biases.\n")
        f.write("  - xi_mean_*      : persistent shifts vs mocks can indicate modified growth /\n")
        f.write("                     IR behaviour; strong tension could disfavour simple LCDM and\n")
        f.write("                     motivate specific COS-IR or anisotropic scenarios.\n")

    print("[ensemble] Saved ensemble p-value summary to:", json_path)
    print("[ensemble] Human-readable summary:", txt_path)
    return ensemble


# ----------------------------------------------------------------------
# main() – DESI adat + random → 2PCF, hemiszféra, MST (+ opcionális ensemble)
# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="COS-DESI Test Suite (DESI DR1 LSS oriented)")

    ap.add_argument("--data", required=True, help="Galaxy clustering catalog FITS (RA, DEC, Z, weights).")
    ap.add_argument("--randoms", required=True, help="Random catalog FITS (same footprint, weights).")
    ap.add_argument("--outdir", default="cos_desi_results", help="Output directory.")

    ap.add_argument("--ra-col", default="RA")
    ap.add_argument("--dec-col", default="DEC")
    ap.add_argument("--z-col", default="Z")

    ap.add_argument("--w-cols-data", default=None,
                    help="Comma-separated weight columns for data, e.g. WEIGHT or WEIGHT_FKP,WEIGHT_SYS,...")
    ap.add_argument("--w-cols-random", default=None,
                    help="Comma-separated weight columns for randoms (usually WEIGHT or WEIGHT_FKP).")

    ap.add_argument("--z-min", type=float, default=0.1)
    ap.add_argument("--z-max", type=float, default=0.8)

    ap.add_argument("--r-min", type=float, default=5.0, help="Min separation for 2PCF [Mpc].")
    ap.add_argument("--r-max", type=float, default=150.0, help="Max separation for 2PCF [Mpc].")
    ap.add_argument("--r-bins", type=int, default=20, help="Number of radial bins.")

    ap.add_argument("--subsample-2pcf", type=int, default=40000,
                    help="Max number of data galaxies for 2PCF sub-sample.")
    ap.add_argument("--subsample-hemi", type=int, default=80000,
                    help="Max number of data galaxies for hemispherical test.")
    ap.add_argument("--mst-n-max", type=int, default=5000,
                    help="Max number of points for MST (filamentarity).")

    ap.add_argument("--n-axes", type=int, default=256,
                    help="Number of axes (directions) for hemispherical asymmetry.")
    ap.add_argument("--axes-seed", type=int, default=1234,
                    help="Random seed for Fibonacci-sphere jitter.")
    ap.add_argument("--axes-jitter", type=float, default=0.0,
                    help="Small random jitter for axes (0 = pure Fibonacci sphere).")

    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for subsampling.")

    # Új, opcionális ensemble / p-érték kapcsolók
    ap.add_argument("--run-ensemble", action="store_true",
                    help="If set, also compute ensemble p-values using mock summaries.")
    ap.add_argument("--mock-summaries-glob", default=None,
                    help="Glob pattern for cos_summary.json files of mock runs (for p-values).")
    ap.add_argument("--mock-model-label", default="LCDM",
                    help="Label for the mock cosmology model used for the mocks (e.g. LCDM).")

    args = ap.parse_args()
    t0 = time.time()
    ensure_dir(args.outdir)

    def parse_wcols(s: Optional[str]) -> Optional[Sequence[str]]:
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    wcols_data = parse_wcols(args.w_cols_data)
    wcols_rand = parse_wcols(args.w_cols_random)

    # ---- adat + random beolvasása, vágások ----
    print("[*] Loading data:", args.data)
    cat = load_fits_catalog(args.data, args.ra_col, args.dec_col, args.z_col, wcols_data)
    cat = apply_basic_cuts(cat, args.z_min, args.z_max)
    print(f"    Data after cuts: {len(cat.ra)} rows (z in [{args.z_min}, {args.z_max}])")

    print("[*] Loading randoms:", args.randoms)
    ran = load_fits_catalog(args.randoms, args.ra_col, args.dec_col, args.z_col, wcols_rand)
    ran = apply_basic_cuts(ran, args.z_min, args.z_max)
    print(f"    Randoms after cuts: {len(ran.ra)} rows (z in [{args.z_min}, {args.z_max}])")

    # ---- Koordináták átváltása komoving xyz-re ----
    print("[*] Converting to comoving Cartesian coordinates")
    rd = comoving_distance_from_z(cat.z)
    xd, yd, zd = sph_to_cart(cat.ra, cat.dec, rd)
    xyzd = np.vstack([xd, yd, zd]).T

    rr = comoving_distance_from_z(ran.z)
    xr, yr, zr = sph_to_cart(ran.ra, ran.dec, rr)
    xyzr = np.vstack([xr, yr, zr]).T

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 2PCF (Landy–Szalay)
    # ------------------------------------------------------------------
    print("[*] 2PCF (Landy–Szalay) estimation")

    n_data_2pcf = min(args.subsample_2pcf, xyzd.shape[0])
    idx_d_2pcf = subsample_indices(xyzd.shape[0], n_data_2pcf, rng)
    xyzd_2pcf = xyzd[idx_d_2pcf]
    wd_2pcf = None if cat.w is None else cat.w[idx_d_2pcf]

    # randomból lehet több, de érdemes ugyanakkora nagyságrendűre venni
    n_rand_2pcf = min(xyzr.shape[0], 2 * n_data_2pcf)
    idx_r_2pcf = subsample_indices(xyzr.shape[0], n_rand_2pcf, rng)
    xyzr_2pcf = xyzr[idx_r_2pcf]
    wr_2pcf = None if ran.w is None else ran.w[idx_r_2pcf]

    bins = np.linspace(args.r_min, args.r_max, args.r_bins + 1)
    r_centers = 0.5 * (bins[1:] + bins[:-1])

    xi, counts, norms = landy_szalay_2pcf(xyzd_2pcf, xyzr_2pcf, wd_2pcf, wr_2pcf, bins)

    rows_2pcf = []
    for i in range(len(r_centers)):
        rows_2pcf.append([
            r_centers[i],
            xi[i],
            counts["DD_w"][i],
            counts["DR_w"][i],
            counts["RR_w"][i],
            counts["DD_norm"][i],
            counts["DR_norm"][i],
            counts["RR_norm"][i],
        ])

    save_csv(rows_2pcf,
             ["r_Mpc", "xi_LS",
              "DD_w", "DR_w", "RR_w",
              "DD_norm", "DR_norm", "RR_norm"],
             os.path.join(args.outdir, "two_point_correlation.csv"))

    plt.figure()
    plt.plot(r_centers, xi, marker=".")
    plt.xlabel("Separation r [Mpc]")
    plt.ylabel("xi_LS(r)")
    plt.title("Two-point correlation (Landy-Szalay, weighted, normalized)")
    plt.grid(True, linestyle=":")
    plt.savefig(os.path.join(args.outdir, "two_point_correlation.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Hemiszférikus aszimmetria
    # ------------------------------------------------------------------
    print("[*] Hemispherical asymmetry scan (mask-corrected)")

    n_data_hemi = min(args.subsample_hemi, xyzd.shape[0])
    idx_d_hemi = subsample_indices(xyzd.shape[0], n_data_hemi, rng)
    xyzd_hemi = xyzd[idx_d_hemi]
    wd_hemi = None if cat.w is None else cat.w[idx_d_hemi]

    n_rand_hemi = min(xyzr.shape[0], 2 * n_data_hemi)
    idx_r_hemi = subsample_indices(xyzr.shape[0], n_rand_hemi, rng)
    xyzr_hemi = xyzr[idx_r_hemi]
    wr_hemi = None if ran.w is None else ran.w[idx_r_hemi]

    axes_rng = np.random.default_rng(args.axes_seed)
    axes = fibonacci_sphere(args.n_axes, randomize=True,
                            jitter=args.axes_jitter, rng=axes_rng)

    A_data, A_rand, A_corr, best_idx = hemispherical_asymmetry_mask_corrected(
        xyzd_hemi, xyzr_hemi, wd_hemi, wr_hemi, axes
    )

    best_axis = {
        "idx": int(best_idx),
        "axis_vector": axes[best_idx].tolist() if best_idx >= 0 else None,
        "A_data": float(A_data[best_idx]) if best_idx >= 0 else None,
        "A_rand": float(A_rand[best_idx]) if best_idx >= 0 else None,
        "A_corr": float(A_corr[best_idx]) if best_idx >= 0 else None,
    }

    # mentés
    rows_hemi = []
    for i in range(axes.shape[0]):
        rows_hemi.append([
            axes[i, 0], axes[i, 1], axes[i, 2],
            A_data[i], A_rand[i], A_corr[i]
        ])
    save_csv(rows_hemi,
             ["nx", "ny", "nz", "A_data", "A_rand", "A_corr"],
             os.path.join(args.outdir, "hemispherical_asymmetry.csv"))

    plt.figure()
    plt.hist(A_corr[np.isfinite(A_corr)], bins=30, alpha=0.7)
    plt.xlabel("A_corr")
    plt.ylabel("Count")
    plt.title("Hemispherical asymmetry (A_corr = A_data - A_rand)")
    plt.grid(axis="y", linestyle=":")
    plt.savefig(os.path.join(args.outdir, "hemispherical_asymmetry_hist.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # MST filamentaritás proxy
    # ------------------------------------------------------------------
    print("[*] Filamentarity proxy (MST total length) on subsample")

    N_mst = min(args.mst_n_max, xyzd.shape[0], xyzr.shape[0])
    idx_d_mst = subsample_indices(xyzd.shape[0], N_mst, rng)
    idx_r_mst = subsample_indices(xyzr.shape[0], N_mst, rng)
    xyzd_mst = xyzd[idx_d_mst]
    xyzr_mst = xyzr[idx_r_mst]

    total_len_data = mst_total_length(xyzd_mst)
    total_len_rand = mst_total_length(xyzr_mst)

    with open(os.path.join(args.outdir, "mst_total_length_data.txt"), "w", encoding="utf-8") as f:
        f.write(f"{total_len_data:.10e}\n")
    with open(os.path.join(args.outdir, "mst_total_length_random.txt"), "w", encoding="utf-8") as f:
        f.write(f"{total_len_rand:.10e}\n")

    plt.figure()
    plt.bar(["data", "random"], [total_len_data, total_len_rand])
    plt.ylabel("MST total length")
    plt.title("Filamentarity proxy (MST total length)")
    plt.grid(axis="y", linestyle=":")
    plt.savefig(os.path.join(args.outdir, "mst_total_length.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Összegzés JSON + TXT
    # ------------------------------------------------------------------
    runtime = time.time() - t0

    cos_summary = {
        "config": {
            "data_file": os.path.basename(args.data),
            "randoms_file": os.path.basename(args.randoms),
            "z_min": args.z_min,
            "z_max": args.z_max,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "r_bins": args.r_bins,
            "subsample_2pcf": args.subsample_2pcf,
            "subsample_hemi": args.subsample_hemi,
            "mst_n_max": args.mst_n_max,
            "cosmology": "Planck18 (astropy)",
        },
        "data_counts": {
            "N_data_after_cuts": int(len(cat.ra)),
            "N_random_after_cuts": int(len(ran.ra)),
        },
        "two_point_correlation": {
            "r_centers_Mpc": r_centers.tolist(),
            "xi_LS": xi.tolist(),
            "DD_w": counts["DD_w"].tolist(),
            "DR_w": counts["DR_w"].tolist(),
            "RR_w": counts["RR_w"].tolist(),
            "DD_norm": counts["DD_norm"].tolist(),
            "DR_norm": counts["DR_norm"].tolist(),
            "RR_norm": counts["RR_norm"].tolist(),
            "N_dd_eff": norms["N_dd"],
            "N_dr_eff": norms["N_dr"],
            "N_rr_eff": norms["N_rr"],
        },
        "hemispherical_asymmetry": {
            "axes": axes.tolist(),
            "A_data": A_data.tolist(),
            "A_rand": A_rand.tolist(),
            "A_corr": A_corr.tolist(),
            "best_axis": best_axis,
        },
        "mst": {
            "N_mst": int(N_mst),
            "MST_total_length_data": float(total_len_data),
            "MST_total_length_random": float(total_len_rand),
        },
        "runtime_sec": runtime,
    }

    save_json(cos_summary, os.path.join(args.outdir, "cos_summary.json"))

    with open(os.path.join(args.outdir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("COS-DESI Analysis Suite Summary\n")
        f.write("================================\n")
        f.write(f"Data rows after cuts: {len(cat.ra)}\n")
        f.write(f"Random rows after cuts: {len(ran.ra)}\n")
        f.write(f"2PCF sample size (data/random): {xyzd_2pcf.shape[0]}/{xyzr_2pcf.shape[0]}\n")
        f.write(f"Hemispherical asymmetry sample size (data/random): {xyzd_hemi.shape[0]}/{xyzr_hemi.shape[0]}\n")
        f.write(f"Number of axes scanned: {len(axes)}\n")
        f.write(f"Best |A_corr| axis idx: {best_axis['idx']}, A_corr={best_axis['A_corr']:.6e}\n")
        f.write(f"MST N: {N_mst}\n")
        f.write(f"MST total length (data): {total_len_data:.6f}\n")
        f.write(f"MST total length (random): {total_len_rand:.6f}\n")
        f.write(f"Runtime (s): {runtime:.1f}\n")

    # Opcionális ensemble p-érték analízis
    if args.run_ensemble:
        if args.mock_summaries_glob:
            data_summary_path = os.path.join(args.outdir, "cos_summary.json")
            print("[ensemble] Running ensemble analysis with mocks glob:", args.mock_summaries_glob)
            run_ensemble_analysis(
                data_summary_path=data_summary_path,
                mock_glob=args.mock_summaries_glob,
                outdir=args.outdir,
                model_label=args.mock_model_label,
            )
        else:
            print("[ensemble] --run-ensemble set but --mock-summaries-glob is empty; skipping ensemble analysis.")

    print("[*] Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
