#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COS–QSF CMB modulation pipeline

This script generates the PDF figures referenced in Appendix F and saves an
`arrays.npz` bundle for reproducible re-plotting.

Key design choices (vs earlier drafts):
  - Deterministic RNG with an explicit Generator (no hidden global RNG coupling).
  - Consistent "observed space" comparison: recovered C_ell vs (C_ell * (B_ell P_ell)^2 + N_ell).
  - Pixel window handling is robust:
        * Prefer explicit FITS window file if provided,
        * else try healpy.pixwin (if available),
        * else fall back to a well-documented Gaussian approximation (with a warning).
  - Single-realization "difference map" uses identical phases for baseline & modulated cases.

Outputs (in --outdir):
  - COS_QSF_modulation.pdf
  - COS_QSF_spectra_input.pdf
  - COS_QSF_spectra_recovered_single.pdf
  - COS_QSF_recovered_input_binned.pdf
  - COS_QSF_map_modulated.pdf
  - COS_QSF_map_difference.pdf
  - arrays.npz

Example runs (as referenced in the thesis text):
  python COS_QSF_cmb_pipeline_best.py --nside 256 --lmax 767 --outdir outputs/ --nreal 50
  python COS_QSF_cmb_pipeline_best.py --nside 256 --lmax 767 --effective_lmax 767 --outdir outputs/ --nreal 50

Dependencies:
  numpy, matplotlib, healpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp

# -----------------------------
# Utilities
# -----------------------------

def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _float_or_none(x: str) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip()
    if x.lower() in {"none", "null", ""}:
        return None
    return float(x)

def load_baseline_from_csv(csv_path: Path, ell: np.ndarray) -> np.ndarray:
    """
    Load a baseline temperature spectrum C_ell from a CSV with at least two columns:
    ell, Cl  (header names flexible: ell/L/l, Cl/TT/Ctt).

    Interpolates onto the given ell grid. Missing low-ell values are filled with 0.
    """
    import csv

    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    # detect header
    header = [h.strip() for h in rows[0]]
    has_header = any(any(ch.isalpha() for ch in h) for h in header)

    if has_header:
        data_rows = rows[1:]
        colmap = {h.lower(): i for i, h in enumerate(header)}
        # best-effort column detection
        def find_col(candidates):
            for cand in candidates:
                for k, idx in colmap.items():
                    if cand in k:
                        return idx
            return None

        i_ell = find_col(["ell", "l"])
        i_cl  = find_col(["cl", "ctt", "tt"])
        if i_ell is None or i_cl is None:
            raise ValueError(
                f"Could not detect columns in {csv_path}. "
                f"Header={header} (need ell and Cl-like columns)."
            )
    else:
        data_rows = rows
        i_ell, i_cl = 0, 1

    ell_in = []
    cl_in = []
    for r in data_rows:
        if not r or len(r) <= max(i_ell, i_cl):
            continue
        try:
            li = float(r[i_ell])
            ci = float(r[i_cl])
        except ValueError:
            continue
        if li < 0:
            continue
        ell_in.append(li)
        cl_in.append(ci)

    if len(ell_in) < 5:
        raise ValueError(f"Not enough numeric data in {csv_path} (found {len(ell_in)} rows).")

    ell_in = np.asarray(ell_in, dtype=float)
    cl_in = np.asarray(cl_in, dtype=float)
    order = np.argsort(ell_in)
    ell_in = ell_in[order]
    cl_in = cl_in[order]

    cl_out = np.interp(ell, ell_in, cl_in, left=0.0, right=float(cl_in[-1]))
    cl_out[ell < 2] = 0.0
    cl_out = np.maximum(cl_out, 0.0)
    return cl_out

def toy_baseline(ell: np.ndarray, A: float = 1.0e-9, ell0: float = 80.0, ns: float = -2.2, damp: float = 1400.0) -> np.ndarray:
    """
    Smooth toy TT spectrum in (uK^2) units (arbitrary normalization).
    """
    cl = np.zeros_like(ell, dtype=float)
    m = ell >= 2
    x = np.maximum(ell[m] / ell0, 1e-12)
    cl[m] = A * (x ** ns) * np.exp(-(ell[m] / damp) ** 2)
    return cl

def make_modulation(ell: np.ndarray, model: str, amp: float,
                    comb_L0: int = 250, comb_width: int = 10,
                    gauss_mu: int = 250, gauss_sigma: float = 35.0) -> np.ndarray:
    """
    Multiplicative modulation M_ell such that C_ell^mod = M_ell * C_ell^base.
    """
    M = np.ones_like(ell, dtype=float)
    if model == "none":
        return M

    if model == "comb":
        # "comb": alternating sign narrow Gaussians at multiples of L0
        for k in range(1, int(ell.max() // comb_L0) + 1):
            center = k * comb_L0
            sgn = -1.0 if (k % 2 == 0) else +1.0
            M += sgn * amp * np.exp(-0.5 * ((ell - center) / comb_width) ** 2)
        return M

    if model == "gauss":
        M += amp * np.exp(-0.5 * ((ell - gauss_mu) / gauss_sigma) ** 2)
        return M

    raise ValueError(f"Unknown modulation model: {model}")

def weighted_bin_edges(lmin: int, lmax: int, nbins: int) -> np.ndarray:
    edges = np.linspace(lmin, lmax + 1, nbins + 1).astype(int)
    edges[0] = max(2, edges[0])
    edges[-1] = lmax + 1
    # ensure strictly increasing
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough bins; increase nbins or l-range.")
    return edges

def binned_weighted_mean(ell: np.ndarray, y: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin y(ell) using weights (2ell+1).
    Returns (bin_centers, y_binned).
    """
    w = 2.0 * ell + 1.0
    xc = []
    yb = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ell >= a) & (ell < b)
        if not np.any(m):
            continue
        ww = w[m]
        xc.append(float(np.average(ell[m], weights=ww)))
        yb.append(float(np.average(y[m], weights=ww)))
    return np.asarray(xc), np.asarray(yb)

def plot_pdf(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# -----------------------------
# Pixel window handling
# -----------------------------

def _pixwin_gaussian_approx(nside: int, ell: np.ndarray) -> np.ndarray:
    """
    Conservative Gaussian approximation to the HEALPix pixel window.

    HEALPix pixels are not Gaussian; this is a fallback when exact pixwin data
    are unavailable. We approximate an effective sigma from the pixel area.
    """
    npix = hp.nside2npix(nside)
    omega_pix = 4.0 * math.pi / npix  # sr
    # effective FWHM from equal-area disk: area = pi (theta/2)^2 -> theta = 2 sqrt(area/pi)
    theta = 2.0 * math.sqrt(omega_pix / math.pi)  # radians
    sigma = theta / math.sqrt(8.0 * math.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma * sigma)

def get_pixwin(nside: int, lmax: int, *,
              use_pixwin: bool = True,
              pixwin_fits: Optional[Path] = None,
              verbose: bool = True) -> np.ndarray:
    """
    Return P_ell (length lmax+1).

    Priority:
      1) explicit FITS file if provided (first extension data),
      2) healpy.pixwin if it works,
      3) Gaussian approximation with warning.
    """
    ell = np.arange(lmax + 1, dtype=float)

    if not use_pixwin:
        return np.ones(lmax + 1, dtype=float)

    if pixwin_fits is not None:
        try:
            from astropy.io import fits
            with fits.open(pixwin_fits) as hdul:
                data = np.array(hdul[1].data).squeeze()
            # Some pixwin FITS contain columns; accept either 1D array or table with "TEMPERATURE"
            if data.ndim == 1:
                P = data
            else:
                # try common names
                colnames = [c.upper() for c in hdul[1].columns.names]
                if "TEMPERATURE" in colnames:
                    P = np.array(hdul[1].data["TEMPERATURE"]).squeeze()
                elif "TT" in colnames:
                    P = np.array(hdul[1].data["TT"]).squeeze()
                else:
                    raise ValueError(f"Unrecognized pixwin FITS columns: {hdul[1].columns.names}")
            if len(P) < lmax + 1:
                raise ValueError(f"pixwin file too short: len={len(P)} need >= {lmax+1}")
            return np.asarray(P[:lmax+1], dtype=float)
        except Exception as e:
            if verbose:
                print(f"[pixwin] Failed to load pixwin from {pixwin_fits}: {e}")

    # Try healpy's built-in pixwin (may require external data in some environments)
    try:
        P = hp.pixwin(nside, lmax=lmax)
        P = np.asarray(P, dtype=float)
        if len(P) >= lmax + 1:
            return P[:lmax+1]
    except Exception as e:
        if verbose:
            print(f"[pixwin] healpy.pixwin unavailable: {e}")

    # Fallback
    if verbose:
        print("[pixwin] Falling back to Gaussian pixel-window approximation. "
              "For publication-grade results, provide --pixwin_fits or ensure healpy pixwin data are installed.")
    return _pixwin_gaussian_approx(nside, ell)

# -----------------------------
# Harmonic simulation (deterministic RNG)
# -----------------------------

def synalm_from_rng(cl: np.ndarray, lmax: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate alm (m>=0 storage) for an isotropic Gaussian real field with power spectrum cl.

    For m=0: alm is real with Var=cl_l
    For m>0: Re,Im ~ N(0, cl_l/2) so E[|alm|^2]=cl_l

    Returns complex alm array of size hp.Alm.getsize(lmax).
    """
    cl = np.asarray(cl, dtype=float)
    if cl.shape[0] < lmax + 1:
        raise ValueError("cl too short for lmax")
    cl = np.maximum(cl[:lmax+1], 0.0)

    size = hp.Alm.getsize(lmax)
    alm = np.zeros(size, dtype=np.complex128)

    for ell in range(lmax + 1):
        c = float(cl[ell])
        if c <= 0.0:
            continue
        # m=0 (real)
        idx0 = hp.Alm.getidx(lmax, ell, 0)
        alm[idx0] = complex(rng.normal(scale=math.sqrt(c)), 0.0)
        if ell == 0:
            continue
        # m=1..ell (complex)
        s = math.sqrt(c / 2.0)
        for m in range(1, ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            alm[idx] = complex(rng.normal(scale=s), rng.normal(scale=s))
    return alm

def apply_ell_filter(alm: np.ndarray, fl: np.ndarray, lmax: int) -> np.ndarray:
    fl = np.asarray(fl, dtype=float)
    if fl.shape[0] < lmax + 1:
        raise ValueError("filter too short")
    return hp.almxfl(alm, fl[:lmax+1], inplace=False)

def white_noise_map_uKarcmin(nside: int, sigma_uKarcmin: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Return (noise_map, N_ell_white) for a white noise level in uK-arcmin.

    N_ell for white pixel noise with variance sigma_pix^2 is sigma_pix^2 * Omega_pix (uK^2 * sr).
    """
    if sigma_uKarcmin <= 0:
        return np.zeros(hp.nside2npix(nside), dtype=float), 0.0

    npix = hp.nside2npix(nside)
    omega_pix = 4.0 * math.pi / npix  # sr
    area_arcmin2 = omega_pix * (180.0 / math.pi * 60.0) ** 2
    sigma_pix = sigma_uKarcmin / math.sqrt(area_arcmin2)  # uK per pixel
    noise = rng.normal(loc=0.0, scale=sigma_pix, size=npix).astype(float)

    N_ell = (sigma_pix ** 2) * omega_pix  # uK^2 * sr
    return noise, float(N_ell)

# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    nside: int = 256
    lmax: int = 767
    effective_lmax: Optional[int] = None
    leff_factor: float = 2.5
    seed: int = 42
    outdir: str = "outputs"
    baseline_csv: Optional[str] = None

    model: str = "comb"  # comb | gauss | none
    amp: float = 0.015
    comb_L0: int = 250
    comb_width: int = 10
    gauss_mu: int = 250
    gauss_sigma: float = 35.0

    beam_fwhm_arcmin: float = 0.0
    noise_uKarcmin: float = 0.0

    use_pixwin: bool = True
    pixwin_fits: Optional[str] = None

    nreal: int = 50
    bin_lmin: int = 30
    nbins: int = 20

# -----------------------------
# Pipeline
# -----------------------------

def run(cfg: Config) -> None:
    outdir = Path(cfg.outdir)
    safe_makedirs(outdir)

    lmax = int(cfg.lmax)
    nside = int(cfg.nside)
    if lmax > 3 * nside - 1:
        lmax = 3 * nside - 1
        print(f"[info] lmax clipped to 3*nside-1 = {lmax}")

    lmax_eval = int(cfg.effective_lmax) if cfg.effective_lmax is not None else int(min(lmax, math.floor(cfg.leff_factor * nside)))
    if lmax_eval < lmax:
        print(f"[info] Using effective_lmax={lmax_eval} for validation/binning/plots (pixel-scale safe)")

    ell = np.arange(lmax + 1, dtype=float)

    # Baseline spectrum
    if cfg.baseline_csv:
        Cl_base = load_baseline_from_csv(Path(cfg.baseline_csv), ell)
    else:
        Cl_base = toy_baseline(ell)

    # Modulation and modulated spectrum
    M = make_modulation(
        ell, cfg.model, cfg.amp,
        comb_L0=cfg.comb_L0, comb_width=cfg.comb_width,
        gauss_mu=cfg.gauss_mu, gauss_sigma=cfg.gauss_sigma
    )
    Cl_mod = np.maximum(Cl_base * M, 0.0)

    # Beam
    if cfg.beam_fwhm_arcmin and cfg.beam_fwhm_arcmin > 0:
        fwhm_rad = math.radians(cfg.beam_fwhm_arcmin / 60.0)
        B_ell = hp.gauss_beam(fwhm_rad, lmax=lmax)
    else:
        B_ell = np.ones(lmax + 1, dtype=float)

    # Pixel window
    pixwin_path = Path(cfg.pixwin_fits) if cfg.pixwin_fits else None
    P_ell = get_pixwin(nside, lmax, use_pixwin=cfg.use_pixwin, pixwin_fits=pixwin_path, verbose=True)

    W_ell = B_ell * P_ell  # window applied to signal alms
    W2 = W_ell ** 2

    # White noise (map domain) and its flat N_ell
    rng_master = np.random.default_rng(cfg.seed)
    noise_demo, N_ell_white = white_noise_map_uKarcmin(nside, cfg.noise_uKarcmin, rng_master)

    # Reference spectra in "observed" space (signal window + additive white noise)
    Cl_base_for_compare = Cl_base * W2 + N_ell_white
    Cl_mod_for_compare  = Cl_mod  * W2 + N_ell_white

    # -----------------
    # Figure 1: modulation M_ell
    # -----------------
    plt.figure()
    plt.plot(ell, M)
    plt.xlabel(r"$\ell$")
    plt.xlim(2, lmax_eval)
    plt.ylabel(r"$\mathcal{M}_\ell$")
    plt.title("QSF modulation")
    plot_pdf(outdir / "COS_QSF_modulation.pdf")

    # -----------------
    # Figure 2: input spectra (baseline vs modulated)
    # -----------------
    plt.figure()
    plt.loglog(ell[2:], Cl_base[2:], label="baseline")
    plt.loglog(ell[2:], Cl_mod[2:],  label="modulated")
    plt.xlabel(r"$\ell$")
    plt.xlim(2, lmax_eval)
    plt.ylabel(r"$C_\ell$")
    plt.legend()
    plt.title("Input spectra")
    plot_pdf(outdir / "COS_QSF_spectra_input.pdf")

    # -----------------
    # Single-realization demo maps & recovered spectrum
    # -----------------
    # Use a fixed unit-spectrum alm and rescale => identical phases for baseline vs modulated.
    rng_demo = np.random.default_rng(cfg.seed + 12345)
    Cl_unit = np.ones(lmax + 1, dtype=float)
    Cl_unit[:2] = 0.0
    alm_unit = synalm_from_rng(Cl_unit, lmax, rng_demo)

    alm_base = apply_ell_filter(alm_unit, np.sqrt(np.maximum(Cl_base, 0.0)), lmax)
    alm_modd = apply_ell_filter(alm_unit, np.sqrt(np.maximum(Cl_mod,  0.0)), lmax)

    alm_base_obs = apply_ell_filter(alm_base, W_ell, lmax)
    alm_modd_obs = apply_ell_filter(alm_modd, W_ell, lmax)

    m_base = hp.alm2map(alm_base_obs, nside=nside, lmax=lmax)
    m_mod  = hp.alm2map(alm_modd_obs, nside=nside, lmax=lmax)

    # Add the same noise realization to both (so the difference isolates modulation)
    if cfg.noise_uKarcmin > 0:
        m_base = m_base + noise_demo
        m_mod  = m_mod  + noise_demo

    # Map figure
    hp.mollview(m_mod, title="Modulated map (T)", unit="uK", cmap="coolwarm")
    plt.savefig(outdir / "COS_QSF_map_modulated.pdf", bbox_inches="tight")
    plt.close()

    hp.mollview(m_mod - m_base, title="Difference map (mod - base)", unit="uK", cmap="coolwarm")
    plt.savefig(outdir / "COS_QSF_map_difference.pdf", bbox_inches="tight")
    plt.close()

    # Single recovered spectrum
    cl_rec_single = hp.anafast(m_mod, lmax=lmax, iter=0)
    plt.figure()
    plt.loglog(ell[2:], Cl_mod_for_compare[2:], label=r"reference: $C_\ell^{mod}(B_\ell P_\ell)^2 + N_\ell$")
    plt.loglog(ell[2:], cl_rec_single[2:], label=r"recovered from map")
    plt.xlabel(r"$\ell$")
    plt.xlim(2, lmax_eval)
    plt.ylabel(r"$C_\ell$")
    plt.legend()
    plt.title("Recovered vs reference (single realization)")
    plot_pdf(outdir / "COS_QSF_spectra_recovered_single.pdf")

    # -----------------
    # Ensemble: recover and bin relative differences
    # -----------------
    # To keep strict reproducibility, we draw fresh seeds per realization from rng_master.
    seeds = rng_master.integers(low=0, high=2**32 - 1, size=int(cfg.nreal), dtype=np.uint64)

    rel_diffs = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        alm = synalm_from_rng(Cl_mod, lmax, rng)
        alm = apply_ell_filter(alm, W_ell, lmax)
        m = hp.alm2map(alm, nside=nside, lmax=lmax)
        if cfg.noise_uKarcmin > 0:
            noise_i, _ = white_noise_map_uKarcmin(nside, cfg.noise_uKarcmin, rng)
            m = m + noise_i
        cl_hat = hp.anafast(m, lmax=lmax, iter=0)
        rel = np.zeros_like(cl_hat)
        denom = np.maximum(Cl_mod_for_compare, 1e-60)
        rel[2:] = cl_hat[2:] / denom[2:] - 1.0
        rel_diffs.append(rel)

    rel_diffs = np.asarray(rel_diffs)  # (nreal, lmax+1)
    rel_mean = rel_diffs.mean(axis=0)
    rel_sem  = rel_diffs.std(axis=0, ddof=1) / math.sqrt(max(1, cfg.nreal))

    edges = weighted_bin_edges(cfg.bin_lmin, lmax_eval, cfg.nbins)
    bin_c, bin_mean = binned_weighted_mean(ell, rel_mean, edges)
    _,     bin_sem  = binned_weighted_mean(ell, rel_sem,  edges)

    mask_bin = bin_c <= lmax_eval
    plt.figure()
    plt.errorbar(bin_c[mask_bin], bin_mean[mask_bin], yerr=bin_sem[mask_bin], fmt="o", capsize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylim(-0.05, 0.05)
    plt.xlabel(r"binned $\ell$")
    plt.xlim(cfg.bin_lmin, lmax_eval)
    plt.ylabel(r"$\langle \hat C_\ell / C_\ell^{ref} - 1 \rangle$")
    plt.title(f"Binned relative difference (nreal={cfg.nreal})")
    plot_pdf(outdir / "COS_QSF_recovered_input_binned.pdf")

    # -----------------
    # Save arrays bundle
    # -----------------
    params = asdict(cfg)
    params["effective_lmax"] = int(lmax_eval)
    params["N_ell_white"] = float(N_ell_white)

    np.savez(
        outdir / "arrays.npz",
        ell=ell.astype(np.int32),
        M=M.astype(np.float64),
        B_ell=B_ell.astype(np.float64),
        P_ell=P_ell.astype(np.float64),
        W_ell=W_ell.astype(np.float64),
        C_ell_base=Cl_base.astype(np.float64),
        C_ell_mod=Cl_mod.astype(np.float64),
        C_ell_base_for_compare=Cl_base_for_compare.astype(np.float64),
        C_ell_mod_for_compare=Cl_mod_for_compare.astype(np.float64),
        rel_mean=rel_mean.astype(np.float64),
        rel_sem=rel_sem.astype(np.float64),
        bin_centers=bin_c.astype(np.float64),
        bin_rel_mean=bin_mean.astype(np.float64),
        bin_rel_sem=bin_sem.astype(np.float64),
        params_json=json.dumps(params, ensure_ascii=False, indent=2),
    )

    print(f"[done] Wrote outputs to: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COS–QSF CMB modulation pipeline (publication version).")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--lmax", type=int, default=767)
    p.add_argument("--effective_lmax", type=int, default=None,
                   help="Maximum multipole used for validation/binning/plots. "
                        "If omitted, uses min(lmax, floor(leff_factor*nside)).")
    p.add_argument("--leff_factor", type=float, default=2.5,
                   help="Factor for effective_lmax = floor(leff_factor*nside) when effective_lmax is not set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs")

    p.add_argument("--baseline_csv", type=str, default=None)

    p.add_argument("--model", type=str, default="comb", choices=["comb", "gauss", "none"])
    p.add_argument("--amp", type=float, default=0.015)
    p.add_argument("--comb_L0", type=int, default=250)
    p.add_argument("--comb_width", type=int, default=10)
    p.add_argument("--gauss_mu", type=int, default=250)
    p.add_argument("--gauss_sigma", type=float, default=35.0)

    p.add_argument("--beam_fwhm_arcmin", type=float, default=0.0)
    p.add_argument("--noise_uKarcmin", type=float, default=0.0)

    p.add_argument("--use_pixwin", action="store_true", help="Enable pixel window (default).")
    p.add_argument("--no_pixwin", action="store_true", help="Disable pixel window.")
    p.add_argument("--pixwin_fits", type=str, default=None, help="Optional FITS file with P_ell.")

    p.add_argument("--nreal", type=int, default=50)
    p.add_argument("--bin_lmin", type=int, default=30)
    p.add_argument("--nbins", type=int, default=20)
    return p


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    use_pixwin = True
    if args.no_pixwin:
        use_pixwin = False
    elif args.use_pixwin:
        use_pixwin = True  # explicit

    cfg = Config(
        nside=args.nside,
        lmax=args.lmax,
        effective_lmax=args.effective_lmax,
        leff_factor=args.leff_factor,
        seed=args.seed,
        outdir=args.outdir,
        baseline_csv=args.baseline_csv,
        model=args.model,
        amp=args.amp,
        comb_L0=args.comb_L0,
        comb_width=args.comb_width,
        gauss_mu=args.gauss_mu,
        gauss_sigma=args.gauss_sigma,
        beam_fwhm_arcmin=args.beam_fwhm_arcmin,
        noise_uKarcmin=args.noise_uKarcmin,
        use_pixwin=use_pixwin,
        pixwin_fits=args.pixwin_fits,
        nreal=args.nreal,
        bin_lmin=args.bin_lmin,
        nbins=args.nbins,
    )
    run(cfg)


if __name__ == "__main__":
    main()
