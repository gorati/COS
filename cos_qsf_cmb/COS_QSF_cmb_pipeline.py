#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSF CMB pipeline v5 - TT, with robust comparisons (TeX-safe)
What's new: (2*ell+1)-weighted binning, pixwin reference, "safe" L-range, denominator threshold.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


# ---------- helper functions ----------

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def has_header_csv(path: str) -> bool:
    try:
        with open(path, "r") as f:
            first = f.readline()
        return ("ell" in first) or ("Cl" in first) or ("C_ell" in first) or ("C" in first)
    except Exception:
        return False

def read_baseline_csv(path: str, lmax: int):
    """Two columns: ell, C_ell (TT). If shorter, interpolate up to lmax; set ell<2 to 0."""
    data = np.loadtxt(path, delimiter=",", skiprows=1) if has_header_csv(path) \
           else np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("baseline CSV format: two columns (ell, C_ell).")
    ell_in = data[:, 0].astype(int)
    C_in   = data[:, 1].astype(float)
    ell = np.arange(lmax+1)
    C   = np.interp(ell, ell_in, C_in, left=0.0, right=0.0)
    C[:2] = 0.0
    return ell, C

def generate_toy_baseline(lmax: int, A: float = 1e-9, ell_peak: float = 220.0, width: float = 90.0):
    """Toy, LCDM-like TT spectrum: one acoustic bump + decaying tail."""
    ell = np.arange(lmax+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bump = np.exp(-0.5*((ell-ell_peak)/width)**2)
        tail = 1.0/np.maximum(ell*(ell+1.0), 1.0)
        C = A * (3.0*bump + 0.1*tail) * np.exp(-ell/2000.0)
    C[:2] = 0.0
    return ell, C

def apply_beam_to_Cl(C_ell: np.ndarray, fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """Gaussian beam: C_ell -> C_ell * b_ell^2 ; fwhm in arcminutes."""
    if fwhm_arcmin <= 0:
        return C_ell
    fwhm_rad = (fwhm_arcmin/60.0) * (np.pi/180.0)
    b_ell = hp.gauss_beam(fwhm=fwhm_rad, lmax=lmax)
    return C_ell * (b_ell**2)

def bin_spectrum_weighted(Cl: np.ndarray, lmin: int = 2, lmax: int | None = None, dL: int = 25):
    """(2*ell+1)-weighted top-hat binning over the [lmin, lmax] range."""
    if lmax is None:
        lmax = len(Cl)-1
    edges = np.arange(lmin, lmax+1, dL)
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ell = np.arange(a, b)
        w = 2*ell + 1
        num = np.sum(w * Cl[a:b])
        den = np.sum(w)
        centers.append(0.5*(a+b-1))
        means.append(num/den if den > 0 else np.nan)
    return np.array(centers), np.array(means)

def noise_sigma_per_pixel(nside: int, noise_uKarcmin: float) -> float:
    """White noise (uK-arcmin) -> uK/pixel using the HEALPix pixel solid angle."""
    if noise_uKarcmin <= 0:
        return 0.0
    omega_pix = 4*np.pi / hp.nside2npix(nside)       # sr
    arcmin2_per_sr = (180.0/np.pi*60.0)**2
    return noise_uKarcmin / np.sqrt(arcmin2_per_sr/omega_pix)


# ---------- modulation templates ----------

def modulation_harmonic(ell: np.ndarray, lam1=0.05, lam3=0.02, L0=300.0, alpha=0.9):
    """Smooth, decaying sin/cos combination - 'harmonic' placeholder."""
    e = np.asarray(ell, float)
    M = (lam1*np.sin((e/max(L0, 1.0))**alpha) + lam3*np.cos((e/max(2*L0, 1.0))**alpha)) * np.exp(-e/2500.0)
    M[:2] = 0.0
    return M

def modulation_comb(ell: np.ndarray, L0=250.0, n_peaks=4, amp=0.015, width=45.0, decay=2500.0):
    """Alternating-sign Gaussian lobes around L0, 2L0, ... - 'comb' (L-selective)."""
    e = np.asarray(ell, float)
    M = np.zeros_like(e)
    for k in range(1, n_peaks+1):
        center = k*L0
        sign = (-1)**k
        M += sign*np.exp(-0.5*((e-center)/width)**2)
    M *= amp*np.exp(-e/decay)
    M[:2] = 0.0
    return M


# ---------- main run ----------

def main():
    ap = argparse.ArgumentParser(description="QSF CMB map-level pipeline (TT)")
    ap.add_argument("--baseline_csv", type=str, default=None,
                    help="CSV with two columns: ell,C_ell (TT). Header optional.")
    ap.add_argument("--nside", type=int, default=256, help="HEALPix NSIDE.")
    ap.add_argument("--lmax", type=int, default=None, help="Maximum multipole to use.")
    ap.add_argument("--seed", type=int, default=137, help="RNG seed.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    # instrument
    ap.add_argument("--beam_fwhm_arcmin", type=float, default=0.0,
                    help="Gaussian beam FWHM in arcminutes. 0 disables beam.")
    ap.add_argument("--noise_uKarcmin", type=float, default=0.0,
                    help="Map white noise level in uK-arcmin. 0 disables noise.")
    # modulation choice
    ap.add_argument("--model", type=str, choices=["harmonic", "comb"], default="comb",
                    help="Modulation template.")
    # harmonic parameters
    ap.add_argument("--lam1", type=float, default=0.02, help="Amplitude for sine term.")
    ap.add_argument("--lam3", type=float, default=0.01, help="Amplitude for cosine term.")
    ap.add_argument("--L0",   type=float, default=300.0, help="Scale parameter for harmonic model.")
    ap.add_argument("--alpha", type=float, default=0.9, help="Exponent in harmonic phase.")
    # comb parameters
    ap.add_argument("--comb_L0", type=float, default=250.0, help="Base spacing for comb peaks.")
    ap.add_argument("--n_peaks", type=int, default=4, help="Number of comb peaks.")
    ap.add_argument("--amp", type=float, default=0.015, help="Overall comb amplitude.")
    ap.add_argument("--width", type=float, default=45.0, help="Peak width (Gaussian sigma).")
    ap.add_argument("--decay", type=float, default=2500.0, help="Exponential high-L decay scale.")
    # statistics
    ap.add_argument("--nreal", type=int, default=20, help="Number of realizations for binned stats.")
    ap.add_argument("--dL",    type=int, default=25, help="Binning width Delta L.")
    ap.add_argument("--Lmin_stat", type=int, default=30,
                    help="Lower bin edge (cosmic-variance cut).")
    args = ap.parse_args()

    np.random.seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    lmax = args.lmax if args.lmax is not None else (3*args.nside - 1)
    lmax_eff = min(lmax, int(2.5*args.nside))  # conservative range for statistics

    # 1) baseline C_ell
    if args.baseline_csv and os.path.exists(args.baseline_csv):
        ell, C_ell_base = read_baseline_csv(args.baseline_csv, lmax)
    else:
        ell, C_ell_base = generate_toy_baseline(lmax)

    # 2) modulation: Delta C_ell / C_ell
    if args.model == "harmonic":
        M = modulation_harmonic(ell, lam1=args.lam1, lam3=args.lam3, L0=args.L0, alpha=args.alpha)
    else:
        M = modulation_comb(ell, L0=args.comb_L0, n_peaks=args.n_peaks,
                            amp=args.amp, width=args.width, decay=args.decay)

    # 3) modified spectrum + beam
    C_ell_mod  = np.clip(C_ell_base*(1.0+M), a_min=0.0, a_max=None)
    C_ell_base_beam = apply_beam_to_Cl(C_ell_base, args.beam_fwhm_arcmin, lmax)
    C_ell_mod_beam  = apply_beam_to_Cl(C_ell_mod,  args.beam_fwhm_arcmin, lmax)

    # 4) pixwin (pixel window) for the comparison reference
    w_l = hp.pixwin(args.nside, lmax=lmax)
    C_ell_base_for_compare = C_ell_base_beam * (w_l**2)
    C_ell_mod_for_compare  = C_ell_mod_beam  * (w_l**2)

    # 5) noise (uK-arcmin -> uK/pixel)
    sigma_pix = noise_sigma_per_pixel(args.nside, args.noise_uKarcmin)

    # --- Figure: injected modulation
    plt.figure()
    plt.plot(ell, M); plt.axhline(0, ls="--")
    plt.xlabel(r"$L$"); plt.ylabel(r"$\Delta C_\ell / C_\ell$")
    plt.title("QSF modulation (input)"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "COS_QSF_modulation.pdf")); plt.close()

    # --- Figure: baseline vs modulated (input)
    plt.figure()
    plt.loglog(ell[2:], C_ell_base[2:], label="Baseline")
    plt.loglog(ell[2:], C_ell_mod[2:],  label="Modulated (input)")
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell$")
    plt.title("Input spectra"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "COS_QSF_spectra_input.pdf")); plt.close()

    # --- baseline & modulated maps (same seed) + difference
    np.random.seed(args.seed)
    alm_base = hp.synalm(C_ell_base_beam, lmax=lmax, new=True, verbose=False)
    m_base   = hp.alm2map(alm_base, nside=args.nside, lmax=lmax, verbose=False)
    if sigma_pix > 0:
        m_base = m_base + np.random.normal(scale=sigma_pix, size=m_base.size)

    np.random.seed(args.seed)
    alm_mod = hp.synalm(C_ell_mod_beam, lmax=lmax, new=True, verbose=False)
    m_mod   = hp.alm2map(alm_mod, nside=args.nside, lmax=lmax, verbose=False)
    if sigma_pix > 0:
        m_mod = m_mod + np.random.normal(scale=sigma_pix, size=m_mod.size)

    m_diff = m_mod - m_base
    hp.mollview(m_mod, title="QSF-modulated CMB map (T)", unit=r"$\mu$K", norm="hist"); hp.graticule()
    plt.savefig(os.path.join(outdir, "COS_QSF_map_modulated.pdf")); plt.close()
    hp.mollview(m_diff, title="Difference map: modulated - baseline", unit=r"$\mu$K", norm="hist"); hp.graticule()
    plt.savefig(os.path.join(outdir, "COS_QSF_map_difference.pdf")); plt.close()

    # --- recovered spectrum from a single realization (beam x pixwin reference)
    Cl_hat_1 = hp.anafast(m_mod, lmax=lmax)
    plt.figure()
    plt.loglog(ell[2:], C_ell_mod_for_compare[2:], label="Input (beam $\\times$ pixwin)")
    plt.loglog(np.arange(len(Cl_hat_1))[2:], Cl_hat_1[2:], label="Recovered")
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell$")
    plt.title("Recovered vs input (single realization)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "COS_QSF_spectra_recovered_single.pdf")); plt.close()

    # --- multiple realizations: (2*ell+1)-weighted binned stats over the conservative range
    nreal = int(args.nreal); dL = int(args.dL)
    Lmin = int(args.Lmin_stat)
    centers_ref, Cb_mod_ref = bin_spectrum_weighted(C_ell_mod_for_compare, lmin=Lmin, lmax=lmax_eff, dL=dL)

    Cb_hats = []
    for r in range(nreal):
        np.random.seed(args.seed + r + 1)
        alm = hp.synalm(C_ell_mod_beam, lmax=lmax, new=True, verbose=False)
        m   = hp.alm2map(alm, nside=args.nside, lmax=lmax, verbose=False)
        if sigma_pix > 0:
            m = m + np.random.normal(scale=sigma_pix, size=m.size)
        Clh = hp.anafast(m, lmax=lmax)
        _, Cb_h = bin_spectrum_weighted(Clh, lmin=Lmin, lmax=lmax_eff, dL=dL)
        Cb_hats.append(Cb_h)
    Cb_hats = np.vstack(Cb_hats)
    Cb_mean = Cb_hats.mean(axis=0)
    Cb_std  = Cb_hats.std(axis=0, ddof=1)

    # small denominator threshold
    eps = 1e-18
    mask = Cb_mod_ref > eps
    rel     = Cb_mean[mask]/Cb_mod_ref[mask] - 1.0
    rel_err = Cb_std[mask]/(np.sqrt(nreal)*Cb_mod_ref[mask])

    plt.figure()
    plt.errorbar(centers_ref[mask], rel, yerr=rel_err, fmt='o')
    plt.axhline(0, ls='--')
    plt.ylim(-0.05, 0.05)  # +/-5%
    plt.xlabel(r"$L$"); plt.ylabel(r"$\hat C_L/C_L^{\rm mod}-1$")
    plt.title(f"Recovered / input (binned, mean $\\pm$ s.e., nreal={nreal})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "COS_QSF_recovered_input_binned.pdf")); plt.close()

    # --- numerical save (for reproduction)
    np.savez(os.path.join(outdir, "arrays.npz"),
             ell=ell,
             C_ell_base=C_ell_base, C_ell_mod=C_ell_mod,
             C_ell_base_beam=C_ell_base_beam, C_ell_mod_beam=C_ell_mod_beam,
             C_ell_base_for_compare=C_ell_base_for_compare, C_ell_mod_for_compare=C_ell_mod_for_compare,
             M=M,
             nside=args.nside, lmax=lmax, lmax_eff=lmax_eff, seed=args.seed,
             model=args.model,
             lam1=args.lam1, lam3=args.lam3, L0=args.L0, alpha=args.alpha,
             comb_L0=args.comb_L0, n_peaks=args.n_peaks, amp=args.amp, width=args.width, decay=args.decay,
             beam_fwhm_arcmin=args.beam_fwhm_arcmin, noise_uKarcmin=args.noise_uKarcmin,
             nreal=nreal, dL=dL, Lmin=Lmin, eps=eps)

    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
