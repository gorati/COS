#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COS vs Planck CMB Analyzer — v4.4.0
-----------------------------------
"""

from __future__ import annotations
import argparse, math, os, sys, json, hashlib, warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
import astropy

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): return it

# Optional healpy
try:
    import healpy as _hp
    _HAVE_HEALPY = True
except Exception:
    _HAVE_HEALPY = False

import ducc0
import ducc0.healpix as hp0

TRAPZ = getattr(np, "trapezoid", np.trapz)

def nside_to_npix(nside:int)->int: return 12*int(nside)*int(nside)
def npix_to_nside(npix:int)->int:
    nside = int(round((npix/12.0)**0.5))
    if 12*nside*nside != npix: raise ValueError(f"npix={npix} is not valid for any NSIDE")
    return nside
def alm_size(lmax:int)->int: return (lmax+1)*(lmax+2)//2
def alm_getidx(lmax:int, l:int, m:int)->int: return m*(2*lmax - m + 1)//2 + l  # m-major

# -------------------------- FITS reader --------------------------

def read_healpix_fits_map(path: str|Path,
                          prefer_colnames: Tuple[str,...]=("I_STOKES","TEMPERATURE","T","SIGNAL","MAP")
                         ) -> Tuple[np.ndarray, int, str, Dict]:
    path = str(path)
    hdul = fits.open(path, memmap=True)
    try:
        nside = None; ordering=None; header_store={}; candidate=None
        for h in hdul:
            hdr=h.header; data=h.data
            for key in ("NSIDE","ORDERING","ORDER","PIXTYPE","COORDSYS"):
                if key in hdr: header_store[key]=hdr[key]
            if nside is None and "NSIDE" in hdr: nside=int(hdr["NSIDE"])
            if ordering is None:
                if "ORDERING" in hdr: ordering = hdr["ORDERING"].strip().upper()
                elif "ORDER" in hdr: ordering = hdr["ORDER"].strip().upper()
            if data is None: continue
            has_cols = hasattr(h,"columns") or (isinstance(data,np.ndarray) and data.dtype.names is not None)
            if has_cols:
                if hasattr(h,"columns"):
                    cols=[c.name for c in h.columns]; getter=lambda name: np.asarray(h.data[name]).ravel()
                else:
                    cols=list(data.dtype.names); getter=lambda name: np.asarray(data[name]).ravel()
                chosen=None
                for name in prefer_colnames:
                    if name in cols:
                        arr=np.asarray(getter(name), dtype=np.float64)
                        if nside is None:
                            try: nside=npix_to_nside(arr.size)
                            except: continue
                        if arr.size==nside_to_npix(nside): chosen=arr; break
                if chosen is None:
                    for name in cols:
                        try: arr=np.asarray(getter(name), dtype=np.float64)
                        except: continue
                        if nside is None:
                            try: nside=npix_to_nside(arr.size)
                            except: continue
                        if arr.size==nside_to_npix(nside): chosen=arr; break
                if chosen is not None: candidate=chosen; break
            if isinstance(data,np.ndarray) and data.dtype.names is None and data.ndim==1:
                arr=np.asarray(data, dtype=np.float64)
                if nside is None:
                    try: nside=npix_to_nside(arr.size)
                    except: arr=None
                if arr is not None and arr.size==nside_to_npix(nside): candidate=arr; break
        if candidate is None or nside is None:
            raise RuntimeError("Nem találtam HEALPix térképoszlopot (T/I_STOKES/MAP...).")
        if ordering is None: ordering = header_store.get("ORDERING","RING").upper()
        if ordering not in ("RING","NEST","NESTED"): ordering="RING"
        if ordering.startswith("NEST"):
            hp_pix = hp0.Healpix_Base(int(nside), "NEST")
            ring_idx = hp_pix.nest2ring(np.arange(candidate.size, dtype=np.int64))
            candidate = candidate[ring_idx]; ordering="RING"
        return candidate.astype(np.float64, copy=False), int(nside), ordering, header_store
    finally:
        hdul.close()

# -------------------------- degrade (pixel-space) --------------------------

def degrade_ring_mean(map_ring_hi: np.ndarray, nside_hi: int, nside_lo: int) -> np.ndarray:
    if nside_hi % nside_lo != 0: raise ValueError(f"NSIDE degrade integer faktor kell: {nside_hi} nem osztható {nside_lo}-val")
    if map_ring_hi.size != 12*nside_hi*nside_hi: raise ValueError("Input map méret != 12*nside_hi^2")
    hp_hi_ring = hp0.Healpix_Base(int(nside_hi), "RING")
    idx_nest = hp_hi_ring.ring2nest(np.arange(map_ring_hi.size, dtype=np.int64))
    map_nest_hi = map_ring_hi[idx_nest]
    f = nside_hi // nside_lo; block = f*f; npix_lo = 12*nside_lo*nside_lo
    map_nest_hi = map_nest_hi.reshape(npix_lo, block)
    map_nest_lo = map_nest_hi.mean(axis=1)
    hp_lo_nest = hp0.Healpix_Base(int(nside_lo), "NEST")
    idx_ring = hp_lo_nest.nest2ring(np.arange(npix_lo, dtype=np.int64))
    return map_nest_lo[idx_ring]

# ---------------- SHT backend (ducc0.sht elsődleges, healpy fallback) ----------------

class SHTBackend:
    def __init__(self, nside:int):
        self.nside = int(nside)
        self.hp_ring = hp0.Healpix_Base(self.nside, "RING")

        # Gyűrű leírás (uint64 indexek!)
        npix = nside_to_npix(self.nside)
        ang = self.hp_ring.pix2ang(np.arange(npix, dtype=np.int64))
        theta_pix = np.ascontiguousarray(ang[:,0], dtype=np.float64)
        phi_pix   = np.ascontiguousarray(ang[:,1], dtype=np.float64)

        tol = 1e-13
        is_new = np.empty(npix, dtype=bool); is_new[0]=True
        dth = np.abs(theta_pix[1:] - theta_pix[:-1]); is_new[1:] = (dth > tol)
        ringstart_i64 = np.where(is_new)[0]
        nring = ringstart_i64.size

        nphi = np.empty(nring, dtype=np.uint64)
        theta = np.empty(nring, dtype=np.float64)
        phi0  = np.empty(nring, dtype=np.float64)
        for i in range(nring):
            a = int(ringstart_i64[i]); b = int(ringstart_i64[i+1]) if (i+1<nring) else npix
            nphi[i]  = np.uint64(b - a)
            theta[i] = float(theta_pix[a])
            phi0[i]  = float(phi_pix[a])

        self.theta64 = np.ascontiguousarray(theta, dtype=np.float64)
        self.phi064  = np.ascontiguousarray(phi0,  dtype=np.float64)
        self.nphi64  = np.ascontiguousarray(nphi,  dtype=np.uint64)
        self.rstart64= np.ascontiguousarray(ringstart_i64.astype(np.uint64), dtype=np.uint64)

        # ducc0.sht állapot
        self._have_dsht = False
        self._have_analysis = False
        self._have_synthesis= False
        try:
            import ducc0.sht as dsht
            self.dsht = dsht
            self._have_dsht = True
            self._have_analysis = hasattr(dsht, "analysis")
            self._have_synthesis= hasattr(dsht, "synthesis")
        except Exception:
            self.dsht = None
            self._have_dsht = False

        self._have_healpy = _HAVE_HEALPY
        self.SHT_OK = (self._have_dsht and self._have_synthesis) or self._have_healpy
        if not self.SHT_OK:
            warnings.warn("SHT backend nem elérhető (nincs ducc0.sht.synthesis és healpy sem). Reduced mode indul.")
        else:
            if self._have_dsht and not self._have_synthesis:
                warnings.warn("ducc0.sht elérhető, de 'synthesis' nincs → healpy fallback (ha van).")

    def map2alm(self, m_in: np.ndarray, lmax: int, nthreads: int = 0) -> np.ndarray:
        arr = np.ascontiguousarray(m_in, dtype=np.float64)
        if self._have_dsht:
            # Prefer analysis; ha hibázik (2D elvárás), essünk vissza
            if self._have_analysis:
                try:
                    return self.dsht.analysis(
                        map=arr, theta=self.theta64, nphi=self.nphi64, phi0=self.phi064, ringstart=self.rstart64,
                        lmax=int(lmax), mmax=int(lmax), spin=0, nthreads=int(nthreads), mode="STANDARD",
                    )
                except Exception:
                    warnings.warn("ducc0.sht.analysis hiba → adjoint_synthesis fallback.")
            # adjoint_synthesis (ha itt is 2D-t vár: fallback)
            try:
                return self.dsht.adjoint_synthesis(
                    map=arr, theta=self.theta64, nphi=self.nphi64, phi0=self.phi064, ringstart=self.rstart64,
                    lmax=int(lmax), mmax=int(lmax), spin=0, nthreads=int(nthreads), mode="STANDARD",
                    lstride=1, pixstride=1, theta_interpol=False, alm=None
                )
            except Exception as e:
                if self._have_healpy:
                    warnings.warn(f"ducc0.sht.adjoint_synthesis hiba → healpy fallback:\n{e}")
                else:
                    raise
        if self._have_healpy:
            alm = _hp.sphtfunc.map2alm(arr, lmax=int(lmax), iter=0, pol=False, use_weights=False)
            return np.asarray(alm, dtype=np.complex128, copy=False)
        raise RuntimeError("Sem ducc0.sht, sem healpy nem áll rendelkezésre map2alm-hoz.")

    def alm2map(self, alm: np.ndarray, lmax: int, nthreads: int = 0) -> np.ndarray:
        a = np.ascontiguousarray(alm, dtype=np.complex128)
        if self._have_dsht and self._have_synthesis:
            try:
                m = self.dsht.synthesis(
                    alm=a, theta=self.theta64, nphi=self.nphi64, phi0=self.phi064, ringstart=self.rstart64,
                    lmax=int(lmax), mmax=int(lmax), spin=0, nthreads=int(nthreads), mode="STANDARD",
                )
                return np.asarray(m, dtype=np.float64, copy=False)
            except Exception as e:
                if self._have_healpy:
                    warnings.warn(f"ducc0.sht.synthesis hiba → healpy fallback:\n{e}")
                else:
                    raise
        if self._have_healpy:
            m = _hp.sphtfunc.alm2map(a, nside=self.nside, lmax=int(lmax), verbose=False)
            return np.asarray(m, dtype=np.float64, copy=False)
        raise RuntimeError("Sem ducc0.sht.synthesis, sem healpy nincs alm2map-hoz.")

# -------------------------- Gaussian beam & smoothing --------------------------

def gaussian_beam_window(lmax:int, fwhm_arcmin:float)->NDArray[np.float64]:
    if fwhm_arcmin<=0: return np.ones(lmax+1, dtype=np.float64)
    sigma = (fwhm_arcmin/60.0) * np.pi/180.0 / np.sqrt(8.0*np.log(2.0))
    ell = np.arange(lmax+1, dtype=np.float64)
    return np.exp(-0.5*ell*(ell+1.0)*sigma*sigma)

def smooth_map_via_alm(sky:NDArray[np.float64], sht:SHTBackend, lmax:int, fwhm_arcmin:float,
                       nthreads:int=0, beam_cache:Optional[NDArray[np.float64]]=None)->NDArray[np.float64]:
    if fwhm_arcmin<=0: return sky
    alm = sht.map2alm(sky, lmax, nthreads=nthreads)
    win = beam_cache if beam_cache is not None else gaussian_beam_window(lmax, fwhm_arcmin)
    for l in range(lmax+1):
        wl = win[l]
        if wl==1.0: continue
        for m in range(0, l+1):
            alm[alm_getidx(lmax, l, m)] *= wl
    return sht.alm2map(alm, lmax, nthreads=nthreads)

# -------------------------- Power spectrum & sims --------------------------

def alm2cl(alm:NDArray[np.complex128], lmax:int)->NDArray[np.float64]:
    cl = np.zeros(lmax+1, dtype=np.float64)
    for l in range(lmax+1):
        s = 0.0
        a0 = alm[alm_getidx(lmax, l, 0)]
        s += (a0.real*a0.real + a0.imag*a0.imag)
        for m in range(1, l+1):
            a = alm[alm_getidx(lmax, l, m)]
            s += 2.0*(a.real*a.real + a.imag*a.imag)
        cl[l] = s/(2.0*l+1.0)
    return cl

def simulate_gaussian_sky_from_cl(nside:int, lmax:int, cl:NDArray[np.float64],
                                  seed:Optional[int]=None, nthreads:int=0)->Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    rng = np.random.default_rng(seed)
    alm = np.zeros(alm_size(lmax), dtype=np.complex128)
    for l in range(lmax+1):
        c = max(float(cl[l]), 0.0)
        if c==0.0:
            alm[alm_getidx(lmax, l, 0)] = 0.0+0.0j
        else:
            alm[alm_getidx(lmax, l, 0)] = rng.normal(scale=math.sqrt(c), size=1)[0] + 0.0j
        if c>0.0:
            sig = math.sqrt(c/2.0)
            for m in range(1, l+1):
                alm[alm_getidx(lmax, l, m)] = (rng.normal(scale=sig) + 1j*rng.normal(scale=sig))
    sht = SHTBackend(nside)
    sky = sht.alm2map(alm, lmax, nthreads=nthreads)
    return sky, alm

# -------------------------- Maszk & apodizáció --------------------------

def read_mask(path: str|Path, expected_nside: int) -> np.ndarray:
    m_in, nside_in, ordering, _ = read_healpix_fits_map(path)
    m_in = np.asarray(m_in, dtype=np.float64)
    if not ((m_in >= 0).all() and (m_in <= 1).all()):
        m_in = (m_in > 0).astype(np.float64)
    m = degrade_ring_mean(m_in, nside_in, expected_nside) if nside_in != expected_nside else m_in
    m = np.clip(m, 0.0, 1.0).astype(np.float64, copy=False)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    return m

def apodize_mask_gaussian(mask: NDArray[np.float64], nside:int, lmax:int, apod_arcmin:float,
                          nthreads:int=0)->NDArray[np.float64]:
    if apod_arcmin<=0: return mask
    sht = SHTBackend(nside)
    if not sht.SHT_OK:
        warnings.warn("SHT nem elérhető → maszk apodizáció kihagyva (apod_arcmin ignorálva).")
        return mask
    alm = sht.map2alm(mask, lmax, nthreads=nthreads)
    win = gaussian_beam_window(lmax, apod_arcmin)
    for l in range(lmax+1):
        wl = win[l]
        if wl==1.0: continue
        for m in range(0, l+1):
            alm[alm_getidx(lmax, l, m)] *= wl
    out = sht.alm2map(alm, lmax, nthreads=nthreads)
    return np.clip(out, 0.0, 1.0)

def apply_mask(map_in:NDArray[np.float64], mask:Optional[NDArray[np.float64]])->NDArray[np.float64]:
    return map_in if mask is None else (map_in*mask)

# -------------------------- Statisztikák --------------------------

def hemispherical_asymmetry_stat(sky:NDArray[np.float64], nside:int,
                                 mask:Optional[NDArray[np.float64]]=None,
                                 centers:Optional[NDArray[np.float64]]=None,
                                 hp:Optional[hp0.Healpix_Base]=None)->float:
    """Variance-aszimmetria hemiszféra-saját átlagokkal, maszk-súlyozva.
    r = |Var_HA - Var_HB| / (Var_HA + Var_HB), Var_H = ⟨(T - ⟨T⟩_H)^2⟩_H, ahol ⟨·⟩_H maszk-súlyozott.
    """
    sky = np.asarray(sky, dtype=np.float64)
    sky = np.nan_to_num(sky, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        w = np.ones_like(sky, dtype=np.float64)
    else:
        w = np.asarray(mask, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    if hp is None: hp = hp0.Healpix_Base(nside, "RING")
    if centers is None:
        low = 32 if nside>=64 else max(8, nside//2)
        hp_low = hp0.Healpix_Base(int(low), "RING")
        centers = hp_low.pix2ang(np.arange(nside_to_npix(low)))

    vecs = hp.pix2vec(np.arange(nside_to_npix(nside)))
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-15)

    max_ratio = 0.0
    for th, ph in centers:
        axis = np.array([math.sin(th)*math.cos(ph),
                         math.sin(th)*math.sin(ph),
                         math.cos(th)], dtype=np.float64)
        dots = vecs @ axis

        # szimmetrikus határkezelés: pontos egyenlítő kizárva mindkét félgömbből
        idxA = (dots > 0.0)
        idxB = (dots < 0.0)
        if not idxA.any() or not idxB.any():
            continue

        wA, tA = w[idxA], sky[idxA]
        wB, tB = w[idxB], sky[idxB]
        mA, mB = float(wA.sum()), float(wB.sum())
        if mA <= 1e-12 or mB <= 1e-12:
            continue

        muA = float((wA * tA).sum() / mA)
        muB = float((wB * tB).sum() / mB)
        varA = float((wA * (tA - muA)**2).sum() / mA)
        varB = float((wB * (tB - muB)**2).sum() / mB)

        denom = varA + varB
        if denom <= 1e-30:
            continue
        r = abs(varA - varB) / denom
        if r > max_ratio:
            max_ratio = r

    return float(max_ratio)

def parity_asymmetry_stat(cl:NDArray[np.float64], lmin:int=2, lmax:int=40)->float:
    """Publikációs low-ℓ paritás: P = |∑ (-1)^ℓ (2ℓ+1) C_ℓ| / ∑ (2ℓ+1) C_ℓ, ℓ∈[lmin,lmax]."""
    cl = np.asarray(cl, dtype=np.float64)
    cl = np.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)

    lmin = max(2, int(lmin))
    lmax = min(int(lmax), len(cl) - 1)
    if lmax < lmin:
        return 0.0

    ell = np.arange(lmin, lmax + 1, dtype=int)
    w = (2 * ell + 1).astype(np.float64)
    sgn = (-1.0) ** ell

    num = np.abs(np.sum(sgn * w * cl[ell]))
    den = np.sum(w * cl[ell])
    if den <= 1e-30:
        return 0.0
    return float(num / den)

def s12_stat_from_cl(cl:NDArray[np.float64], lmax:int, theta_min_deg:float=60.0)->float:
    import numpy.polynomial.legendre as npl
    cl = np.array(cl, dtype=float, copy=True)
    if cl.size>0: cl[0] = 0.0
    if cl.size>1: cl[1] = 0.0
    x = np.linspace(-1.0, math.cos(np.deg2rad(theta_min_deg)), 512)
    ell = np.arange(len(cl))
    coeffs = ((2*ell+1)/(4*np.pi))*cl
    Cx = npl.legval(x, coeffs)
    return float(TRAPZ(Cx*Cx, x))

def quadrupole_octopole_alignment(alm:NDArray[np.complex128], lmax:int)->float:
    def lowl_map(l):
        a = np.zeros_like(alm)
        for m in range(0, l+1):
            a[alm_getidx(lmax, l, m)] = alm[alm_getidx(lmax, l, m)]
        return a
    ns = 64
    sht = SHTBackend(ns); base_small = hp0.Healpix_Base(ns, "RING")
    def principal_axis(a):
        sky = sht.alm2map(a, lmax)
        vecs = base_small.pix2vec(np.arange(nside_to_npix(ns)))
        w = np.abs(sky); wsum = w.sum() + 1e-30
        I = (w[:,None,None]*(np.eye(3)[None,:,:] - vecs[:,:,None]*vecs[:,None,:])).sum(axis=0)/wsum
        evals, evecs = np.linalg.eigh(I)
        return evecs[:, np.argmin(evals)]
    ax2 = principal_axis(lowl_map(2)); ax3 = principal_axis(lowl_map(3))
    return abs(float(np.dot(ax2/np.linalg.norm(ax2), ax3/np.linalg.norm(ax3))))

def ring_scan_stat(sky:NDArray[np.float64], nside:int, mask:Optional[NDArray[np.float64]]=None,
                   radii_deg:Tuple[float,...]=(2,3,4,5,7,10,12,15), width_deg:float=0.5,
                   grid_nside:int=32, centers:Optional[NDArray[np.float64]]=None)->float:
    base = hp0.Healpix_Base(nside, "RING")
    if centers is None:
        gns = min(grid_nside, max(8, nside//4))
        g = hp0.Healpix_Base(int(gns), "RING")
        centers = g.pix2ang(np.arange(nside_to_npix(gns)))
    w_mask = None if mask is None else np.asarray(mask, dtype=float)
    def disc_indices(rad, th, ph):
        rr = base.query_disc(np.array([th, ph]), rad)
        idx = []
        for a,b in rr:
            idx.extend(range(int(a), int(b)))
        return np.fromiter(idx, dtype=np.int64)
    max_abs = 0.0
    for th, ph in centers:
        for rdeg in radii_deg:
            r = math.radians(rdeg); w = math.radians(width_deg)
            inner = disc_indices(r, th, ph)
            inner2 = disc_indices(max(r-w, 0.0), th, ph)
            inner_ring = np.setdiff1d(inner, inner2, assume_unique=False)
            outer = disc_indices(r+w, th, ph)
            outer_ring = np.setdiff1d(outer, inner, assume_unique=False)
            if inner_ring.size<10 or outer_ring.size<10: continue
            if w_mask is None:
                ti = sky[inner_ring].mean(); to = sky[outer_ring].mean()
            else:
                wi = w_mask[inner_ring]; ti = sky[inner_ring]
                wo = w_mask[outer_ring]; to = sky[outer_ring]
                si = np.sum(wi*ti); so = np.sum(wo*to)
                mi = np.sum(wi) + 1e-30; mo = np.sum(wo) + 1e-30
                ti = si/mi; to = so/mo
            s = (ti - to)
            if abs(s)>max_abs: max_abs = abs(s)
    return float(max_abs)

# -------------------------- Többszörös teszt-korrekciók --------------------------

def fdr_bh(pvals:List[float], alpha:float=0.05)->Tuple[List[float], float, List[bool]]:
    p = np.asarray(pvals, dtype=float); n = p.size
    order = np.argsort(p); ranked = p[order]
    q = np.empty_like(ranked); prev = 1.0
    for i in range(n-1, -1, -1):
        q[i] = min(prev, ranked[i]*n/(i+1)); prev = q[i]
    q_unsort = np.empty_like(q); q_unsort[order] = q
    thr = 0.0; rejs = [False]*n
    for i,pi in enumerate(ranked, start=1):
        if pi <= (i/n)*alpha: thr = (i/n)*alpha
    for i,pi in enumerate(p): rejs[i] = (pi <= thr and thr>0)
    return list(q_unsort), float(thr), rejs

def fdr_by(pvals:List[float], alpha:float=0.05)->Tuple[List[float], float, List[bool]]:
    p = np.asarray(pvals, dtype=float); n = p.size
    Hn = np.sum(1.0/np.arange(1, n+1))
    order = np.argsort(p); ranked = p[order]
    q = np.empty_like(ranked); prev = 1.0
    for i in range(n-1, -1, -1):
        q[i] = min(prev, ranked[i]*n*Hn/(i+1)); prev = q[i]
    q_unsort = np.empty_like(q); q_unsort[order] = q
    thr = 0.0; rejs = [False]*n
    for i,pi in enumerate(ranked, start=1):
        if pi <= (i/(n*Hn))*alpha: thr = (i/(n*Hn))*alpha
    for i,pi in enumerate(p): rejs[i] = (pi <= thr and thr>0)
    return list(q_unsort), float(thr), rejs

def holm_adjust(pvals:List[float])->List[float]:
    p = np.asarray(pvals, dtype=float); n = p.size
    order = np.argsort(p); ranked = p[order]
    adj = np.empty_like(ranked); prev = 0.0
    for i,pi in enumerate(ranked, start=1):
        val = (n - i + 1)*pi
        adj[i-1] = max(prev, min(1.0, val))
        prev = adj[i-1]
    adj_unsort = np.empty_like(adj); adj_unsort[order] = adj
    return list(adj_unsort)

# -------------------------- Dipólus & társai --------------------------

def parse_axis(s:str, coords:str="gal")->Tuple[float,float]:
    parts = s.replace(";",",").replace(" ",",").split(","); parts=[p for p in parts if p!=""]
    if len(parts)!=2: raise ValueError("Tengely: 'a,b' (pl. '227,-27')")
    a = float(parts[0]); b = float(parts[1])
    if coords=="gal":
        l_deg=a; b_deg=b; theta = math.radians(90.0 - b_deg); phi = math.radians((l_deg % 360.0))
    else:
        theta = math.radians(a); phi = math.radians(b % 360.0)
    return theta, phi

def sph_unitvec(theta:float, phi:float)->np.ndarray:
    return np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)], dtype=float)

def dipole_factors(nside:int, axis_theta:float, axis_phi:float)->NDArray[np.float64]:
    base = hp0.Healpix_Base(nside, "RING")
    vecs = base.pix2vec(np.arange(nside_to_npix(nside)))
    axis = sph_unitvec(axis_theta, axis_phi)
    return (vecs @ axis).astype(np.float64, copy=False)

def dipole_modulate_map(sky:NDArray[np.float64], nside:int, A:float, axis_theta:float, axis_phi:float)->NDArray[np.float64]:
    return sky * (1.0 + A * dipole_factors(nside, axis_theta, axis_phi))

def cos_ir_window(lmax:int, L0:int, eps:float, gamma:float)->NDArray[np.float64]:
    """Régi, polinomiális COS-IR ablak: Γ_ℓ = 1 - ε · [1 - ℓ/L0]_+^γ (0≤Γ≤1)."""
    ell = np.arange(lmax+1, dtype=float)
    L0 = max(1, int(L0)); eps=float(np.clip(eps,0.0,1.0)); gamma=max(0.0,float(gamma))
    x = np.clip(1.0 - ell/float(L0), 0.0, 1.0)
    return np.clip(1.0 - eps*(x**gamma), 0.0, 1.0)

def ir_window_exp(lmax:int, ell0:float, p:float)->NDArray[np.float64]:
    """Új, exponenciális IR-ablak: Γ_ℓ = exp[-(ℓ/ℓ0)^p]."""
    ell = np.arange(lmax+1, dtype=np.float64)
    ell0 = max(float(ell0), 1.0); p = max(float(p), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        x = (ell/ell0)**p if p>0 else np.zeros_like(ell)
        out = np.exp(-x)
    out[0] = 1.0
    return out

def split_low_high_alm(alm:NDArray[np.complex128], lmax:int, L0:int)->Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    low = np.zeros_like(alm); high = np.zeros_like(alm); L0=int(max(0,min(L0,lmax)))
    for l in range(lmax+1):
        for m in range(0, l+1):
            idx=alm_getidx(lmax,l,m); (low if l<=L0 else high)[idx]=alm[idx]
    return low, high

def weighted_low_alm(alm_low:NDArray[np.complex128], lmax:int, L0:int, kind:str="exp")->NDArray[np.complex128]:
    out = np.zeros_like(alm_low)
    if L0<=0: return out
    for l in range(lmax+1):
        if l==0: w=1.0
        elif l<=L0: w = math.exp(-float(l)/float(L0)) if kind=="exp" else 1.0
        else: w=0.0
        for m in range(0, l+1):
            idx=alm_getidx(lmax,l,m); out[idx]=alm_low[idx]*w
    return out

# -------------------------- CL betöltés --------------------------

def load_theory_cl(path:str|Path, lmax:int, cl_type:str="Cl")->NDArray[np.float64]:
    try:
        arr = np.loadtxt(path, dtype=float, ndmin=2, comments=('#',';','%'))
    except Exception:
        arr = np.loadtxt(path, dtype=float, ndmin=2, comments=('#',';','%'), delimiter=',')
    if arr.ndim!=2 or arr.shape[1]<2: raise ValueError("CL fájlhoz legalább két oszlop kell: ell, C_l/D_l")
    ell_in = arr[:,0].astype(int); val_in = arr[:,1].astype(float)
    if ell_in.min()<0: warnings.warn("CL fájlban negatív ell? Eltérő formátum gyanú.")
    cl = np.zeros(lmax+1, dtype=float)
    if ell_in.max()<lmax: warnings.warn(f"CL fájl ℓ_max={ell_in.max()} < LMAX={lmax}; a hiányzó ℓ-ek 0-k maradnak.")
    mask_valid=(ell_in>=0)&(ell_in<=lmax); ell_v=ell_in[mask_valid]; val_v=val_in[mask_valid]
    if cl_type.lower()=="dl":
        with np.errstate(divide="ignore", invalid="ignore"):
            ellf=ell_v.astype(float); denom=ellf*(ellf+1.0)/(2.0*np.pi); denom[denom==0]=np.inf
            cl_vals = val_v/denom
        cl[ell_v]=cl_vals
    else:
        cl[ell_v]=val_v
        ell_test = ell_v[(ell_v>=2)&(ell_v<=min(30,lmax))]
        if ell_test.size>0:
            with np.errstate(divide="ignore", invalid="ignore"):
                Dl_est=(ell_test*(ell_test+1)/(2.0*np.pi))*cl[ell_test]
            med=float(np.nanmedian(Dl_est))
            if med>1e5: warnings.warn(f"CL fájl 'Cl'-nek jelölt, de Dl-nek tűnik (median Dl~{med:.2g}). Ellenőrizd a --cl-type-ot.")
    return cl

# -------------------------- Stabil seedek --------------------------

def stable_sim_seed(base_seed:int, sim_index:int, salt:str)->int:
    s=f"{int(base_seed)}:{int(sim_index)}:{salt}".encode("utf-8",errors="ignore")
    h=hashlib.sha256(s).digest(); val=int.from_bytes(h[:8],"little",signed=False)
    return int(1 + (val % (2**31 - 2)))

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="COS vs Planck analyzer, v4.4.0")
    ap.add_argument("--map", required=True); ap.add_argument("--mask", default=None); ap.add_argument("--out", default="cos_out")
    ap.add_argument("--nside", type=int, default=None); ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--nsims", type=int, default=300); ap.add_argument("--fwhm-arcmin", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--hemi-grid-nside", type=int, default=None); ap.add_argument("--ring-grid-nside", type=int, default=32)
    ap.add_argument("--ring-width-deg", type=float, default=0.5); ap.add_argument("--ring-radii-deg", type=str, default=None)
    ap.add_argument("--alt", choices=["lcdm","dipole","cosir","sdipole","psfeature"], default="lcdm")
    ap.add_argument("--dipA", type=float, default=0.07); ap.add_argument("--dip-axis", type=str, default="227,-27")
    ap.add_argument("--dip-coords", choices=["gal","sph"], default="gal")
    # IR options
    ap.add_argument("--ir-kind", choices=["exp","poly"], default="exp",
                    help="IR window kind for --alt=cosir: exp ⇒ Γℓ=exp[-(ℓ/ℓ0)^p], poly ⇒ Γℓ=1-ε[1-ℓ/L0]_+^γ")
    ap.add_argument("--ir-L0", type=int, default=30)
    ap.add_argument("--ir-eps", type=float, default=0.2)
    ap.add_argument("--ir-gamma", type=float, default=1.0)
    ap.add_argument("--ir-p", type=float, default=1.5)
    # scale-dependent dipole (QHF proxy)
    ap.add_argument("--sdip-L0", type=int, default=40); ap.add_argument("--sdip-A0", type=float, default=0.03)
    ap.add_argument("--sdip-shape", choices=["step","exp"], default="exp"); ap.add_argument("--sdip-axis", type=str, default="227,-27")
    ap.add_argument("--sdip-coords", choices=["gal","sph"], default="gal")
    # simple IR feature
    ap.add_argument("--pf-type", choices=["step","tilt"], default="step"); ap.add_argument("--pf-L0", type=int, default=30)
    ap.add_argument("--pf-amp", type=float, default=0.2); ap.add_argument("--pf-tilt", type=float, default=0.0)
    ap.add_argument("--cl-file", type=str, default=None); ap.add_argument("--cl-type", choices=["Cl","Dl"], default="Cl")
    try:
        ap.add_argument("--match-seeds-across-alts", default=True, action=argparse.BooleanOptionalAction)
    except Exception:
        ap.add_argument("--match-seeds-across-alts", action="store_true", default=True)
        ap.add_argument("--no-match-seeds-across-alts", dest="match_seeds_across_alts", action="store_false")
    ap.add_argument("--mask-apod-arcmin", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    def logp(msg): print(msg, flush=True)

    sky_raw, nside_in, ordering, hdr = read_healpix_fits_map(args.map)
    logp(f"Loaded map: NSIDE_in={nside_in}, ORDERING={ordering}, NPIX={sky_raw.size}")
    nside = int(args.nside if args.nside else min(nside_in, 512))
    if nside != nside_in: logp(f"Using NSIDE={nside} (pixel degrade).")
    lmax = int(args.lmax if args.lmax else min(3*nside-1, 3*nside_in-1))
    logp(f"Using LMAX={lmax}")

    try:
        sht_probe = SHTBackend(nside)
        logp(f"Harmonic backend OK? {bool(sht_probe.SHT_OK)}")
    except Exception as e:
        warnings.warn("SHT backend init hiba: " + str(e))
        sht_probe = None

    sky = degrade_ring_mean(sky_raw, nside_in, nside) if (nside != nside_in) else sky_raw.copy()
    sky = np.asarray(sky, dtype=np.float64)
    sky = np.nan_to_num(sky, nan=0.0, posinf=0.0, neginf=0.0)

    beam_win = gaussian_beam_window(lmax, args.fwhm_arcmin) if args.fwhm_arcmin>0 else np.ones(lmax+1, float)
    if args.fwhm_arcmin>0:
        if sht_probe is None or not sht_probe.SHT_OK:
            warnings.warn("SHT nem elérhető → simítás kihagyva (fwhm ignorálva).")
        else:
            logp(f"Smoothing with Gaussian beam {args.fwhm_arcmin:.2f} arcmin...")
            sky = smooth_map_via_alm(sky, sht_probe, lmax, args.fwhm_arcmin, nthreads=args.threads, beam_cache=beam_win)

    mask = None
    if args.mask:
        mask = read_mask(args.mask, expected_nside=nside)
        if args.mask_apod_arcmin>0:
            mask = apodize_mask_gaussian(mask, nside, lmax, args.mask_apod_arcmin, nthreads=args.threads)
        logp(f"Mask loaded; masked fraction={(mask<=1e-6).mean():.3f}")

    have_sht = (sht_probe is not None) and sht_probe.SHT_OK
    if not have_sht:
        warnings.warn("Harmonic statok (parity/S12/QO) kihagyva, mert SHT nem elérhető ezen a rendszeren.")

    if args.ring_radii_deg:
        try: radii = tuple(float(x) for x in args.ring_radii_deg.replace(";",",").split(",") if x.strip()!="")
        except Exception: raise ValueError("Nem sikerült értelmezni --ring-radii-deg paramétert.")
    else:
        radii = (2,3,4,5,7,10,12,15)

    if args.hemi_grid_nside is None: hemi_low = 32 if nside>=64 else max(8, nside//2)
    else: hemi_low = int(args.hemi_grid_nside)
    hemi_centers = hp0.Healpix_Base(int(hemi_low), "RING").pix2ang(np.arange(nside_to_npix(hemi_low)))

    ring_gns = min(args.ring_grid_nside, max(8, nside//4))
    ring_centers = hp0.Healpix_Base(int(ring_gns), "RING").pix2ang(np.arange(nside_to_npix(ring_gns)))

    stats: Dict[str, float|str] = {}
    stats["hemi_asym"]   = hemispherical_asymmetry_stat(sky, nside, mask=mask, centers=hemi_centers)
    stats["ring_scan"]   = ring_scan_stat(sky, nside, mask=mask, radii_deg=radii,
                                          width_deg=args.ring_width_deg, grid_nside=args.ring_grid_nside,
                                          centers=ring_centers)
    if have_sht:
        alm_obs = sht_probe.map2alm(apply_mask(sky, mask) if mask is not None else sky, lmax, nthreads=args.threads)
        cl_obs = alm2cl(alm_obs, lmax)
        cl_obs_for_S12 = cl_obs.copy()
        if cl_obs_for_S12.size>0: cl_obs_for_S12[0] = 0.0
        if cl_obs_for_S12.size>1: cl_obs_for_S12[1] = 0.0
        stats["parity_lowL"] = parity_asymmetry_stat(cl_obs, lmin=2, lmax=min(40,lmax))
        stats["S12"]         = s12_stat_from_cl(cl_obs_for_S12, lmax, theta_min_deg=60.0)
        stats["QO_align"]    = quadrupole_octopole_alignment(alm_obs, lmax)
        np.save(outdir/"cl_obs.npy", cl_obs)
    else:
        stats["parity_lowL"] = "NA"
        stats["S12"]         = "NA"
        stats["QO_align"]    = "NA"

    print("Observed statistics:")
    for k in ["hemi_asym","parity_lowL","S12","QO_align","ring_scan"]:
        v = stats[k]
        if isinstance(v, str): print(f"  {k:>12s} = {v}")
        else:                  print(f"  {k:>12s} = {float(v):.6g}")

    cl_theory = None
    if args.cl_file:
        cl_theory = load_theory_cl(args.cl_file, lmax, cl_type=args.cl_type)
        if args.fwhm_arcmin>0 and have_sht:
            cl_theory = cl_theory * (beam_win*beam_win)

    th_ax = ph_ax = None; salt=f"alt={args.alt}"
    if args.alt=="dipole":
        th_ax, ph_ax = parse_axis(args.dip_axis, coords=args.dip_coords)
        print(f"Dipole modulation: A={args.dipA:.4g}, axis (theta,phi)=({math.degrees(th_ax):.2f}°, {math.degrees(ph_ax):.2f}°)")
        cl_base = cl_theory if cl_theory is not None and have_sht else None
        salt += f"|A={args.dipA}|axis={args.dip_axis}|coords={args.dip_coords}"
    elif args.alt=="cosir":
        if args.ir_kind=="exp":
            Gamma = ir_window_exp(lmax, args.ir_L0, args.ir_p)
            print(f"COS-IR(exp): L0={args.ir_L0}, p={args.ir_p}  ⇒ Gamma[0]={Gamma[0]:.3f}, Gamma[L0]={Gamma[min(args.ir_L0,lmax)]:.3f}")
        else:
            Gamma = cos_ir_window(lmax, args.ir_L0, args.ir_eps, args.ir_gamma)
            print(f"COS-IR(poly): L0={args.ir_L0}, eps={args.ir_eps}, gamma={args.ir_gamma}  ⇒ Gamma[0]={Gamma[0]:.3f}, Gamma[L0]={Gamma[min(args.ir_L0,lmax)]:.3f}")
        base = cl_theory if (cl_theory is not None and have_sht) else None
        cl_base = (base*(Gamma*Gamma)) if base is not None else None
        salt += f"|kind={args.ir_kind}|L0={args.ir_L0}|p={args.ir_p}|eps={args.ir_eps}|gamma={args.ir_gamma}"
    elif args.alt=="sdipole":
        th_ax, ph_ax = parse_axis(args.sdip_axis, coords=args.sdip_coords)
        print(f"SDipole: A0={args.sdip_A0:.4g}, L0={args.sdip_L0}, shape={args.sdip_shape}, "
              f"axis (theta,phi)=({math.degrees(th_ax):.2f}°, {math.degrees(ph_ax):.2f}°)")
        cl_base = cl_theory if (cl_theory is not None and have_sht) else None
        salt += f"|A0={args.sdip_A0}|L0={args.sdip_L0}|shape={args.sdip_shape}|axis={args.sdip_axis}|coords={args.sdip_coords}"
    elif args.alt=="psfeature":
        base = cl_theory if (cl_theory is not None and have_sht) else None
        if base is not None:
            ell = np.arange(lmax+1, dtype=float); out = base.copy(); L0=max(2,int(args.pf_L0))
            if args.pf_type=="step":
                a=float(np.clip(args.pf_amp,0.0,1.0)); out[ell<=L0]=(1.0-a)*out[ell<=L0]; print(f"PSFeature(step): L0={L0}, amp={a}")
            else:
                l0=float(L0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr=np.maximum(ell,1.0)/l0; fac=rr**float(args.pf_tilt)
                fac[:2]=1.0; out*=fac; print(f"PSFeature(tilt): L0={L0}, tilt={args.pf_tilt}")
            cl_base=np.maximum(out,0.0)
        else:
            cl_base=None
        salt += f"|type={args.pf_type}|L0={args.pf_L0}|amp={args.pf_amp}|tilt={args.pf_tilt}"
    else:
        cl_base = cl_theory if (cl_theory is not None and have_sht) else None

    if have_sht and cl_base is None:
        alm_obs = sht_probe.map2alm(apply_mask(sky, mask) if mask is not None else sky, lmax, nthreads=args.threads)
        cl_base = alm2cl(alm_obs, lmax)
        if args.fwhm_arcmin>0:
            cl_base = cl_base * (beam_win*beam_win)

    if have_sht:
        np.save(outdir/"cl_base.npy", cl_base)

    reduced_mode = not have_sht
    print(f"Running {args.nsims} surrogate skies under '{args.alt}' model (mask matched, reduced={reduced_mode})...")
    sims_vals = {k: [] for k in ["hemi_asym","parity_lowL","S12","QO_align","ring_scan"]}

    for i in tqdm(range(args.nsims)):
        seed_salt = (salt if args.match_seeds_across_alts else f"alt={args.alt}|indep")
        seed_i = stable_sim_seed(args.seed, i, seed_salt)

        if have_sht:
            sky_sim, alm_sim = simulate_gaussian_sky_from_cl(nside, lmax, cl_base, seed=seed_i, nthreads=args.threads)
            if args.alt=="dipole":
                dots = dipole_factors(nside, th_ax, ph_ax)
                sky_sim = sky_sim * (1.0 + args.dipA*dots)
            elif args.alt=="sdipole":
                dots = dipole_factors(nside, th_ax, ph_ax)
                alm_low, alm_high = split_low_high_alm(alm_sim, lmax, args.sdip_L0)
                map_low  = sht_probe.alm2map(alm_low,  lmax, nthreads=args.threads)
                map_high = sht_probe.alm2map(alm_high, lmax, nthreads=args.threads)
                if args.sdip_shape=="step":
                    map_low_mod = map_low * (1.0 + args.sdip_A0*dots); sky_sim = map_high + map_low_mod
                else:
                    alm_w = weighted_low_alm(alm_low, lmax, args.sdip_L0, kind="exp")
                    map_w = sht_probe.alm2map(alm_w, lmax, nthreads=args.threads)
                    mod_w = map_w * (1.0 + args.sdip_A0*dots)
                    sky_sim = map_high + map_low + (mod_w - map_w)
            sky_sim_masked = apply_mask(sky_sim, mask) if mask is not None else sky_sim
            alm_sim2 = sht_probe.map2alm(sky_sim_masked, lmax, nthreads=args.threads)
            cl_sim = alm2cl(alm_sim2, lmax)
            cl_sim_for_S12 = cl_sim.copy()
            if cl_sim_for_S12.size>0: cl_sim_for_S12[0]=0.0
            if cl_sim_for_S12.size>1: cl_sim_for_S12[1]=0.0

            sims_vals["hemi_asym"].append(hemispherical_asymmetry_stat(sky_sim, nside, mask=mask, centers=hemi_centers))
            sims_vals["parity_lowL"].append(parity_asymmetry_stat(cl_sim, lmin=2, lmax=min(40,lmax)))
            sims_vals["S12"].append(s12_stat_from_cl(cl_sim_for_S12, lmax, theta_min_deg=60.0))
            sims_vals["QO_align"].append(quadrupole_octopole_alignment(alm_sim2, lmax))
            sims_vals["ring_scan"].append(ring_scan_stat(sky_sim, nside, mask=mask, radii_deg=radii,
                                                         width_deg=args.ring_width_deg, grid_nside=args.ring_grid_nside,
                                                         centers=ring_centers))
        else:
            rng = np.random.default_rng(seed_i)
            noise = rng.normal(scale=float(np.std(sky))*1e-3, size=sky.size)
            sky_sim = sky + noise
            sims_vals["hemi_asym"].append(hemispherical_asymmetry_stat(sky_sim, nside, mask=mask, centers=hemi_centers))
            sims_vals["ring_scan"].append(ring_scan_stat(sky_sim, nside, mask=mask, radii_deg=radii,
                                                         width_deg=args.ring_width_deg, grid_nside=args.ring_grid_nside,
                                                         centers=ring_centers))
            sims_vals["parity_lowL"].append(np.nan)
            sims_vals["S12"].append(np.nan)
            sims_vals["QO_align"].append(np.nan)

    pvals: Dict[str, float|str] = {}
    for k,obs in stats.items():
        if isinstance(obs, str):
            pvals[k] = "NA"; continue
        arr = np.asarray([x for x in sims_vals[k] if (not isinstance(x, float) or np.isfinite(x))], dtype=float)
        if arr.size==0:
            pvals[k] = "NA"; continue
        if k=="S12":
            p = (np.count_nonzero(arr <= obs)+1.0)/(arr.size+1.0)
        else:
            p = (np.count_nonzero(arr >= obs)+1.0)/(arr.size+1.0)
        pvals[k]=float(p)

    print("Nominal (uncorrected) p-values:")
    for k in ["hemi_asym","parity_lowL","S12","QO_align","ring_scan"]:
        v = pvals[k]
        if isinstance(v,str): print(f"  {k:>12s} : {v}")
        else:                 print(f"  {k:>12s} : p = {float(v):.4g}")

    keys_real=[k for k in ["hemi_asym","parity_lowL","S12","QO_align","ring_scan"] if isinstance(pvals[k], float)]
    pvec=[float(pvals[k]) for k in keys_real]
    if len(pvec)>0:
        q_BH, thr_BH, rej_BH = fdr_bh(pvec, alpha=0.05)
        q_BY, thr_BY, rej_BY = fdr_by(pvec, alpha=0.05)
        p_Holm = holm_adjust(pvec); rej_Holm=[(p_Holm[i]<=0.05) for i in range(len(pvec))]
        print(f"FDR(BH) alpha=0.05 threshold={thr_BH:.4g}")
        print(f"FDR(BY) alpha=0.05 threshold={thr_BY:.4g}")
        print(f"Holm-FWER alpha=0.05 rejections: {sum(rej_Holm)}/{len(pvec)}")
    else:
        q_BH=[]; thr_BH=0.0; rej_BH=[]
        q_BY=[]; thr_BY=0.0; rej_BY=[]
        p_Holm=[]; rej_Holm=[]

    supportive_BH=[]
    for k,rej in zip(keys_real, rej_BH):
        if rej: supportive_BH.append(k)
    model_name={"lcdm":"Gauss–ΛCDM","dipole":"dipólus-modulált","cosir":"COS-IR","sdipole":"skála-függő dipólus (low-ℓ)","psfeature":"izotróp IR-feature"}[args.alt]
    verdict=(f"TÁMOGATÓ EREDMÉNY: {len(supportive_BH)} teszt ({', '.join(supportive_BH)}) BH-FDR után is szignifikáns a {model_name} referenciahoz képest."
             if supportive_BH else
             f"NINCS DÖNTŐ EVIDENCIA: a vizsgált COS-mintázatokra nem kaptunk BH-FDR utáni szignifikanciát a {model_name} referenciahoz képest.")
    print("\n" + "="*72 + "\nVÉGSŐ VERDIKT:\n" + verdict + "\n" + "="*72)

    prov={"backend":
              ("ducc0.sht" if (sht_probe and sht_probe._have_dsht and sht_probe._have_synthesis)
               else ("healpy" if (sht_probe and sht_probe._have_healpy) else "none")),
          "hemi_asym":"variance difference on hemispheres with hemisphere-specific means (mask-weighted)",
          "parity_lowL":"|∑_{ℓ=2..40} (-1)^ℓ (2ℓ+1) C_ℓ| / ∑_{ℓ=2..40} (2ℓ+1) C_ℓ (harmonic)",
          "S12":"∫ C(θ)^2 dcosθ, θ≥60° (monopole/dipole removed; harmonic)",
          "QO_align":"principal-axis inertia(|T|) proxy (ℓ=2,3; harmonic)",
          "ring_scan":"weighted annulus-mean contrast (real-space)",
          "cl_source":("theory" if args.cl_file else ("observed_pseudo" if have_sht else "NA")),
          "mask_apod":(f"gaussian_{args.mask_apod_arcmin:.2f}arcmin" if args.mask_apod_arcmin>0 else "none"),
          "ir_kind": (args.ir_kind if args.alt=="cosir" else None),
          "ir_params": ({"L0":args.ir_L0, "p":args.ir_p, "eps":args.ir_eps, "gamma":args.ir_gamma} if args.alt=="cosir" else None)}
    versions={"python":sys.version.split()[0],"numpy":np.__version__,
              "astropy":getattr(astropy,"__version__","unknown"),
              "ducc0":getattr(ducc0,"__version__","unknown"),
              "healpy":(getattr(_hp,"__version__","NA") if _HAVE_HEALPY else "NA")}
    out_json={"version":"4.4.0","map":str(args.map),"mask":str(args.mask) if args.mask else None,"out":str(args.out),
              "nside_in":nside_in,"nside":nside,"lmax":lmax,"mask_used":bool(args.mask),
              "fwhm_arcmin":args.fwhm_arcmin,"nsims":args.nsims,"seed":args.seed,"threads":args.threads,
              "match_seeds_across_alts":bool(args.match_seeds_across_alts),"alt_model":args.alt,
              "dipA":args.dipA if args.alt=="dipole" else None,"dip_axis":args.dip_axis if args.alt=="dipole" else None,"dip_coords":args.dip_coords if args.alt=="dipole" else None,
              "ir_kind":args.ir_kind if args.alt=="cosir" else None,
              "ir_L0":args.ir_L0 if args.alt=="cosir" else None,"ir_eps":args.ir_eps if args.alt=="cosir" else None,"ir_gamma":args.ir_gamma if args.alt=="cosir" else None,"ir_p":args.ir_p if args.alt=="cosir" else None,
              "sdip_L0":args.sdip_L0 if args.alt=="sdipole" else None,"sdip_A0":args.sdip_A0 if args.alt=="sdipole" else None,"sdip_shape":args.sdip_shape if args.alt=="sdipole" else None,
              "sdip_axis":args.sdip_axis if args.alt=="sdipole" else None,"sdip_coords":args.sdip_coords if args.alt=="sdipole" else None,
              "pf_type":args.pf_type if args.alt=="psfeature" else None,"pf_L0":args.pf_L0 if args.alt=="psfeature" else None,
              "pf_amp":args.pf_amp if (args.alt=="psfeature" and args.pf_type=="step") else None,
              "pf_tilt":args.pf_tilt if (args.alt=="psfeature" and args.pf_type=="tilt") else None,
              "stats":stats,"pvals":{k:pvals[k] for k in ["hemi_asym","parity_lowL","S12","QO_align","ring_scan"]},
              "provenance":prov,"versions":versions,
              "notes":"If backend='none', harmonic stats are NA; MC uses same mask/processing for comparability."}
    (outdir/"summary.json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir/"verdict.txt").write_text(verdict+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
