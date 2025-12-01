#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cmb_time_arrow_MI_scan_axes.py

Tengely-szkenneléses „időnyíl-teszt” CMB térképen, kölcsönös információ (MI)
alapján (CPU- és memóriatakarékos verzió).

Ötlet:

- CMB térkép (pl. Planck SMICA) + maszk beolvasása.
- Opcionálisan degradálás kisebb NSIDE-ra (pl. 256).
- Több ℓ_max értékig low-pass szűrés → skála-sorozat:
    X_0(ℓ_max[0]), X_1(ℓ_max[1]), ..., X_{k-1}(ℓ_max[k-1]).
- Adjacent skálapárokra:
    (ℓ_i, ℓ_{i+1}) minden i=0..k-2.
- Ugyanazon pixeleken levő értékpárokból
    MI(X_i, X_{i+1})
  becslése:
    * globálisan (full sky),
    * félgömbönként (A, B) minden tengely mellett.
- Definiálunk:
    ΔMI_i = | MI_A(ℓ_i, ℓ_{i+1}) - MI_B(ℓ_i, ℓ_{i+1}) |
  → ez egy skála-indexelt sorozat (i=0..k-2).
- Erre számolunk egy monotonicity-score-t (−1..+1).
- Ugyanezt sok véletlen tengelyre is, plusz opcionálisan egy COS-tengelyre.
- Megnézzük, hogy a COS-tengely abszolút monotonicitása (|mono|) hányadik
  percentilisbe esik a random tengelyek abszolút monotonicitás-eloszlásában.

Kimenet:

- JSON, benne:
  - lmax_grid és lmax_pairs (pl. [ [8,16], [16,24], ... ])
  - MI_full_pairs: globális MI(ℓ_i, ℓ_{i+1})
  - monotonicity_MI_full_pairs
  - tengelyenként:
    * lon, lat, is_cos_axis
    * delta_MI_pairs (ΔMI_i sorozat)
    * monotonicity_delta_MI
    * abs_monotonicity_delta_MI
  - cos_axis blokk:
    * monotonicity_delta_MI
    * abs_monotonicity_delta_MI
    * percentile_abs_vs_random
  - time_arrow_MI_assessment blokk:
    * "level": "none" | "weak" | "strong" | "undetermined"
    * "message_hu": szöveges értelmezés
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np

try:
    import healpy as hp
except Exception:
    hp = None


# ---------------------------------------------------------------------------
# Helper függvények
# ---------------------------------------------------------------------------

def require_healpy() -> None:
    if hp is None:
        raise RuntimeError(
            "A script futásához a 'healpy' csomag szükséges. "
            "Telepítés:  pip install healpy"
        )


def load_map(path: str) -> Tuple[np.ndarray, int]:
    """CMB térkép beolvasása FITS-ből, NSIDE meghatározás."""
    require_healpy()
    print(f"[info] Térkép beolvasása: {path}", flush=True)
    m = hp.read_map(path, dtype=float)
    nside = hp.get_nside(m)
    print(f"[info] Eredeti NSIDE={nside}, npix={hp.nside2npix(nside)}", flush=True)
    return m, nside


def degrade_to_work_nside(
    m: np.ndarray,
    mask: Optional[np.ndarray],
    work_nside: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Térkép (és opcionális maszk) degradálása work_nside-ra.

    Ha az eredeti NSIDE már kisebb vagy egyenlő, akkor nem degradálunk felfelé,
    csak figyelmeztetünk, és maradunk az eredeti NSIDE-nál.
    """
    require_healpy()
    nside_orig = hp.get_nside(m)
    if nside_orig == work_nside:
        print(f"[info] NSIDE már {nside_orig}, nem degradálunk.", flush=True)
        return m, mask
    if nside_orig < work_nside:
        print(
            f"[warn] Eredeti NSIDE={nside_orig} < work_nside={work_nside}, "
            "nem degradálunk felfelé.",
            flush=True,
        )
        return m, mask

    print(f"[info] Térkép degradálása NSIDE={nside_orig} → NSIDE={work_nside}", flush=True)
    m_d = hp.ud_grade(m, nside_out=work_nside, pess=False)

    mask_d = None
    if mask is not None:
        print(
            f"[info] Maszk degradálása NSIDE={hp.get_nside(mask)} → NSIDE={work_nside}",
            flush=True,
        )
        mask_d = hp.ud_grade(mask, nside_out=work_nside, pess=False)
        mask_d = np.clip(mask_d, 0.0, 1.0)

    return m_d, mask_d


def standardize_map(m: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Z-score normalizálás: (T - mean) / std, csak maszk által engedett pixeleken."""
    if mask is not None:
        good = mask > 0.5
        vals = m[good]
    else:
        vals = m
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("A térképen (maszk után) nincs érvényes pixel.")
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    if not np.isfinite(sigma) or sigma == 0.0:
        print("[warn] A térkép szórása 0 vagy nem véges, csak középértéket vonunk ki.", flush=True)
        return m - mu
    print(f"[info] Standardizálás: mean={mu:.4e}, std={sigma:.4e}", flush=True)
    return (m - mu) / sigma


def lowpass_map(m: np.ndarray, lmax: int) -> np.ndarray:
    """Low-pass: map2alm lmax-ig, majd alm2map ugyanarra az NSIDE-ra."""
    require_healpy()
    nside = hp.get_nside(m)
    print(f"[info] Low-pass szűrés: ℓ_max={lmax}", flush=True)
    alm = hp.map2alm(m, lmax=lmax)
    m_lp = hp.alm2map(alm, nside=nside, lmax=lmax)
    return m_lp


def precompute_pix_vectors(nside: int) -> np.ndarray:
    """NSIDE-hez pixelek irányvektorai, shape=(3,npix)."""
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
    Egy tengelyhez félgömb maszk (A,B).

    A koordinátarendszer jelenleg:
      - 'gal' (galaktikus), és feltesszük, hogy a térkép is ilyen.
    """
    require_healpy()
    axis_coords = axis_coords.lower()
    if axis_coords not in ("gal",):
        raise ValueError("axis_coords jelenleg csak 'gal' lehet ebben a verzióban.")

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
    """Maszkolt pixelek 1D-vektora."""
    if mask is None:
        return m.reshape(-1)
    good = mask > 0.5
    return m[good].reshape(-1)


def discrete_entropy(x: np.ndarray, bins: int = 64) -> float:
    """Diszkrét Shannon-entrópia a pixelek hisztogramjából (1D)."""
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
    Kétváltozós kölcsönös információ (MI) 2D hisztogramból.

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

    # pxy>0 pontok
    mask = pxy > 0
    pxy_nz = pxy[mask]

    px_exp = px[:, np.newaxis]
    py_exp = py[np.newaxis, :]
    denom = px_exp * py_exp
    denom_nz = denom[mask]

    # Csak azok, ahol denom>0
    valid = denom_nz > 0
    if not np.any(valid):
        return float("nan")

    pxy_v = pxy_nz[valid]
    denom_v = denom_nz[valid]

    return float(np.sum(pxy_v * np.log(pxy_v / denom_v)))


def monotonicity_score(values: np.ndarray) -> float:
    """
    „Időnyíl” jellegű irányosság (−1..+1):
      +1 ≈ növekvő trend, −1 ≈ csökkenő, 0 ≈ nincs tiszta tendencia.
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
    n darab véletlen tengely galaktikus koordinátában (egyenletes a gömbön).
    Vissza: shape=(n,2): lon[deg], lat[deg]
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
    Heurisztikus értelmezés a COS-tengely MI-alapú időnyíl-mutatójára.

    Itt az abszolút monotonicitást (|mono|) nézzük:

      - nagy |mono| → erősen rendezett („nyilas”) viselkedés,
      - kicsi |mono| → inkább zajos, nem trendszerű.

    A random tengelyek abszolút monotonicitás-eloszlásához képest
    meghatározzuk a COS-tengely helyzetét (percentilis).

    Visszaad:
      {
        "level": "none" | "weak" | "strong" | "undetermined",
        "message_hu": str,
        "cos_abs_mono": float or None,
        "percentile_abs_vs_random": float or None
      }
    """
    if cos_abs_mono is None or not np.isfinite(cos_abs_mono):
        return {
            "level": "undetermined",
            "message_hu": (
                "A COS-tengely MI-alapú időnyíl-statisztikáját ebben a futásban "
                "nem lehetett megbízhatóan kiértékelni (hiányzó vagy nem véges érték)."
            ),
            "cos_abs_mono": None,
            "percentile_abs_vs_random": None,
        }

    if rnd_abs_monos is None:
        return {
            "level": "undetermined",
            "message_hu": (
                "Nincsenek random tengelyekhez tartozó MI-időnyíl értékek, így a COS "
                "eredménye nem hasonlítható össze velük."
            ),
            "cos_abs_mono": float(cos_abs_mono),
            "percentile_abs_vs_random": None,
        }

    rnd_abs = np.asarray(rnd_abs_monos, dtype=float)
    rnd_abs = rnd_abs[np.isfinite(rnd_abs)]
    if rnd_abs.size == 0:
        return {
            "level": "undetermined",
            "message_hu": (
                "A random tengelyek MI-időnyíl értékei nem voltak érvényesek, "
                "így a COS-tengely elhelyezkedése nem meghatározható."
            ),
            "cos_abs_mono": float(cos_abs_mono),
            "percentile_abs_vs_random": None,
        }

    # Percentilis: P(|mono_random| <= |mono_COS|)
    count_le = int(np.count_nonzero(rnd_abs <= cos_abs_mono))
    percentile = (count_le + 0.5) / (rnd_abs.size + 1.0)

    # Heurisztikus küszöbök:
    # - strong: abs-mono percentilis > 0.95
    # - weak:   0.80 < percentilis ≤ 0.95
    # - none:   percentilis ≤ 0.80
    if percentile > 0.95:
        level = "strong"
        msg = (
            "Erős, szokatlan MI-alapú időnyíl-szerű mintázat látszik ebben a statisztikában: "
            "a COS-tengely abszolút monotonicitása nagyobb, mint a véletlen tengelyek "
            "döntő többségéé (>95%)."
        )
    elif percentile > 0.80:
        level = "weak"
        msg = (
            "Gyenge vagy mérsékelt MI-alapú időnyíl-szerű eltérés látszik ebben a statisztikában: "
            "a COS-tengely abszolút monotonicitása a véletlen tengelyek felső ~20%-ában van."
        )
    else:
        level = "none"
        msg = (
            "Ebben a MI-alapú statisztikában nem találtunk erős, szignifikáns időnyílra utaló "
            "eltérést: a COS-tengely abszolút monotonicitása nem különleges a véletlen "
            "tengelyek eloszlásához képest."
        )

    return {
        "level": level,
        "message_hu": msg,
        "cos_abs_mono": float(cos_abs_mono),
        "percentile_abs_vs_random": float(percentile),
    }


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
        description="CMB MI-alapú időnyíl-teszt: tengely-szkennelés ΔMI(ℓ-párok) monotóniája alapján."
    )
    ap.add_argument("--map", required=True, help="CMB térkép FITS (pl. SMICA)")
    ap.add_argument("--mask", default=None, help="Opcionális maszk FITS")
    ap.add_argument(
        "--work-nside",
        type=int,
        default=256,
        help="Munka-NSIDE (térkép degradálása erre). Default: 256",
    )
    ap.add_argument(
        "--lmax-grid",
        type=str,
        default="8,16,24,32,48,64,96,128,192,256",
        help="ℓ_max értékek vesszővel elválasztva, pl. '8,16,24,32,48,64,96,128,192,256'",
    )
    ap.add_argument(
        "--entropy-bins",
        type=int,
        default=64,
        help="Entrópia hisztogram binszáma az 1D entropiákhoz (default: 64)",
    )
    ap.add_argument(
        "--mi-bins",
        type=int,
        default=32,
        help="2D hisztogram binszáma a MI számításhoz (default: 32)",
    )
    ap.add_argument(
        "--n-axes",
        type=int,
        default=100,
        help="Véletlen tengelyek száma (default: 100)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Véletlen generátor seed",
    )
    ap.add_argument(
        "--cos-axis",
        type=str,
        default=None,
        help="COS-tengely lon,lat fokban, pl. '227,-27' (galaktikus).",
    )
    ap.add_argument(
        "--cos-coords",
        type=str,
        default="gal",
        help="COS-tengely koordinátarendszere (jelenleg csak 'gal').",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="cmb_time_arrow_MI_scan_axes.json",
        help="Kimeneti JSON fájl neve",
    )
    args = ap.parse_args()

    # ℓ_max lista
    try:
        l_grid: Sequence[int] = [
            int(x) for x in args.lmax_grid.replace(";", ",").split(",") if x.strip() != ""
        ]
    except Exception as e:
        raise SystemExit(f"Hibás --lmax-grid formátum: {args.lmax_grid!r}  ({e})")

    if len(l_grid) < 3:
        raise SystemExit("Legalább három ℓ_max értéket meg kell adni, hogy legyenek ℓ-párok.")

    print(f"[info] ℓ_max grid: {l_grid}", flush=True)
    print(f"[info] work_nside={args.work_nside}", flush=True)
    print(f"[info] Entrópia binszám (1D): {args.entropy_bins}", flush=True)
    print(f"[info] MI binszám (2D): {args.mi_bins}", flush=True)
    print(f"[info] Véletlen tengelyek száma: {args.n_axes}, seed={args.seed}", flush=True)

    rng = np.random.default_rng(args.seed)

    # Térkép + maszk beolvasása
    m_raw, nside_orig = load_map(args.map)
    mask_raw = None
    if args.mask is not None:
        mask_raw = hp.read_map(args.mask, dtype=float)
        print(f"[info] Eredeti maszk NSIDE={hp.get_nside(mask_raw)}", flush=True)

    # Degradálás work_nside-ra
    m_work, mask_work = degrade_to_work_nside(m_raw, mask_raw, args.work_nside)
    nside = hp.get_nside(m_work)
    print(f"[info] Munka-NSIDE={nside}, npix={hp.nside2npix(nside)}", flush=True)

    # Maszk "normálása" (0..1)
    if mask_work is not None:
        mask_work = np.clip(mask_work, 0.0, 1.0)

    # Standardizált térkép
    m_std = standardize_map(m_work, mask_work)

    # Pixelek irányvektorai
    print("[info] Pixelek irányvektorainak előkalkulálása...", flush=True)
    pix_vecs = precompute_pix_vectors(nside)
    print("[info] Pixelek irányvektorai készen.", flush=True)

    # Tengelylista: COS + random
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
            raise SystemExit(f"Hibás --cos-axis formátum: {args.cos_axis!r}  ({e})")
        print(
            f"[info] COS-tengely: lon={cos_axis_lon}, lat={cos_axis_lat} [{args.cos_coords}]",
            flush=True,
        )
        axes_list.append((cos_axis_lon, cos_axis_lat, True))

    for i in range(random_axes.shape[0]):
        lon_deg, lat_deg = random_axes[i]
        axes_list.append((float(lon_deg), float(lat_deg), False))

    n_axes_total = len(axes_list)
    print(f"[info] Tengelyek összesen (COS + random): {n_axes_total}", flush=True)

    # Low-pass térképek minden ℓ_max-ra
    maps_lp: List[np.ndarray] = []
    for lmax in l_grid:
        m_lp = lowpass_map(m_std, lmax=lmax)
        maps_lp.append(m_lp)

    # ℓ-párok (indexek és konkrét értékek)
    l_pairs: List[Tuple[int, int]] = []
    for i in range(len(l_grid) - 1):
        l_pairs.append((l_grid[i], l_grid[i + 1]))
    print(f"[info] ℓ-párok a MI-hez: {l_pairs}", flush=True)

    # Globális MI a ℓ-párokra (full sky, maszk után)
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

    # Félgömb maszkok minden tengelyre (egy NSIDE-en belül, ℓ-független)
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
            f"[info] Félgömb maszkok felépítve tengelyre [axis {axis_idx} | {tag}] "
            f"lon={lon_deg:.2f}, lat={lat_deg:.2f}",
            flush=True,
        )

    # ΔMI(ℓ-párok) minden tengelyre
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

    # COS-tengely abszolút monotonicitása és random eloszlás
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

    time_arrow_assessment = interpret_time_arrow_mi_result(cos_abs_mono, rnd_abs_monos)

    if time_arrow_assessment["percentile_abs_vs_random"] is not None:
        print(
            f"[info] COS |mono(ΔMI)| ≈ {time_arrow_assessment['cos_abs_mono']:.3f}, "
            f"percentilis(abs-mono) ≈ "
            f"{100.0 * time_arrow_assessment['percentile_abs_vs_random']:.1f}%",
            flush=True,
        )

    print(
        f"[info] MI-időnyíl értékelés: {time_arrow_assessment['message_hu']} "
        f"(szint={time_arrow_assessment['level']})",
        flush=True,
    )

    # JSON kiírás
    out_data: Dict[str, Any] = {
        "map": args.map,
        "mask": args.mask,
        "nside_orig": int(nside_orig),
        "nside_work": int(nside),
        "lmax_grid": [int(x) for x in l_grid],
        "lmax_pairs": [[int(a), int(b)] for (a, b) in l_pairs],
        "entropy_bins": int(args.entropy_bins),
        "mi_bins": int(args.mi_bins),
        "n_axes_total": int(n_axes_total),
        "n_axes_random": int(args.n_axes),
        "seed": int(args.seed),
        "MI_full_pairs": MI_full_arr.tolist(),
        "monotonicity_MI_full_pairs": float(mono_MI_full),
        "axes": [asdict(ar) for ar in axis_results],
        "cos_axis": {
            "lon_deg": float(cos_axis_lon) if cos_axis_lon is not None else None,
            "lat_deg": float(cos_axis_lat) if cos_axis_lat is not None else None,
            "coords": args.cos_coords,
        },
        "time_arrow_MI_assessment": time_arrow_assessment,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[info] Kimenet elmentve: {out_path}", flush=True)


if __name__ == "__main__":
    main()
