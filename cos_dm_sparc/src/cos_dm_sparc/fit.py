
from __future__ import annotations

"""
COS-DM SPARC fitter

Purpose
-------
This version keeps the smooth VT modulator but uses a tempered arctan SBeff
channel to interpolate between the too-aggressive bounded-tanh v6.7 channel
and the too-gentle direct-bounded v6.8 channel.

    u_S   = tanh(kappa_S * (z_S - s0S))
    eta_S = etaSamp * (2/pi) * atan(alphaS * u_S)

The goal is to retain the healthier internal behavior of v6.8 while allowing
a somewhat stronger, but still controlled, SBeff response.

Model (default: amplitude modulation)
-------------------------------------
For galaxy i:
    xV_i = log(max(Vflat_i, 100) / 100)
    xS_i = log(max(SBeff_i, 100) / 100)
    xT_i = (7 - T_i) / 3

    A_i = A * exp(beta_V * xV_i + beta_T * xT_i)  # SBeff handled in separate arctan channel

    g_X,i(r) =
        A_i * sqrt(a0 * g_bar(r))
        / (1 + (g_bar(r) / g1)^gamma)
        * exp(Bs * s(r))

where
    s(r) = d ln g_bar / d ln r.

Optional alternative:
    g1_i = g1 * exp(eta_VT + eta_S)

where eta_VT is tanh-saturated and eta_S is the bounded arctan SBeff channel.
This is controlled by --modulate amplitude|g1|hybrid.

Data needed
-----------
1) SPARC Newtonian mass-model directory (Rotmod_LTG extracted)
2) SPARC Galaxy Sample Table1.mrt

Outputs
-------
- *_fit.json
- *_summary.csv
- *_report.json

This is still a prototype / falsification code, not a full hierarchical
cosmology pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

A0_DEFAULT = 1.2e-10
KPC_M = 3.085677581e19

def normalize_galaxy_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"_rotmod$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace("_", "")
    return s.upper()

MRT_COL_LINE_RE = re.compile(
    r"^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+(.+?)\s{2,}"
)
MRT_SINGLE_COL_LINE_RE = re.compile(
    r"^\s*(\d+)\s+\S+\s+\S+\s+(.+?)\s{2,}"
)

LABEL_MAP = {
    "Galaxy": "Galaxy",
    "T": "T",
    "D": "D",
    "e_D": "e_D",
    "f_D": "f_D",
    "Inc": "Inc",
    "e_Inc": "e_Inc",
    "L[3.6]": "L36",
    "e_L[3.6]": "e_L36",
    "Reff": "Reff",
    "SBeff": "SBeff",
    "Rdisk": "Rdisk",
    "SBdisk": "SBdisk",
    "MHI": "MHI",
    "RHI": "RHI",
    "Vflat": "Vflat",
    "e_Vflat": "e_Vflat",
    "Q": "Q",
    "Ref.": "Ref",
}

NUMERIC_COLS = [
    "T", "D", "e_D", "f_D", "Inc", "e_Inc", "L36", "e_L36",
    "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q",
]

T_LABELS = {
    0: "S0", 1: "Sa", 2: "Sab", 3: "Sb", 4: "Sbc", 5: "Sc", 6: "Scd",
    7: "Sd", 8: "Sdm", 9: "Sm", 10: "Im", 11: "BCD",
}

def _parse_byte_layout(lines: list[str]) -> list[tuple[str, int, int, str]]:
    """Parse byte ranges from a CDS/MRT byte-by-byte description.

    Returns tuples: (normalized_label, start0, end0_exclusive, raw_label).
    """
    layout: list[tuple[str, int, int, str]] = []
    in_bytes_block = False
    for line in lines:
        if "Byte-by-byte Description of file" in line:
            in_bytes_block = True
            continue
        if not in_bytes_block:
            continue
        if line.startswith("Note ("):
            break
        m = MRT_COL_LINE_RE.match(line)
        if m:
            start_1b = int(m.group(1))
            end_1b = int(m.group(2))
            raw_label = m.group(3).strip().split()[0]
            norm = LABEL_MAP.get(raw_label)
            if norm is not None:
                layout.append((norm, start_1b - 1, end_1b, raw_label))
            continue
        m = MRT_SINGLE_COL_LINE_RE.match(line)
        if m:
            start_1b = int(m.group(1))
            raw_label = m.group(2).strip().split()[0]
            norm = LABEL_MAP.get(raw_label)
            if norm is not None:
                layout.append((norm, start_1b - 1, start_1b, raw_label))
            continue

    expected = {
        "Galaxy", "T", "D", "e_D", "f_D", "Inc", "e_Inc", "L36", "e_L36",
        "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref"
    }
    got = {x[0] for x in layout}
    missing = expected - got
    if missing:
        raise ValueError(f"Failed to parse complete Table1 layout; missing: {sorted(missing)}")
    layout.sort(key=lambda t: t[1])
    return layout


def _find_data_start(lines: list[str]) -> int:
    dashed = "-" * 80
    idxs = [i for i, line in enumerate(lines) if line.strip() == dashed]
    if not idxs:
        raise ValueError("Could not find dashed separator in Table1.mrt")
    start = idxs[-1] + 1
    if start >= len(lines):
        raise ValueError("Table1.mrt appears to contain no data rows")
    return start


def _looks_like_data_row(line: str, max_end: int) -> bool:
    del max_end
    if not line.strip():
        return False
    toks = line.split()
    if len(toks) != 19:
        return False
    galaxy, t_field, d_field = toks[0], toks[1], toks[2]
    if not galaxy:
        return False
    if not re.fullmatch(r"[+-]?\d+", t_field):
        return False
    if not re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)", d_field):
        return False
    return True


def parse_table1_mrt(path: Path) -> tuple[pd.DataFrame, dict]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    layout = _parse_byte_layout(lines)
    max_end = max(end for _, _, end, _ in layout)
    data_start = _find_data_start(lines)

    records = []
    bad_lines = 0
    ordered_names = [name for name, *_ in layout]
    for raw in lines[data_start:]:
        if not _looks_like_data_row(raw, max_end=max_end):
            if raw.strip():
                bad_lines += 1
            continue
        toks = raw.split()
        if len(toks) != len(ordered_names):
            bad_lines += 1
            continue
        rec = dict(zip(ordered_names, toks))
        records.append(rec)

    if not records:
        raise ValueError(f"No Table1 records parsed from {path}")

    df = pd.DataFrame.from_records(records)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Galaxy"].astype(str).str.len() > 0].copy()
    df["galaxy_key"] = df["Galaxy"].map(normalize_galaxy_name)
    df["T_label"] = df["T"].round().astype("Int64").map(T_LABELS)
    validation = {
        "source": str(path),
        "n_rows": int(len(df)),
        "skipped_non_data_lines_after_data_start": int(bad_lines),
        "Q_unique_values": sorted([int(x) for x in df["Q"].dropna().astype(int).unique().tolist()]),
        "T_range": [float(df["T"].min()), float(df["T"].max())],
        "Inc_range": [float(df["Inc"].min()), float(df["Inc"].max())],
        "Vflat_positive_count": int((df["Vflat"] > 0).sum()),
    }
    return df.reset_index(drop=True), validation

@dataclass
class GalaxyCurve:
    name: str
    key: str
    r_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bul: np.ndarray

@dataclass
class GalaxySystem:
    curve: GalaxyCurve
    T: float
    T_label: str
    Vflat: float
    SBeff: float
    Q: float
    xV: float
    xS: float
    xT: float
    zV: float = 0.0
    zS: float = 0.0
    zT: float = 0.0

def _guess_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    def find_any(candidates: Iterable[str]) -> str | None:
        for cand in candidates:
            for low, orig in lower.items():
                if cand in low:
                    return orig
        return None
    mapping = {
        "r": find_any(["rad", "radius", "r_kpc", "r"]),
        "vobs": find_any(["vobs", "v_obs", "vrot"]),
        "verr": find_any(["e_vobs", "v_err", "err", "sigma", "ev"]),
        "vgas": find_any(["vgas", "gas"]),
        "vdisk": find_any(["vdisk", "disk"]),
        "vbul": find_any(["vbul", "bul", "bulge"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Could not infer columns: {missing}; got {cols}")
    return mapping

def load_sparc_file(path: Path) -> GalaxyCurve:
    try:
        df = pd.read_csv(path, comment="#", sep=r"\s+", engine="python")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}")
    if df.empty:
        raise ValueError(f"No data in {path}")
    try:
        mapping = _guess_columns(df)
        arr = lambda key: pd.to_numeric(df[mapping[key]], errors="coerce").to_numpy(dtype=float)
    except Exception:
        raw = pd.read_csv(path, comment="#", sep=r"\s+", engine="python", header=None)
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if raw.shape[1] < 6:
            raise ValueError(f"Could not recover 6 numeric columns from {path}")
        raw = raw.iloc[:, :6].copy()
        raw.columns = ["r", "vobs", "verr", "vgas", "vdisk", "vbul"]
        arr = lambda key: raw[key].to_numpy(dtype=float)
    g = GalaxyCurve(
        name=path.stem,
        key=normalize_galaxy_name(path.stem),
        r_kpc=arr("r"),
        v_obs=arr("vobs"),
        v_err=arr("verr"),
        v_gas=arr("vgas"),
        v_disk=arr("vdisk"),
        v_bul=arr("vbul"),
    )
    mask = np.isfinite(g.r_kpc) & np.isfinite(g.v_obs) & np.isfinite(g.v_err)
    mask &= np.isfinite(g.v_gas) & np.isfinite(g.v_disk) & np.isfinite(g.v_bul)
    mask &= (g.r_kpc > 0) & (g.v_err > 0)
    parsed = GalaxyCurve(
        name=g.name, key=g.key,
        r_kpc=g.r_kpc[mask], v_obs=g.v_obs[mask], v_err=g.v_err[mask],
        v_gas=g.v_gas[mask], v_disk=g.v_disk[mask], v_bul=g.v_bul[mask],
    )
    if parsed.r_kpc.size == 0:
        raise ValueError(f"No usable rows in {path}")
    return parsed

def load_sparc_directory(directory: Path, pattern: str = "*") -> list[GalaxyCurve]:
    files = sorted([p for p in directory.glob(pattern) if p.is_file()])
    if not files and pattern == "*":
        files = sorted([p for p in directory.rglob("*") if p.is_file()])
    curves = []
    for p in files:
        if p.suffix.lower() not in {".dat", ".txt", ".mrt", ""}:
            continue
        try:
            curves.append(load_sparc_file(p))
        except Exception:
            continue
    if not curves:
        raise FileNotFoundError(f"No parsable SPARC mass-model files found in {directory}")
    return curves

def baryonic_velocity_squared(curve: GalaxyCurve, y_disk: float, y_bul: float) -> np.ndarray:
    vgas2 = np.sign(curve.v_gas) * curve.v_gas**2
    vdisk2 = y_disk * curve.v_disk**2
    vbul2 = y_bul * curve.v_bul**2
    return np.clip(vgas2 + vdisk2 + vbul2, 0.0, None)

def smooth_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return y
    window = min(9, n if n % 2 == 1 else n - 1)
    if window < 5:
        return y
    try:
        return savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    except Exception:
        return y

def local_shape_slope(r_kpc: np.ndarray, g_bar: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(r_kpc, 1e-30, None))
    y = np.log(np.clip(g_bar, 1e-300, None))
    y = smooth_series(y)
    slope = np.gradient(y, x)
    slope = smooth_series(slope)
    return np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)

def residual_loss(resid: np.ndarray, mode: str = "chi2", huber_delta: float = 3.0) -> float:
    r = np.asarray(resid, dtype=float)
    if mode == "chi2":
        return float(np.sum(r**2))
    if mode == "huber":
        a = np.abs(r)
        quad = a <= huber_delta
        out = np.empty_like(a)
        out[quad] = 0.5 * a[quad] ** 2
        out[~quad] = huber_delta * (a[~quad] - 0.5 * huber_delta)
        return float(np.sum(out))
    raise ValueError(f"Unknown loss mode: {mode}")

def unpack_dynamic_params(params: np.ndarray | list[float], modulate: str) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    vals = np.asarray(params, dtype=float).tolist()
    if modulate == "hybrid":
        if len(vals) < 11:
            raise ValueError(f"Hybrid mode expects at least 11 dynamic parameters, got {len(vals)}")
        A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS, lambda_mix = vals[:11]
    else:
        if len(vals) < 10:
            raise ValueError(f"{modulate} mode expects at least 10 dynamic parameters, got {len(vals)}")
        A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS = vals[:10]
        lambda_mix = 0.0
    return (
        float(A), float(log10_g1), float(gamma), float(Bs),
        float(betaV), float(betaT), float(kappaS),
        float(s0S), float(etaSamp), float(alphaS), float(lambda_mix)
    )



def bound_hits(pars: dict, modulate: str, beta_bound: float, lambda_bound: float, kappa_bound: float, eta_s_bound: float, alpha_s_bound: float) -> dict:
    hits = {}
    tol = 1e-6
    checks = {
        "A": (0.0, 10.0),
        "log10_g1": (-13.0, -8.0),
        "gamma": (0.1, 3.0),
        "Bs": (-1.5, 1.5),
        "Ydisk": (0.1, 1.5),
        "Ybul": (0.0, 1.5),
        "betaV": (-beta_bound, beta_bound),
        "betaT": (-beta_bound, beta_bound),
        "kappaS": (0.1, kappa_bound),
        "s0S": (-2.5, 2.5),
        "etaSamp": (-eta_s_bound, eta_s_bound),
        "alphaS": (0.25, alpha_s_bound),
    }
    if modulate == "hybrid":
        checks["lambda_mix"] = (-lambda_bound, lambda_bound)
    for name, (lo, hi) in checks.items():
        val = float(pars[name])
        hits[name] = {
            "value": val,
            "lower": lo,
            "upper": hi,
            "at_lower": abs(val - lo) <= tol,
            "at_upper": abs(val - hi) <= tol,
            "fraction_of_upper_abs": abs(val) / max(abs(hi), 1e-12),
        }
    return hits

def v6_extra_acceleration(
    r_kpc: np.ndarray,
    g_bar_si: np.ndarray,
    slope: np.ndarray,
    system: GalaxySystem,
    params: np.ndarray,
    a0: float,
    modulate: str = "amplitude",
    eta_max: float = 3.0,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS, lambda_mix = unpack_dynamic_params(params, modulate=modulate)
    uS = math.tanh(kappaS * (system.zS - s0S))
    eta_vt_linear = betaV * system.zV + betaT * system.zT
    eta_vt = eta_max * math.tanh(eta_vt_linear)
    eta_s = etaSamp * (2.0 / math.pi) * math.atan(alphaS * uS)
    eta_total = eta_vt + eta_s
    if modulate == "amplitude":
        eta_amp = eta_total
        A_eff = A * math.exp(eta_amp)
        g1_eff = 10.0 ** log10_g1
        eta_g1 = 0.0
    elif modulate == "g1":
        A_eff = A
        eta_amp = 0.0
        eta_g1 = eta_total
        g1_eff = (10.0 ** log10_g1) * math.exp(eta_g1)
    elif modulate == "hybrid":
        eta_amp = eta_vt
        A_eff = A * math.exp(eta_amp)
        eta_g1 = eta_vt + lambda_mix * eta_s
        g1_eff = (10.0 ** log10_g1) * math.exp(eta_g1)
    else:
        raise ValueError(f"Unknown modulate mode: {modulate}")
    lead = A_eff * np.sqrt(np.clip(a0 * g_bar_si, 0.0, None))
    transition = 1.0 / (1.0 + np.power(np.clip(g_bar_si / g1_eff, 1e-30, None), gamma))
    mod = np.exp(np.clip(Bs * slope, -4.0, 4.0))
    return lead * transition * mod, eta_amp, eta_g1, eta_vt_linear, uS, eta_vt, eta_s

def predicted_velocity(system: GalaxySystem, params: np.ndarray, y_disk: float, y_bul: float, a0: float, modulate: str, eta_max: float):
    c = system.curve
    vb2 = baryonic_velocity_squared(c, y_disk=y_disk, y_bul=y_bul)
    g_bar_si = (vb2 * 1e6) / (c.r_kpc * KPC_M)
    slope = local_shape_slope(c.r_kpc, g_bar_si)
    g_cos_si, eta_amp, eta_g1, eta_linear, uS, eta_vt, eta_s = v6_extra_acceleration(c.r_kpc, g_bar_si, slope, system, params, a0=a0, modulate=modulate, eta_max=eta_max)
    vcos2 = (g_cos_si * c.r_kpc * KPC_M) / 1e6
    vtot2 = np.clip(vb2 + vcos2, 0.0, None)
    return np.sqrt(vtot2), g_bar_si, g_cos_si, slope, eta_amp, eta_g1, eta_linear, uS, eta_vt, eta_s

def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    iqr = float(max(q75 - q25, 1e-6))
    return med, iqr


def build_systems(curves: list[GalaxyCurve], meta: pd.DataFrame, max_q: int = 3) -> tuple[list[GalaxySystem], dict[str, float]]:
    meta_index = meta.set_index("galaxy_key", drop=False)
    systems: list[GalaxySystem] = []
    for c in curves:
        if c.key not in meta_index.index:
            continue
        row = meta_index.loc[c.key]
        q = float(row["Q"]) if pd.notna(row["Q"]) else np.nan
        if np.isfinite(q) and q > max_q:
            continue
        Vflat = float(row["Vflat"]) if pd.notna(row["Vflat"]) else 0.0
        SBeff = float(row["SBeff"]) if pd.notna(row["SBeff"]) else 100.0
        T = float(row["T"]) if pd.notna(row["T"]) else 7.0
        xV = 0.0 if not np.isfinite(Vflat) or Vflat <= 0 else math.log(Vflat / 100.0)
        xS = 0.0 if not np.isfinite(SBeff) or SBeff <= 0 else math.log(SBeff / 100.0)
        xT = (7.0 - T) / 3.0
        systems.append(GalaxySystem(
            curve=c, T=T, T_label=str(row.get("T_label", "")),
            Vflat=Vflat, SBeff=SBeff, Q=q, xV=xV, xS=xS, xT=xT
        ))
    medV, iqrV = _robust_center_scale(np.array([s.xV for s in systems], dtype=float))
    medS, iqrS = _robust_center_scale(np.array([s.xS for s in systems], dtype=float))
    medT, iqrT = _robust_center_scale(np.array([s.xT for s in systems], dtype=float))
    for s in systems:
        s.zV = (s.xV - medV) / iqrV
        s.zS = (s.xS - medS) / iqrS
        s.zT = (s.xT - medT) / iqrT
    scaling = {
        "xV_median": medV, "xV_iqr": iqrV,
        "xS_median": medS, "xS_iqr": iqrS,
        "xT_median": medT, "xT_iqr": iqrT,
    }
    return systems, scaling

def objective_factory(systems: list[GalaxySystem], a0: float, loss_mode: str, modulate: str,
                      reg_beta: float, reg_bs: float, reg_kappa: float, reg_lambda: float, reg_eta_s: float, reg_alpha_s: float,
                      lambda_bound: float, eta_max: float, kappa_bound: float, eta_s_bound: float, alpha_s_bound: float):
    def objective(theta: np.ndarray) -> float:
        if modulate == "hybrid":
            A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS, lambda_mix, y_disk, y_bul = theta
        else:
            A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS, y_disk, y_bul = theta
            lambda_mix = 0.0
        if not (0.0 < A < 20.0 and -13.5 < log10_g1 < -7.0 and 0.05 < gamma < 4.0):
            return 1e100
        if not (0.0 < y_disk < 2.0 and 0.0 <= y_bul < 2.0):
            return 1e100
        if not (0.1 < kappaS < kappa_bound):
            return 1e100
        if not (-2.5 < s0S < 2.5):
            return 1e100
        if not (-eta_s_bound < etaSamp < eta_s_bound):
            return 1e100
        if not (0.25 < alphaS < alpha_s_bound):
            return 1e100
        if modulate == "hybrid" and not (-lambda_bound < lambda_mix < lambda_bound):
            return 1e100
        dyn_list = [A, log10_g1, gamma, Bs, betaV, betaT, kappaS, s0S, etaSamp, alphaS]
        if modulate == "hybrid":
            dyn_list.append(lambda_mix)
        dyn = np.array(dyn_list, dtype=float)
        total = 0.0
        npts = 0
        for sys in systems:
            if len(sys.curve.r_kpc) < 5:
                continue
            pred, _, _, _, _, _, _, _, _, _ = predicted_velocity(sys, dyn, y_disk=y_disk, y_bul=y_bul, a0=a0, modulate=modulate, eta_max=eta_max)
            resid = (sys.curve.v_obs - pred) / sys.curve.v_err
            total += residual_loss(resid, mode=loss_mode)
            npts += resid.size
        if npts == 0:
            return 1e100
        reg = reg_bs * (Bs**2) + reg_beta * (betaV**2 + betaT**2) + reg_kappa * ((kappaS - 1.0)**2 + 0.5*(s0S**2)) + reg_eta_s * (etaSamp**2) + reg_alpha_s * ((alphaS - 1.0)**2)
        if modulate == "hybrid":
            reg += reg_lambda * (lambda_mix**2)
        return total + reg
    return objective

def fit_global(systems: list[GalaxySystem], a0: float, loss_mode: str, modulate: str,
               seed: int = 1234, restarts: int = 3, maxiter: int = 160,
               reg_beta: float = 0.25, reg_bs: float = 4.0, reg_kappa: float = 0.5, reg_lambda: float = 0.25,
               reg_eta_s: float = 0.25, reg_alpha_s: float = 0.15, beta_bound_amp: float = 1.5, beta_bound_g1: float = 4.0,
               lambda_bound: float = 4.0, eta_max: float = 3.0, kappa_bound: float = 6.0, eta_s_bound: float = 2.5, alpha_s_bound: float = 8.0) -> dict:
    beta_bound = beta_bound_g1 if modulate in {"g1", "hybrid"} else beta_bound_amp
    if modulate == "hybrid":
        bounds = [
            (0.0, 10.0), (-13.0, -8.0), (0.1, 3.0), (-1.5, 1.5),
            (-beta_bound, beta_bound), (-beta_bound, beta_bound),
            (0.1, kappa_bound), (-2.5, 2.5), (-eta_s_bound, eta_s_bound), (0.25, alpha_s_bound), (-lambda_bound, lambda_bound), (0.1, 1.5), (0.0, 1.5),
        ]
        names = ["A", "log10_g1", "gamma", "Bs", "betaV", "betaT", "kappaS", "s0S", "etaSamp", "alphaS", "lambda_mix", "Ydisk", "Ybul"]
    else:
        bounds = [
            (0.0, 10.0), (-13.0, -8.0), (0.1, 3.0), (-1.5, 1.5),
            (-beta_bound, beta_bound), (-beta_bound, beta_bound),
            (0.1, kappa_bound), (-2.5, 2.5), (-eta_s_bound, eta_s_bound), (0.25, alpha_s_bound), (0.1, 1.5), (0.0, 1.5),
        ]
        names = ["A", "log10_g1", "gamma", "Bs", "betaV", "betaT", "kappaS", "s0S", "etaSamp", "alphaS", "Ydisk", "Ybul"]
    objective = objective_factory(systems, a0=a0, loss_mode=loss_mode, modulate=modulate,
                                 reg_beta=reg_beta, reg_bs=reg_bs, reg_kappa=reg_kappa, reg_lambda=reg_lambda, reg_eta_s=reg_eta_s, reg_alpha_s=reg_alpha_s,
                                 lambda_bound=lambda_bound, eta_max=eta_max, kappa_bound=kappa_bound, eta_s_bound=eta_s_bound, alpha_s_bound=alpha_s_bound)
    best = None
    history = []
    for i in range(restarts):
        local_seed = seed + i
        result = differential_evolution(
            objective, bounds=bounds, polish=True, seed=local_seed,
            updating="deferred", workers=1, maxiter=maxiter
        )
        history.append({"seed": local_seed, "objective": float(result.fun), "success": bool(result.success)})
        if best is None or result.fun < best.fun:
            best = result
    out = {name: float(val) for name, val in zip(names, best.x)}
    out.update({
        "model": "global_mod", "modulate": modulate, "loss": loss_mode,
        "objective": float(best.fun), "success": bool(best.success),
        "message": str(best.message), "restarts": int(restarts), "seed": int(seed),
        "fit_history": history,
        "reg_beta": float(reg_beta), "reg_bs": float(reg_bs), "reg_kappa": float(reg_kappa), "reg_lambda": float(reg_lambda), "reg_eta_s": float(reg_eta_s), "reg_alpha_s": float(reg_alpha_s),
        "beta_bound_amp": float(beta_bound_amp), "beta_bound_g1": float(beta_bound_g1),
        "lambda_bound": float(lambda_bound), "kappa_bound": float(kappa_bound), "eta_s_bound": float(eta_s_bound), "alpha_s_bound": float(alpha_s_bound),
        "eta_max": float(eta_max),
    })
    out["bound_hits"] = bound_hits(out, modulate=modulate, beta_bound=beta_bound, lambda_bound=lambda_bound, kappa_bound=kappa_bound, eta_s_bound=eta_s_bound, alpha_s_bound=alpha_s_bound)
    return out

def per_galaxy_summary(systems: list[GalaxySystem], pars: dict, a0: float) -> pd.DataFrame:
    dyn_list = [pars["A"], pars["log10_g1"], pars["gamma"], pars["Bs"], pars["betaV"], pars["betaT"], pars["kappaS"], pars["s0S"], pars["etaSamp"], pars["alphaS"]]
    if pars.get("modulate") == "hybrid":
        dyn_list.append(pars.get("lambda_mix", 0.5))
    dyn = np.array(dyn_list, dtype=float)
    eta_max = float(pars.get("eta_max", 3.0))
    rows = []
    for sys in systems:
        if len(sys.curve.r_kpc) < 5:
            continue
        pred, gbar_si, gcos_si, slope, eta_amp, eta_g1, eta_linear, uS, eta_vt, eta_s = predicted_velocity(
            sys, params=dyn, y_disk=pars["Ydisk"], y_bul=pars["Ybul"], a0=a0, modulate=pars["modulate"], eta_max=eta_max
        )
        resid = (sys.curve.v_obs - pred) / sys.curve.v_err
        chi2 = float(np.sum(resid**2))
        dof_like = float(chi2 / max(len(sys.curve.r_kpc) - 1, 1))
        if pars["modulate"] == "amplitude":
            A_eff = pars["A"] * math.exp(eta_amp)
            g1_eff = (10.0 ** pars["log10_g1"])
        elif pars["modulate"] == "g1":
            A_eff = pars["A"]
            g1_eff = (10.0 ** pars["log10_g1"]) * math.exp(eta_g1)
        elif pars["modulate"] == "hybrid":
            A_eff = pars["A"] * math.exp(eta_amp)
            g1_eff = (10.0 ** pars["log10_g1"]) * math.exp(eta_g1)
        else:
            A_eff = pars["A"]
            g1_eff = (10.0 ** pars["log10_g1"])
        rows.append({
            "galaxy": sys.curve.name, "npts": int(len(sys.curve.r_kpc)),
            "chi2": chi2, "chi2_dof_like": dof_like,
            "median_abs_pull": float(np.median(np.abs(resid))),
            "p90_abs_pull": float(np.quantile(np.abs(resid), 0.90)),
            "mean_pull": float(np.mean(resid)),
            "vobs_mean": float(np.mean(sys.curve.v_obs)),
            "vpred_mean": float(np.mean(pred)),
            "vpred_vobs_ratio": float(np.mean(pred) / np.mean(sys.curve.v_obs)),
            "gbar_mean_si": float(np.mean(gbar_si)),
            "gcos_mean_si": float(np.mean(gcos_si)),
            "slope_mean": float(np.mean(slope)),
            "eta_linear": float(eta_linear),
            "eta_amp": float(eta_amp),
            "eta_g1": float(eta_g1),
            "eta_vt": float(eta_vt),
            "eta_s": float(eta_s),
            "A_eff": float(A_eff), "g1_eff": float(g1_eff),
            "T": float(sys.T), "T_label": sys.T_label, "Vflat": float(sys.Vflat),
            "SBeff": float(sys.SBeff), "Q": float(sys.Q),
            "xV": float(sys.xV), "xS": float(sys.xS), "xT": float(sys.xT),
            "zV": float(sys.zV), "zS": float(sys.zS), "zT": float(sys.zT), "uS": float(uS),
        })
    return pd.DataFrame(rows).sort_values("chi2_dof_like").reset_index(drop=True)

def build_report(summary: pd.DataFrame, pars: dict, scaling: dict[str, float] | None = None) -> dict:
    out = {
        "overall_metrics": {
            "n_galaxies": int(len(summary)),
            "n_points": int(summary["npts"].sum()),
            "chi2_total": float(summary["chi2"].sum()),
            "chi2_dof_like_median": float(summary["chi2_dof_like"].median()),
            "chi2_dof_like_mean": float(summary["chi2_dof_like"].mean()),
            "bad_fit_fraction_gt20": float((summary["chi2_dof_like"] > 20).mean()),
            "very_bad_fit_fraction_gt100": float((summary["chi2_dof_like"] > 100).mean()),
            "aic_like": float(summary["chi2"].sum() + 2 * len(pars)),
            "bic_like": float(summary["chi2"].sum() + math.log(max(int(summary["npts"].sum()),1)) * len(pars)),
        },
        "fit": pars,
    }
    if scaling is not None:
        out["feature_scaling"] = {k: float(v) for k, v in scaling.items()}
    if len(summary) > 0:
        out["saturation_stats"] = {
            "eta_linear_abs_gt_2_fraction": float((summary["eta_linear"].abs() > 2.0).mean()),
            "eta_amp_abs_gt_0.95_eta_max_fraction": float((summary["eta_amp"].abs() > 0.95 * float(pars.get("eta_max", 3.0))).mean()),
            "eta_g1_abs_gt_0.95_eta_max_fraction": float((summary["eta_g1"].abs() > 0.95 * float(pars.get("eta_max", 3.0))).mean()),
            "eta_amp_range": [float(summary["eta_amp"].min()), float(summary["eta_amp"].max())],
            "eta_g1_range": [float(summary["eta_g1"].min()), float(summary["eta_g1"].max())],
            "eta_vt_range": [float(summary["eta_vt"].min()), float(summary["eta_vt"].max())],
            "eta_s_range": [float(summary["eta_s"].min()), float(summary["eta_s"].max())],
            "eta_s_abs_gt_0.9_fraction": float((summary["eta_s"].abs() > 0.9 * max(abs(float(pars.get("etaSamp", 1.0))), 1e-12)).mean()),
            "eta_s_near_linear_fraction": float((summary["uS"].abs() < 0.5).mean()),
        }
    out["bucket_stats"] = {}
    for feat in ["Vflat", "SBeff", "T", "Q"]:
        s = summary[[feat, "chi2_dof_like"]].dropna()
        if len(s) < 10:
            continue
        if feat == "Q":
            grp = s.groupby(feat)["chi2_dof_like"].agg(["count", "median", "mean"]).reset_index()
        else:
            med = float(s[feat].median())
            grp = pd.DataFrame({
                feat: [f"<=median({med:.3g})", f">median({med:.3g})"],
                "count": [(s[feat] <= med).sum(), (s[feat] > med).sum()],
                "median": [float(s.loc[s[feat] <= med, "chi2_dof_like"].median()),
                           float(s.loc[s[feat] > med, "chi2_dof_like"].median())],
                "mean": [float(s.loc[s[feat] <= med, "chi2_dof_like"].mean()),
                         float(s.loc[s[feat] > med, "chi2_dof_like"].mean())],
            })
        out["bucket_stats"][feat] = grp.to_dict(orient="records")
    out["worst_15"] = summary.sort_values("chi2_dof_like", ascending=False).head(15).to_dict(orient="records")
    out["best_15"] = summary.sort_values("chi2_dof_like", ascending=True).head(15).to_dict(orient="records")
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="COS-DM SPARC fitter with a tempered arctan SBeff channel. This interpolates between the too-aggressive v6.7 bounded-tanh channel and the too-gentle v6.8 direct-bounded channel by using etaS = etaSamp * (2/pi) * atan(alphaS * tanh(kappaS(zS-s0S))).")
    ap.add_argument("sparc_dir", type=Path, help="Directory with extracted SPARC mass-model files")
    ap.add_argument("table1_mrt", type=Path, help="SPARC Galaxy Sample Table1.mrt")
    ap.add_argument("--pattern", default="**/*.dat", help='File pattern inside sparc_dir, default="**/*.dat"')
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--loss", choices=["chi2", "huber"], default="chi2")
    ap.add_argument("--modulate", choices=["amplitude", "g1", "hybrid"], default="amplitude")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-q", type=int, default=3, dest="max_q", help="Keep only galaxies with Q <= max_q")

    ap.add_argument("--reg-beta", type=float, default=0.25, dest="reg_beta",
                    help="L2 regularization strength for betaV and betaT")
    ap.add_argument("--reg-bs", type=float, default=4.0, dest="reg_bs",
                    help="L2 regularization strength for Bs")
    ap.add_argument("--reg-kappa", type=float, default=0.5, dest="reg_kappa",
                    help="L2 regularization strength for kappaS around 1.0")
    ap.add_argument("--reg-lambda", type=float, default=0.25, dest="reg_lambda",
                    help="L2 regularization strength for lambda_mix in hybrid mode")
    ap.add_argument("--reg-eta-s", type=float, default=0.25, dest="reg_eta_s",
                    help="L2 regularization strength for etaSamp around 0.0")
    ap.add_argument("--reg-alpha-s", type=float, default=0.15, dest="reg_alpha_s",
                    help="L2 regularization strength for alphaS around 1.0")
    ap.add_argument("--beta-bound-amp", type=float, default=1.5, dest="beta_bound_amp",
                    help="Absolute bound for beta parameters in amplitude mode")
    ap.add_argument("--beta-bound-g1", type=float, default=4.0, dest="beta_bound_g1",
                    help="Absolute bound for beta parameters in g1/hybrid modes")
    ap.add_argument("--lambda-bound", type=float, default=4.0, dest="lambda_bound",
                    help="Absolute bound for lambda_mix in hybrid mode")
    ap.add_argument("--kappa-bound", type=float, default=6.0, dest="kappa_bound",
                    help="Upper bound for nonlinear SBeff shape parameter kappaS")
    ap.add_argument("--eta-max", type=float, default=3.0, dest="eta_max",
                    help="Smooth saturation scale for the VT tanh global modulator")
    ap.add_argument("--eta-s-bound", type=float, default=2.5, dest="eta_s_bound",
                    help="Absolute bound for the dedicated bounded SBeff contribution etaSamp")
    ap.add_argument("--alpha-s-bound", type=float, default=8.0, dest="alpha_s_bound",
                    help="Upper bound for the arctan SBeff gain alphaS")

    ap.add_argument("--out-prefix", default="cos_dm_v6")
    ap.add_argument("--dry-run", action="store_true", help="Only parse/join data and print counts; do not fit.")
    args = ap.parse_args()

    meta, validation = parse_table1_mrt(args.table1_mrt)
    curves = load_sparc_directory(args.sparc_dir, pattern=args.pattern)
    systems, scaling = build_systems(curves, meta, max_q=args.max_q)

    print(f"Loaded {len(curves)} curve files")
    print(f"Joined {len(systems)} systems with Table1 metadata")
    print(f"Feature scaling: {scaling}")
    print(f"Table1 validation: {validation}")

    if args.dry_run:
        return 0

    pars = fit_global(systems, a0=args.a0, loss_mode=args.loss, modulate=args.modulate,
                      seed=args.seed, restarts=args.restarts, maxiter=args.maxiter,
                      reg_beta=args.reg_beta, reg_bs=args.reg_bs, reg_kappa=args.reg_kappa, reg_lambda=args.reg_lambda, reg_eta_s=args.reg_eta_s, reg_alpha_s=args.reg_alpha_s,
                      beta_bound_amp=args.beta_bound_amp, beta_bound_g1=args.beta_bound_g1,
                      lambda_bound=args.lambda_bound, eta_max=args.eta_max, kappa_bound=args.kappa_bound, eta_s_bound=args.eta_s_bound, alpha_s_bound=args.alpha_s_bound)
    summary = per_galaxy_summary(systems, pars, a0=args.a0)
    report = build_report(summary, pars, scaling=scaling)

    summary_path = Path(f"{args.out_prefix}_summary.csv")
    fit_path = Path(f"{args.out_prefix}_fit.json")
    report_path = Path(f"{args.out_prefix}_report.json")
    summary.to_csv(summary_path, index=False)
    with open(fit_path, "w", encoding="utf-8") as f:
        json.dump(pars, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nBest-fit parameters")
    for k, v in pars.items():
        if k == "fit_history":
            continue
        print(f"{k:>12}: {v}")

    print("\nOverall metrics")
    for k, v in report["overall_metrics"].items():
        print(f"{k:>24}: {v}")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved fit parameters to {fit_path}")
    print(f"Saved report to {report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
