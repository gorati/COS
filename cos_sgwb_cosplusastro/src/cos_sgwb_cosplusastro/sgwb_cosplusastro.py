#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COS + asztrofizikai SGWB illesztés pygwb + bilby segítségével.

Fő lépések:
  1) parameters.ini generálása pygwb_pipe-hoz
  2) pygwb_pipe futtatása GWOSC adatra (H1–L1), akár több ablakra
  3) point_estimate_sigma_*.npz-ek beolvasása, több ablakból kombinálva
  4) Bayes-i illesztés egy analitikus COS-modellre + asztrofizikai power-law háttérre

Modell:
  Ω_GW^COS(f) =
    Ω0 (f / f0)^α [ 1 + A_disc cos( ω ln(f / f_star) + φ ) ]

  Ω_astro(f) =
    Ω_astro0 (f / f0)^α_astro

  Ω_model(f) = Ω_GW^COS(f) + Ω_astro(f)

A kód:
  - tudományosan standard LVK/pygwb workflow-t követ,
  - robusztus (hibaellenőrzések, multi-window mód),
  - tisztán implementálja a COS + asztrofizikai modellt,
  - több modell/prior variáns futtatását is támogatja (COS+astro, astro-only, stb.).
"""

import os
import sys
import textwrap
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, List
from gwpy.timeseries import TimeSeries

import numpy as np
import warnings

from astropy.utils.data import conf as astropy_data_conf

os.environ["GWPY_CACHE"] = "1"

# Suppress NumPy complex-cast warnings triggered inside some sampler/backend code.
# NumPy >= 2.0 no longer exposes ComplexWarning as np.ComplexWarning; it lives
# under numpy.exceptions. Keep a fallback for older NumPy versions.
try:
    from numpy.exceptions import ComplexWarning as _NumpyComplexWarning
except Exception:  # NumPy < 2.0 compatibility
    _NumpyComplexWarning = getattr(np, "ComplexWarning", RuntimeWarning)

warnings.filterwarnings("ignore", category=_NumpyComplexWarning)

astropy_data_conf.remote_timeout = 120.0

try:
    import bilby
except ImportError:
    bilby = None


# ----------------------------------------------------------------------
# 1. KONFIGURÁCIÓ
# ----------------------------------------------------------------------

@dataclass
class Config:
    # --- Egyablakos beállítások (visszafelé kompatibilitás) ---
    t0: int = 1247644138
    tf: int = 1247644138 + 3600   # 1 óra

    # --- Multi-window mód ---
    use_multi_window: bool = False
    multi_t0: Optional[int] = None          # ha None, akkor t0
    multi_tf: Optional[int] = None          # ha None, akkor tf
    multi_window_duration: int = 9000       # 2.5 óra = 9000 s

    # Detektorok
    ifos: Tuple[str, str] = ("H1", "L1")

    # GWOSC csatornák – itt feltételezzük 16 kHz-es strain-t.
    channel_pattern: str = "{ifo}:GWOSC-16KHZ_R1_STRAIN"

    # Mintavétel és szegmentálás (pygwb tipikus beállításai)
    input_sample_rate: int = 16384
    new_sample_rate: int = 4096
    cutoff_frequency: float = 11.0
    segment_duration: int = 192
    number_cropped_seconds: int = 2
    frequency_resolution: float = 1.0 / 32.0  # 1/32 Hz

    # Gating beállítások
    gate_data: bool = True
    gate_tzero: float = 1.0
    gate_tpad: float = 0.5
    gate_threshold: float = 50.0
    cluster_window: float = 0.5
    gate_whiten: bool = True

    # Frekvenciasáv és postprocessing
    flow: float = 20.0
    fhigh: float = 1726.0
    fref: float = 25.0

    # A pygwb-nek átadott "alpha" csak a standard power-law benchmarkhoz kell
    alpha_ref: float = 0.0  # itt lapos referencia-spektrum (Ω ~ f^0)

    # Notch lista – ha van lokális notch listád, itt add meg.
    notch_list_path: str = ""

    # Kimeneti mappák
    workdir: Path = Path("./cos_sgwb_run").resolve()
    pygwb_output_dir: Path = Path("./cos_sgwb_run/pygwb_output").resolve()
    pe_output_dir: Path = Path("./cos_sgwb_run/pe_output").resolve()

    # COS-modell referencia frekvenciák
    f0: float = 25.0       # referencia frekvencia (Hz) az Ω0 normalizációhoz
    f_star: float = 25.0   # referencia frekvencia az oszcilláció argumentumához

    # Bilby futtatási beállítások
    bilby_sampler: str = "dynesty"  # nested sampling
    bilby_nlive: int = 500          # csak dynesty esetén használjuk

    # --- Modell / prior variánsok ---
    # model_variant:
    #   "cos_plus_astro" : COS + asztro háttér
    #   "astro_only"     : csak asztrofizikai háttér
    #   "cos_only"       : csak COS komponens
    model_variant: str = "cos_plus_astro"

    # prior_variant:
    #   "wide" : 1e-12–1e-8 (exploratív)
    #   "lvk"  : 1e-13–1e-9 (LVK-limithez igazított példa)
    prior_variant: str = "wide"

    # --- Prior tartományok (alapérték: "wide") ---
    prior_Omega0_min: float = 1e-12
    prior_Omega0_max: float = 1e-8

    prior_alpha_min: float = -2.0
    prior_alpha_max: float = 2.0

    prior_A_disc_min: float = 0.0
    prior_A_disc_max: float = 0.3  # diszkrét moduláció, de nem extrém nagy

    prior_omega_min: float = 5.0
    prior_omega_max: float = 25.0  # mennyi oszcilláció ln(f)-enként, COS-tartományban

    prior_phi_min: float = 0.0
    prior_phi_max: float = 2.0 * np.pi

    prior_Omega_astro0_min: float = 1e-12
    prior_Omega_astro0_max: float = 1e-8

    prior_alpha_astro_min: float = -2.0
    prior_alpha_astro_max: float = 2.0

    # Fix paraméterek (pl. α = 0, α_astro = 2/3)
    fixed_parameters: dict = None


# ----------------------------------------------------------------------
# 2. SEGÉDFÜGGVÉNYEK: LOG, ELLENŐRZÉS
# ----------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[COS-SGWB] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _which(cmd: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in paths:
        candidate = Path(p) / cmd
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def check_dependencies(config: Config) -> None:
    missing = []

    try:
        import pygwb  # noqa: F401
    except ImportError:
        missing.append("pygwb")

    try:
        import gwpy  # noqa: F401
    except ImportError:
        missing.append("gwpy")

    if bilby is None:
        missing.append("bilby")

    if missing:
        raise RuntimeError(
            "Hiányzó Python csomagok: "
            + ", ".join(missing)
            + "\nTelepítsd például (conda/micromamba):\n"
            + "   micromamba install -c conda-forge pygwb gwpy bilby emcee\n"
        )

    if not _which("pygwb_pipe"):
        raise RuntimeError(
            "Nem találom a 'pygwb_pipe' parancsot a PATH-ban.\n"
            "Győződj meg róla, hogy az IGWN/pygwb környezet aktív."
        )


# ----------------------------------------------------------------------
# 3. PYGWB PARAMÉTERFÁJL (parameters.ini)
# ----------------------------------------------------------------------

def build_channel_string(config: Config) -> str:
    return ",".join(config.channel_pattern.format(ifo=ifo) for ifo in config.ifos)


def write_pygwb_parameters_file(config: Config) -> Path:
    ensure_dir(config.workdir)
    param_path = config.workdir / "parameters.ini"

    ini = textwrap.dedent(f"""
    [data_specs]
    interferometer_list: ["{config.ifos[0]}", "{config.ifos[1]}"]
    t0: {config.t0}
    tf: {config.tf}
    data_type: public
    channel: {build_channel_string(config)}
    frametype:
    time_shift: 0
    random_time_shift: False

    [preprocessing]
    new_sample_rate: {config.new_sample_rate}
    input_sample_rate: {config.input_sample_rate}
    cutoff_frequency: {config.cutoff_frequency}
    segment_duration: {config.segment_duration}
    number_cropped_seconds: {config.number_cropped_seconds}
    window_downsampling: hamming
    ftype: fir

    [gating]
    gate_data: {str(config.gate_data)}
    gate_whiten: {str(config.gate_whiten)}
    gate_tzero: {config.gate_tzero}
    gate_tpad: {config.gate_tpad}
    gate_threshold: {config.gate_threshold}
    cluster_window: {config.cluster_window}

    [window_fft_specs]
    window_fftgram: hann

    [window_fft_welch_specs]
    window_fftgram: hann

    [density_estimation]
    frequency_resolution: {config.frequency_resolution}
    N_average_segments_psd: 2
    coarse_grain_psd: False
    coarse_grain_csd: False
    overlap_factor_welch: 0.5
    overlap_factor: 0.5

    [postprocessing]
    polarization: tensor
    alpha: {config.alpha_ref}
    fref: {config.fref}
    flow: {config.flow}
    fhigh: {config.fhigh}

    [data_quality]
    notch_list_path: {config.notch_list_path}
    calibration_epsilon: 0.0
    alphas_delta_sigma_cut: [-5, 0, 3]
    delta_sigma_cut: 0.2
    return_naive_and_averaged_sigmas: False

    [output]
    save_data_type: npz

    [local_data]
    local_data_path:
    """).strip() + "\n"

    param_path.write_text(ini)
    log(f"parameters.ini létrehozva: {param_path}")
    return param_path


# ----------------------------------------------------------------------
# 4. PYGWB_PIPE FUTTATÁSA
# ----------------------------------------------------------------------

def run_pygwb_pipe(config: Config, param_file: Path) -> None:
    ensure_dir(config.pygwb_output_dir)

    cmd = [
        "pygwb_pipe",
        "--param_file", str(param_file),
        "--output_path", str(config.pygwb_output_dir),
        "--calc_coh", "False",
        "--calc_pt_est", "True",
        "--apply_dsc", "True",
        "--pickle_out", "False",
        "--wipe_ifo", "True",
    ]

    log("pygwb_pipe futtatása...")
    log(" ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            "pygwb_pipe hiba (returncode "
            f"{result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    log("pygwb_pipe lefutott.")


def guess_point_estimate_file(config: Config) -> Path:
    # A valódi fájlnév a pygwb-ből jön, ezért itt inkább keresünk:
    for p in config.pygwb_output_dir.glob("point_estimate_sigma_*.npz"):
        return p

    raise FileNotFoundError(
        f"Nem találom a 'point_estimate_sigma_*.npz' fájlt a {config.pygwb_output_dir} könyvtárban."
    )


# ----------------------------------------------------------------------
# 5. PYGWB KIMENET BEOLVASÁSA
# ----------------------------------------------------------------------

def load_pygwb_point_estimate(
    npz_path: Path,
    flow: float,
    fhigh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    log(f"pygwb kimenet beolvasása: {npz_path}")
    data = np.load(npz_path)

    freqs = data["frequencies"]
    omega_hat = data["point_estimate_spectrum"]
    sigma = data["sigma_spectrum"]

    mask = np.ones_like(freqs, dtype=bool)
    if "frequency_mask" in data:
        try:
            freq_mask = data["frequency_mask"].astype(bool)
            if freq_mask.shape == freqs.shape:
                mask &= freq_mask
        except Exception:
            pass

    mask &= (freqs >= flow) & (freqs <= fhigh)

    freqs_sel = freqs[mask]
    omega_hat_sel = omega_hat[mask]
    sigma_sel = sigma[mask]

    finite_mask = (
        np.isfinite(freqs_sel)
        & np.isfinite(omega_hat_sel)
        & np.isfinite(sigma_sel)
        & (sigma_sel > 0)
    )

    if not np.all(finite_mask):
        n_bad = np.size(finite_mask) - np.count_nonzero(finite_mask)
        log(
            f"Figyelem: {n_bad} frekvenciabin NaN/inf vagy σ<=0 miatt kihagyva "
            f"(eredeti bin szám: {len(freqs_sel)})."
        )
        freqs_sel = freqs_sel[finite_mask]
        omega_hat_sel = omega_hat_sel[finite_mask]
        sigma_sel = sigma_sel[finite_mask]

    if len(freqs_sel) == 0:
        raise RuntimeError(
            "A megadott frekvenciasávban (és tisztítás után) egyetlen használható bin sem maradt – "
            "ellenőrizd a flow/fhigh értékeket vagy a pygwb kimenetet."
        )

    log(f"{len(freqs_sel)} frekvenciabin kiválasztva [{flow}, {fhigh}] Hz tartományban (tisztítás után).")
    return freqs_sel, omega_hat_sel, sigma_sel


# ----------------------------------------------------------------------
# 6. ANALITIKUS COS + ASZTROFIZIKAI MODELL
# ----------------------------------------------------------------------

def omega_cos_analytic(
    f: np.ndarray,
    Omega0: float,
    alpha: float,
    A_disc: float,
    omega: float,
    phi: float,
    f0: float,
    f_star: float,
) -> np.ndarray:
    f = np.asarray(f)
    f_safe = np.clip(f, 1e-9, None)
    base = (f_safe / f0) ** alpha
    osc = np.cos(omega * np.log(f_safe / f_star) + phi)
    return Omega0 * base * (1.0 + A_disc * osc)


def omega_astro_powerlaw(
    f: np.ndarray,
    Omega_astro0: float,
    alpha_astro: float,
    f0: float,
) -> np.ndarray:
    f = np.asarray(f)
    f_safe = np.clip(f, 1e-9, None)
    return Omega_astro0 * (f_safe / f0) ** alpha_astro


def omega_model_total(
    f: np.ndarray,
    params: dict,
    config: Config,
) -> np.ndarray:
    """
    Teljes modell a model_variant szerint:

      - "cos_plus_astro": Ω_COS + Ω_astro
      - "astro_only":     Ω_astro
      - "cos_only":       Ω_COS
    """
    has_cos = config.model_variant in ("cos_plus_astro", "cos_only")
    has_astro = config.model_variant in ("cos_plus_astro", "astro_only")

    cos_part = 0.0
    if has_cos:
        Omega0 = params["Omega0"]
        alpha = params["alpha"]
        A_disc = params["A_disc"]
        omega = params["omega"]
        phi = params["phi"]
        cos_part = omega_cos_analytic(
            f=f,
            Omega0=Omega0,
            alpha=alpha,
            A_disc=A_disc,
            omega=omega,
            phi=phi,
            f0=config.f0,
            f_star=config.f_star,
        )

    astro_part = 0.0
    if has_astro:
        Omega_astro0 = params["Omega_astro0"]
        alpha_astro = params["alpha_astro"]
        astro_part = omega_astro_powerlaw(
            f=f,
            Omega_astro0=Omega_astro0,
            alpha_astro=alpha_astro,
            f0=config.f0,
        )

    return cos_part + astro_part


# ----------------------------------------------------------------------
# 7. BAYES-I ILLESZTÉS BILBY-VEL
# ----------------------------------------------------------------------

class CosPlusAstroLikelihood(bilby.Likelihood):
    def __init__(self, freqs, omega_hat, sigma, config: Config):
        super().__init__(parameters={
            "Omega0": None,
            "alpha": None,
            "A_disc": None,
            "omega": None,
            "phi": None,
            "Omega_astro0": None,
            "alpha_astro": None,
        })

        self.freqs = np.asarray(freqs)
        self.omega_hat = np.asarray(omega_hat)
        self.sigma = np.asarray(sigma)
        self.config = config

        if not (
            self.freqs.shape == self.omega_hat.shape == self.sigma.shape
        ):
            raise ValueError("freqs, omega_hat, sigma különböző méretűek.")

    def log_likelihood(self) -> float:
        if not (
            np.all(np.isfinite(self.freqs))
            and np.all(np.isfinite(self.omega_hat))
            and np.all(np.isfinite(self.sigma))
        ):
            return -np.inf

        full_params = dict(self.parameters)
        if self.config.fixed_parameters:
            for key, val in self.config.fixed_parameters.items():
                full_params[key] = val

        model = omega_model_total(
            f=self.freqs,
            params=full_params,
            config=self.config,
        )

        if not np.all(np.isfinite(model)):
            return -np.inf

        resid = self.omega_hat - model
        # Teljes Gaussian log-likelihood (evidence-hez is jó):
        logL = -0.5 * np.sum(
            (resid / self.sigma) ** 2 + np.log(2 * np.pi * self.sigma ** 2)
        )

        if not np.isfinite(logL):
            return -np.inf

        return float(np.real(logL))


def build_priors(config: Config) -> bilby.core.prior.PriorDict:
    priors = bilby.core.prior.PriorDict()
    fixed = config.fixed_parameters or {}

    has_cos = config.model_variant in ("cos_plus_astro", "cos_only")
    has_astro = config.model_variant in ("cos_plus_astro", "astro_only")

    # Prior-variáns: wide vs LVK-informált
    if config.prior_variant == "wide":
        om0_min, om0_max = config.prior_Omega0_min, config.prior_Omega0_max
        omastro_min, omastro_max = (
            config.prior_Omega_astro0_min,
            config.prior_Omega_astro0_max,
        )
    elif config.prior_variant == "lvk":
        # Példaként LVK-szintű korlátokra hangolt prior
        om0_min, om0_max = 1e-13, 1e-9
        omastro_min, omastro_max = 1e-13, 1e-9
    else:
        raise ValueError(f"Ismeretlen prior_variant: {config.prior_variant}")

    # COS amplitúdó
    if has_cos and "Omega0" not in fixed:
        priors["Omega0"] = bilby.core.prior.LogUniform(
            minimum=om0_min,
            maximum=om0_max,
            name=r"\Omega_0",
        )

    # COS lejtő
    if has_cos and "alpha" not in fixed:
        priors["alpha"] = bilby.core.prior.Uniform(
            minimum=config.prior_alpha_min,
            maximum=config.prior_alpha_max,
            name=r"\alpha_{\rm COS}",
        )

    # Diszkontinuus moduláció paraméterei
    if has_cos and "A_disc" not in fixed:
        priors["A_disc"] = bilby.core.prior.Uniform(
            minimum=config.prior_A_disc_min,
            maximum=config.prior_A_disc_max,
            name=r"A_{\rm disc}",
        )

    if has_cos and "omega" not in fixed:
        priors["omega"] = bilby.core.prior.Uniform(
            minimum=config.prior_omega_min,
            maximum=config.prior_omega_max,
            name=r"\omega",
        )

    if has_cos and "phi" not in fixed:
        priors["phi"] = bilby.core.prior.Uniform(
            minimum=config.prior_phi_min,
            maximum=config.prior_phi_max,
            name=r"\phi",
        )

    # Asztro amplitúdó
    if has_astro and "Omega_astro0" not in fixed:
        priors["Omega_astro0"] = bilby.core.prior.LogUniform(
            minimum=omastro_min,
            maximum=omastro_max,
            name=r"\Omega_{\rm astro,0}",
        )

    # Asztro lejtő
    if has_astro and "alpha_astro" not in fixed:
        priors["alpha_astro"] = bilby.core.prior.Uniform(
            minimum=config.prior_alpha_astro_min,
            maximum=config.prior_alpha_astro_max,
            name=r"\alpha_{\rm astro}",
        )

    return priors


def run_bilby_fit(
    config: Config,
    freqs: np.ndarray,
    omega_hat: np.ndarray,
    sigma: np.ndarray,
    label: str,
):
    if bilby is None:
        raise RuntimeError("A 'bilby' csomag nincs telepítve – nem tudok PE-t futtatni.")

    ensure_dir(config.pe_output_dir)

    likelihood = CosPlusAstroLikelihood(freqs, omega_hat, sigma, config)
    priors = build_priors(config)

    sampler_kwargs = {}
    if config.bilby_sampler.lower() == "dynesty":
        sampler_kwargs.update(
            dict(
                nlive=config.bilby_nlive,
                walks=100,
                dlogz=0.1,
            )
        )
        log(
            f"Bilby nested sampling indul (dynesty, model={config.model_variant}, "
            f"prior={config.prior_variant}, label={label})..."
        )

    elif config.bilby_sampler.lower() == "emcee":
        sampler_kwargs.update(
            dict(
                nwalkers=64,
                nsteps=8000,
            )
        )
        log(
            f"Bilby MCMC indul (emcee, model={config.model_variant}, "
            f"prior={config.prior_variant}, label={label})..."
        )
    else:
        raise ValueError(f"Ismeretlen sampler: {config.bilby_sampler}")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=config.bilby_sampler,
        outdir=str(config.pe_output_dir),
        label=label,
        resume=False,
        clean=True,
        verbose=True,
        **sampler_kwargs,
    )

    log(f"Bilby futás kész (label={label}).")
    log(f"Eredmények a kimeneti könyvtárban: {config.pe_output_dir}")
    return result


def summarize_posterior(result, label: str) -> None:
    """Rövid numerikus összefoglaló stdout-ra."""
    try:
        post = result.posterior
    except Exception as e:
        log(f"[{label}] Nem sikerült a posterior betöltése: {e}")
        return

    log(f"Posterior összefoglaló (label={label}):")
    for name in ["Omega0", "A_disc", "omega", "phi",
                 "Omega_astro0", "alpha", "alpha_astro"]:
        if name in post.columns:
            median = post[name].median()
            low, high = np.quantile(post[name], [0.05, 0.95])
            log(f"  {name}: {median:.3e} (90% CI: [{low:.3e}, {high:.3e}])")


# ----------------------------------------------------------------------
# 8. MULTI-WINDOW ÖSSZEKOMBINÁLÁS
# ----------------------------------------------------------------------

def run_multi_window_pygwb(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-window mód:
      - multi_t0–multi_tf tartományt felvágjuk window_duration hosszú
        ablakokra,
      - minden ablakra lefuttatjuk a pygwb_pipe-ot,
      - a sikeres ablakokra Ω_hat és sigma spektrumot számolunk,
      - ezeket inverz varianciával súlyozva kombináljuk:

        Ω_comb = sum(Ω_i / σ_i^2) / sum(1/σ_i^2)
        σ_comb = (sum(1/σ_i^2))^{-1/2}
    """
    base = config

    multi_t0 = base.multi_t0 if base.multi_t0 is not None else base.t0
    multi_tf = base.multi_tf if base.multi_tf is not None else base.tf
    dt = base.multi_window_duration

    if multi_tf <= multi_t0:
        raise RuntimeError("multi_tf <= multi_t0 – ellenőrizd a multi-window időket.")
    if dt <= 0:
        raise RuntimeError("multi_window_duration <= 0.")

    # Ablaklista
    windows: List[Tuple[int, int]] = []
    t = multi_t0
    while t + dt <= multi_tf:
        windows.append((t, t + dt))
        t += dt

    if not windows:
        raise RuntimeError("Nincs egyetlen multi-window ablak sem – ellenőrizd a multi_t0/multi_tf/dt értékeket.")

    log(f"Multi-window mód: {len(windows)} ablak, egyenként {dt} s (~{dt/3600:.2f} óra).")

    freqs_ref = None
    sum_num = None
    sum_den = None
    n_good = 0
    total_T = 0.0

    for i, (t0, tf) in enumerate(windows):
        log(f"=== Ablak {i+1}/{len(windows)}: t0={t0}, tf={tf} (Δt={tf-t0} s) ===")

        # Ablak-specifikus Config
        local = replace(base)
        local.t0 = int(t0)
        local.tf = int(tf)
        local.workdir = base.workdir / f"win_{i}"
        local.pygwb_output_dir = base.pygwb_output_dir / f"win_{i}"
        local.pe_output_dir = base.pe_output_dir  # bilby-t csak egyszer futtatjuk a végén

        try:
            param_file = write_pygwb_parameters_file(local)
            run_pygwb_pipe(local, param_file)
        except RuntimeError as e:
            msg = str(e)

            skip_reasons = [
                "discontiguous TimeSeries",
                "Cannot find a GWOSC dataset",
                "Cannot find a GWOSC dataset for",
                "`x` must contain at least 2 elements",
                "CubicSpline",
                "TimeoutError",
                "The read operation timed out",
                "HTTPError",
                "502 Server Error",
                "Proxy Error",
            ]

            if any(reason in msg for reason in skip_reasons):
                log(f"Ablak {i+1}: kihagyva (GWOSC adat hiány / nem folytonos időszak).")
                continue
            else:
                raise

        try:
            npz_path = guess_point_estimate_file(local)
            freqs, omega_hat, sigma = load_pygwb_point_estimate(
                npz_path, flow=base.flow, fhigh=base.fhigh
            )
        except Exception as e:
            log(f"Ablak {i+1}: kimenet beolvasási hiba, kihagyva: {e}")
            continue

        if freqs_ref is None:
            freqs_ref = freqs
            sum_num = omega_hat / sigma**2
            sum_den = 1.0 / sigma**2
        else:
            if not np.array_equal(freqs_ref, freqs):
                raise RuntimeError("Frekvenciarács eltérés ablakok között – nem tudom egyszerűen kombinálni.")
            sum_num += omega_hat / sigma**2
            sum_den += 1.0 / sigma**2

        n_good += 1
        total_T += (tf - t0)
        log(f"Ablak {i+1}: sikeres. Eddig {n_good} ablak, összesített idő ~{total_T/3600:.2f} óra.")

    if n_good == 0:
        raise RuntimeError("Egyetlen multi-window ablak sem futott sikeresen (mind lyukas / hibás).")

    omega_comb = sum_num / sum_den
    sigma_comb = 1.0 / np.sqrt(sum_den)

    log(f"Multi-window kombináció kész: {n_good} ablak, effektív idő ~{total_T/3600:.2f} óra.")
    return freqs_ref, omega_comb, sigma_comb


# ----------------------------------------------------------------------
# 9. FŐ FOLYAMAT
# ----------------------------------------------------------------------

def main():
    config = Config()

    # Fizika-motivált fix paraméterek:
    #   - COS lejtő: α = 0 (flat SGWB, LVK benchmark)
    #   - asztro lejtő: α_astro = 2/3 (BBH asztro SGWB)
    config.fixed_parameters = {
        "alpha": 0.0,
        "alpha_astro": 2.0 / 3.0,
    }

    # Multi-window beállítás: ~3 nap, 1 órás ablakokra bontva
    config.use_multi_window = True
    config.multi_t0 = 1247644138
    config.multi_tf = config.multi_t0 + 10 * 24 * 3600   # ~5 nap
    # config.multi_tf = config.multi_t0 + 1 * 3600   # ~1 óra
    config.multi_window_duration = 3600                 # 1 órás ablakok

    # --- Függőségek ---
    check_dependencies(config)

    # --- pygwb / Ω_hat, σ(f) spektrum egyszer kiszámítva ---
    if config.use_multi_window:
        freqs, omega_hat, sigma = run_multi_window_pygwb(config)
    else:
        param_file = write_pygwb_parameters_file(config)
        run_pygwb_pipe(config, param_file)
        npz_path = guess_point_estimate_file(config)
        freqs, omega_hat, sigma = load_pygwb_point_estimate(
            npz_path, flow=config.flow, fhigh=config.fhigh
        )

    # --- Több bilby-futás különböző modellekre / priorokra ---
    # Itt tudod egyszerűen bővíteni a listát, ha új eseteket akarsz.
    run_configs = [
        # (model_variant, prior_variant)
        ("cos_plus_astro", "wide"),
        ("cos_plus_astro", "lvk"),
        ("astro_only", "lvk"),
        # Példa: ha egyszer COS-only modellt is futtatnál:
        # ("cos_only", "lvk"),
    ]

    # adat előletöltés GWOSC-ról
    prefetch_gwosc_data(config)

    for model_variant, prior_variant in run_configs:
        cfg = replace(config)
        cfg.model_variant = model_variant
        cfg.prior_variant = prior_variant

        # Külön PE outdir az átláthatóság kedvéért
        label = f"{model_variant}_{prior_variant}"
        cfg.pe_output_dir = config.pe_output_dir / label
        ensure_dir(cfg.pe_output_dir)

        log(f"=== Bilby futás indul: model={model_variant}, prior={prior_variant}, label={label} ===")
        result = run_bilby_fit(cfg, freqs, omega_hat, sigma, label=label)
        summarize_posterior(result, label=label)


def prefetch_gwosc_data(config: Config) -> None:
    base = config
    multi_t0 = base.multi_t0 if base.multi_t0 is not None else base.t0
    multi_tf = base.multi_tf if base.multi_tf is not None else base.tf
    T = base.multi_window_duration

    nwin = int(np.floor((multi_tf - multi_t0) / T))
    windows = [(multi_t0 + i*T, multi_t0 + (i+1)*T) for i in range(nwin)]

    for ifo in base.ifos:
        for t0, tf in windows:
            try:
                log(f"[PREFETCH] {ifo} {t0}-{tf}")
                # sample_rate opcionális, de 16 kHz-es csatornához jól illik
                TimeSeries.fetch_open_data(ifo, t0, tf, cache=True)
            except Exception as e:
                log(f"[PREFETCH] {ifo} {t0}-{tf}: kihagyva (hiba: {e})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"HIBA: {e}")
        sys.exit(1)
