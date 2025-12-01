import json
import numpy as np
import matplotlib.pyplot as plt
import dynesty
from dynesty import utils as dyfunc
from scipy.stats import norm
from typing import List, Dict, Any, Tuple

# ----------------------------------------------------------------------
# KONFIGURÁCIÓ: JSON fájlok listája (26 MI-időnyíl eredmény)
# ----------------------------------------------------------------------

JSON_FILES: List[str] = [
    "cmb_time_arrow_MI_commander_nside256_axes1000_seed12345.json",
    "cmb_time_arrow_MI_commander_nside256_axes1000_seed54321.json",
    "cmb_time_arrow_MI_commander_nside256_axes1000_seed98765.json",
    "cmb_time_arrow_MI_nilc_nside256_axes1000_seed12345.json",
    "cmb_time_arrow_MI_nilc_nside256_axes1000_seed54321.json",
    "cmb_time_arrow_MI_nilc_nside256_axes1000_seed98765.json",
    "cmb_time_arrow_MI_sevem_nside256_axes1000_seed12345.json",
    "cmb_time_arrow_MI_sevem_nside256_axes1000_seed54321.json",
    "cmb_time_arrow_MI_sevem_nside256_axes1000_seed98765.json",
    "cmb_time_arrow_MI_smica_nside128_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside128_axes1000_seed98765.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed12345_axis227-27.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed54321.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed54321_axis227-27.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed98765.json",
    "cmb_time_arrow_MI_smica_nside256_axes1000_seed98765_axis227-27.json",
    "cmb_time_arrow_MI_smica_nside256_ent32_mi16_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent32_mi32_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent32_mi48_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent64_mi16_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent64_mi32_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent64_mi48_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent96_mi16_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent96_mi32_axes1000_seed12345.json",
    "cmb_time_arrow_MI_smica_nside256_ent96_mi48_axes1000_seed12345.json",
]

# Nested Sampling alapbeállítások
N_LIVE = 1000
DLOGZ_STOP = 0.5
MAXITER = 20000


# ----------------------------------------------------------------------
# 1. ADATBETÖLTÉS
# ----------------------------------------------------------------------

def load_cmb_time_data(file_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    random_monos_all: összes random tengely monotonitásai
    measured_monos_all: COS tengelyek monotonitásai (fájlonként 1)
    delta_mi_matrix: (N_files, N_scales) COS-tengely ΔMI-párok
    """
    all_random_monos: List[float] = []
    measured_monos: List[float] = []
    all_delta_mi: List[np.ndarray] = []
    lmax_pairs_ref: Any = None

    for filename in file_list:
        try:
            with open(filename, "r") as f:
                data: Dict[str, Any] = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Figyelmeztetés: A fájl nem található: {filename}. Kihagyás.")
            continue
        except Exception as e:
            print(f"❌ Hiba a(z) {filename} betöltése közben: {e}. Kihagyás.")
            continue

        axes = data.get("axes", [])
        if not axes:
            print(f"⚠️ Figyelmeztetés: Nincs 'axes' mező a(z) {filename} fájlban. Kihagyás.")
            continue

        # 0. tengely = COS / kitüntetett tengely
        main_axis = axes[0]
        mono_val = main_axis.get("monotonicity_delta_MI", None)
        delta_mi_pairs = main_axis.get("delta_MI_pairs", None)

        if mono_val is None or delta_mi_pairs is None:
            print(f"⚠️ Figyelmeztetés: Hiányzó mező a(z) {filename} fájlban. Kihagyás.")
            continue

        measured_monos.append(float(mono_val))
        all_delta_mi.append(np.array(delta_mi_pairs, dtype=float))

        # random tengelyek
        random_axes = axes[1:-1] if len(axes) > 2 else []
        for ax in random_axes:
            if "monotonicity_delta_MI" in ax:
                all_random_monos.append(float(ax["monotonicity_delta_MI"]))

        if lmax_pairs_ref is None:
            lmax_pairs_ref = data.get("lmax_pairs", None)

    if len(all_delta_mi) == 0 or len(measured_monos) == 0 or len(all_random_monos) == 0:
        raise ValueError("Nincs elegendő adat a JSON fájlokból.")

    delta_mi_matrix = np.vstack(all_delta_mi)           # (N_files, N_scales)
    random_monos_all = np.array(all_random_monos, dtype=float)
    measured_monos_all = np.array(measured_monos, dtype=float)

    return random_monos_all, measured_monos_all, delta_mi_matrix


def extract_lmax_grid_from_example(file_list: List[str]) -> np.ndarray:
    """
    Egy példa JSON-ból kiolvassa az lmax párokat (felső értékek).
    """
    for filename in file_list:
        try:
            with open(filename, "r") as f:
                data: Dict[str, Any] = json.load(f)
        except Exception:
            continue

        if "lmax_pairs" in data:
            lmax_pairs = data["lmax_pairs"]
            try:
                lmax_grid = np.array([pair[1] for pair in lmax_pairs], dtype=int)
                return lmax_grid
            except Exception:
                pass

    # fallback, ha valamiért nincs lmax_pairs
    return np.arange(9, dtype=int)


# ----------------------------------------------------------------------
# 2. MONOTONITÁS + ΔMI STATISZTIKA
# ----------------------------------------------------------------------

def compute_global_monotonicity_stats(random_monos: np.ndarray,
                                      measured_monos: np.ndarray) -> Dict[str, float]:
    """
    Gauss-illesztés a random monotonitások eloszlására,
    majd COS-átlag p-értéke és szigma-eltérése.
    """
    mu_h0, std_h0 = norm.fit(random_monos)
    measured_mean = float(np.mean(measured_monos))
    p_value = norm.cdf(measured_mean, loc=mu_h0, scale=std_h0)   # egyoldali
    sigma_value = float(np.abs(norm.ppf(p_value)))
    return {
        "mu_h0": float(mu_h0),
        "std_h0": float(std_h0),
        "measured_mean_mono": measured_mean,
        "p_value_one_sided": float(p_value),
        "sigma": sigma_value,
    }


def compute_delta_mi_stats(delta_mi_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Átlagolt ΔMI görbe + skálánkénti szórás 26 fájl alapján.
    """
    avg_delta_mi = np.mean(delta_mi_matrix, axis=0)
    if delta_mi_matrix.shape[0] > 1:
        std_delta_mi = np.std(delta_mi_matrix, axis=0, ddof=1)
    else:
        std_delta_mi = np.zeros_like(avg_delta_mi)

    # numerikus stabilitás
    min_err = 1e-4
    std_delta_mi = np.where(std_delta_mi < min_err, min_err, std_delta_mi)
    return avg_delta_mi, std_delta_mi


# ----------------------------------------------------------------------
# 2/b. MONOTONITÁS-HISZTOGRAM ÁBRA
# ----------------------------------------------------------------------

def plot_monotonicity_distribution(random_monos: np.ndarray,
                                   global_stats: Dict[str, float],
                                   outfile: str = "mi_monotonicity_significance.pdf") -> None:
    """
    A ΔMI monotonitás eloszlásának ábrája:
    - hisztogram a random tengelyekre
    - Gauss-illesztés
    - COS-átlag piros vonallal
    - p-értéknek megfelelő árnyékolt terület
    """
    mu = global_stats["mu_h0"]
    sigma0 = global_stats["std_h0"]
    measured_mean = global_stats["measured_mean_mono"]
    sigma_dev = global_stats["sigma"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Hisztogram (sűrűségre normálva)
    n_bins = 50
    counts, bins, patches = ax.hist(
        random_monos,
        bins=n_bins,
        density=True,
        alpha=0.7,
        label=f"Véletlen Tengelyek (N={len(random_monos)})",
    )

    # Gauss-illesztés
    x_vals = np.linspace(bins[0], bins[-1], 400)
    gauss = norm.pdf(x_vals, loc=mu, scale=sigma0)
    ax.plot(x_vals, gauss, "k--", linewidth=2,
            label=f"Gauss-illesztés (μ={mu:.3f}, σ₀={sigma0:.3f})")

    # P-érték terület (bal oldali farok, measured_mean-ig)
    x_fill = np.linspace(bins[0], measured_mean, 300)
    y_fill = norm.pdf(x_fill, loc=mu, scale=sigma0)
    ax.fill_between(x_fill, y_fill, 0, color="tab:red", alpha=0.3, label="P-érték terület")

    # Mért COS érték
    ax.axvline(measured_mean, color="red", linewidth=2,
               label=f"Mért COS-Érték ({measured_mean:.4f})")

    ax.set_title("A ΔMI Monotonitás Eloszlása: ΛCDM vs. COS Tengely", fontsize=14)
    ax.set_xlabel("Korrelációs Koeficiens (Monotonitás: ΔMI skálafüggése)", fontsize=12)
    ax.set_ylabel("Sűrűség (Normalizált Előfordulás)", fontsize=12)

    # Sigma-eltérés dobozban
    text = f"Kvantitatív Eltérés (COS): {sigma_dev:.2f}σ"
    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    print(f"📉 Monotonitás-eloszlás ábra elmentve: {outfile}")


# ----------------------------------------------------------------------
# 3. LIKELIHOODOK: KONSTANS, LINEÁRIS, KVADRATIKUS
# ----------------------------------------------------------------------

def make_loglike_const(x: np.ndarray,
                       y: np.ndarray,
                       y_err: np.ndarray):
    """
    Konstans modell:
      y_i ~ N(mu_C, sigma_tot_i^2),
      sigma_tot_i^2 = y_err_i^2 + sigma_int^2.
    Paraméterek: [mu_C, log10_sigma_int]
    """
    def _loglike(params: np.ndarray) -> float:
        mu_C, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma_tot = np.sqrt(y_err**2 + sigma_int**2)
        return float(np.sum(norm.logpdf(y, loc=mu_C, scale=sigma_tot)))
    return _loglike


def prior_transform_const(u: np.ndarray) -> np.ndarray:
    """
    mu_C ~ U(0,0.3)
    log10_sigma_int ~ U(-4,-1)
    """
    P = np.zeros(2)
    P[0] = 0.0 + 0.3 * u[0]
    P[1] = -4.0 + 3.0 * u[1]
    return P


def make_loglike_linear(x: np.ndarray,
                        y: np.ndarray,
                        y_err: np.ndarray):
    """
    Lineáris modell:
      y_i ~ N(mu_L + m x_i, sigma_tot_i^2)
    """
    def _loglike(params: np.ndarray) -> float:
        mu_L, m, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma_tot = np.sqrt(y_err**2 + sigma_int**2)
        model_y = mu_L + m * x
        return float(np.sum(norm.logpdf(y, loc=model_y, scale=sigma_tot)))
    return _loglike


def prior_transform_linear_symmetric(u: np.ndarray) -> np.ndarray:
    """
    mu_L ~ U(0,0.3)
    m    ~ U(-0.1,0.1)   (szimmetrikus prior)
    log10_sigma_int ~ U(-4,-1)
    """
    P = np.zeros(3)
    P[0] = 0.0 + 0.3 * u[0]
    P[1] = -0.1 + 0.2 * u[1]
    P[2] = -4.0 + 3.0 * u[2]
    return P


def prior_transform_linear_negative(u: np.ndarray) -> np.ndarray:
    """
    COS-specifikus lineáris modell:
    mu_L ~ U(0,0.3)
    m    ~ U(-0.1,0.0)   (elméletileg csökkenő időnyíl)
    log10_sigma_int ~ U(-4,-1)
    """
    P = np.zeros(3)
    P[0] = 0.0 + 0.3 * u[0]
    P[1] = -0.1 + 0.1 * u[1]
    P[2] = -4.0 + 3.0 * u[2]
    return P


def make_loglike_quadratic(x: np.ndarray,
                           y: np.ndarray,
                           y_err: np.ndarray):
    """
    Kvadratikus modell:
      y_i ~ N(mu_Q + a x_i + b x_i^2, sigma_tot_i^2)
    Paraméterek: [mu_Q, a, b, log10_sigma_int]
    """
    def _loglike(params: np.ndarray) -> float:
        mu_Q, a, b, log10_sigma_int = params
        sigma_int = 10.0 ** log10_sigma_int
        sigma_tot = np.sqrt(y_err**2 + sigma_int**2)
        model_y = mu_Q + a * x + b * x**2
        return float(np.sum(norm.logpdf(y, loc=model_y, scale=sigma_tot)))
    return _loglike


def prior_transform_quadratic(u: np.ndarray) -> np.ndarray:
    """
    mu_Q ~ U(0,0.3)
    a    ~ U(-0.1,0.1)
    b    ~ U(-0.05,0.05)
    log10_sigma_int ~ U(-4,-1)
    """
    P = np.zeros(4)
    P[0] = 0.0 + 0.3 * u[0]        # mu_Q
    P[1] = -0.1 + 0.2 * u[1]       # a
    P[2] = -0.05 + 0.10 * u[2]     # b
    P[3] = -4.0 + 3.0 * u[3]       # log10_sigma_int
    return P


# ----------------------------------------------------------------------
# 4. HIERARCHIKUS LINEÁRIS MODELL
# ----------------------------------------------------------------------

def make_loglike_hierarchical_linear(x: np.ndarray,
                                     delta_mi_matrix: np.ndarray):
    """
    Hierarchikus modell:
      y_{j,i} ~ N(mu_L + m x_i + c_j, sigma_int^2)

    - mu_L, m globális paraméterek
    - c_j fájl-specifikus offsetek (random effektek)
    - sigma_int közös szórás minden pontra

    Itt külön "méréshibát" (y_err) már nem használunk; a szórás
    egységesen sigma_int.
    """
    n_files, n_scales = delta_mi_matrix.shape

    def _loglike(params: np.ndarray) -> float:
        mu_L = params[0]
        m = params[1]
        log10_sigma_int = params[2]
        sigma_int = 10.0 ** log10_sigma_int
        if sigma_int <= 0:
            return -np.inf

        offsets = params[4:4 + n_files]   # c_j

        inv_var = 1.0 / (sigma_int ** 2)
        log_norm = -0.5 * np.log(2 * np.pi * sigma_int ** 2)
        logL = 0.0

        for j in range(n_files):
            c_j = offsets[j]
            model = mu_L + m * x + c_j
            resid = delta_mi_matrix[j, :] - model
            logL += np.sum(log_norm - 0.5 * inv_var * resid ** 2)

        return float(logL)

    return _loglike


def make_prior_transform_hierarchical_linear(n_files: int):
    """
    Hierarchikus prior:
      mu_L ~ U(0,0.3)
      m    ~ U(-0.1,0.1)
      log10_sigma_int ~ U(-4,-1)
      sigma_c: log10_sigma_c ~ U(-3,-0.5)  (offsetek tipikus nagysága)
      c_j ~ N(0, sigma_c^2)  (norm.ppf-inverzzel generálva)
    """
    ndim = 4 + n_files

    def _prior(u: np.ndarray) -> np.ndarray:
        if len(u) != ndim:
            raise ValueError(f"Prior input dim={len(u)} != expected {ndim}")
        P = np.zeros(ndim)
        P[0] = 0.0 + 0.3 * u[0]      # mu_L
        P[1] = -0.1 + 0.2 * u[1]     # m (szimmetrikus prior)
        P[2] = -4.0 + 3.0 * u[2]     # log10_sigma_int
        P[3] = -3.0 + 2.5 * u[3]     # log10_sigma_c
        sigma_c = 10.0 ** P[3]
        for j in range(n_files):
            u_j = np.clip(u[4 + j], 1e-6, 1.0 - 1e-6)
            P[4 + j] = sigma_c * norm.ppf(u_j)
        return P

    return _prior, ndim


# ----------------------------------------------------------------------
# 5. NESTED SAMPLING FUTTATÁS + JEFFREYS-SKÁLA
# ----------------------------------------------------------------------

def run_nested_sampling(loglike_func,
                        prior_transform_func,
                        ndim: int,
                        model_name: str) -> Dict[str, Any]:
    """
    Dynesty Nested Sampling driver.
    """
    print(f"\n🚀 Futtatás: {model_name} ({ndim} dimenziós modell)")

    sampler = dynesty.NestedSampler(
        loglike_func,
        prior_transform_func,
        ndim,
        nlive=N_LIVE,
        bound="multi",
        sample="rwalk",
    )

    sampler.run_nested(dlogz=DLOGZ_STOP, maxiter=MAXITER)
    results = sampler.results

    logZ = float(results.logz[-1])
    logZerr = float(results.logzerr[-1])

    # Súlyozott minták -> egyenlő súlyú minták
    weights = np.exp(results.logwt - results.logz[-1])
    samples_equal = dyfunc.resample_equal(results.samples, weights)

    param_means = np.mean(samples_equal, axis=0)
    param_stds = np.std(samples_equal, axis=0)

    print(f"✅ Befejezve. Log-Evidence: log(Z) = {logZ:.3f} ± {logZerr:.3f}")
    for i, (m, s) in enumerate(zip(param_means, param_stds)):
        print(f"   param[{i}] = {m:.5f} ± {s:.5f}")

    return {
        "logZ": logZ,
        "logZerr": logZerr,
        "param_means": param_means.tolist(),
        "param_stds": param_stds.tolist(),
        "samples": samples_equal.tolist(),
    }


def jeffreys_strength(logB10: float) -> str:
    """
    Jeffreys-skála log10(B) szerint.
    """
    if logB10 < 0.5:
        return "Nem említésre méltó"
    elif logB10 < 1.0:
        return "Gyenge / éppen említésre méltó"
    elif logB10 < 2.0:
        return "Erős"
    else:
        return "Döntő"


# ----------------------------------------------------------------------
# 6. EREDMÉNYMENTÉS + ÁBRÁK
# ----------------------------------------------------------------------

def save_results_json(outfile: str,
                      lmax_grid: np.ndarray,
                      avg_delta_mi: np.ndarray,
                      std_delta_mi: np.ndarray,
                      global_stats: Dict[str, float],
                      results_const: Dict[str, Any],
                      results_lin_sym: Dict[str, Any],
                      results_lin_neg: Dict[str, Any],
                      results_quad: Dict[str, Any],
                      results_hier: Dict[str, Any] | None = None) -> None:
    """
    Minden lényeges szám mentése JSON-be, plusz Bayes-tényezők.
    """
    logZ_C = results_const["logZ"]
    logZ_Lsym = results_lin_sym["logZ"]
    logZ_Lneg = results_lin_neg["logZ"]
    logZ_Q = results_quad["logZ"]

    delta_logZ_LsymC = logZ_Lsym - logZ_C
    delta_logZ_LnegC = logZ_Lneg - logZ_C
    delta_logZ_QC = logZ_Q - logZ_C

    logB10_LsymC = delta_logZ_LsymC / np.log(10.0)
    logB10_LnegC = delta_logZ_LnegC / np.log(10.0)
    logB10_QC = delta_logZ_QC / np.log(10.0)

    out: Dict[str, Any] = {
        "input_files": JSON_FILES,
        "lmax_grid": lmax_grid.tolist(),
        "avg_delta_mi": avg_delta_mi.tolist(),
        "std_delta_mi": std_delta_mi.tolist(),
        "global_monotonicity_stats": global_stats,
        "bayes_models": {
            "const": {
                "logZ": results_const["logZ"],
                "logZerr": results_const["logZerr"],
                "param_means": results_const["param_means"],
                "param_stds": results_const["param_stds"],
            },
            "linear_symmetric": {
                "logZ": results_lin_sym["logZ"],
                "logZerr": results_lin_sym["logZerr"],
                "param_means": results_lin_sym["param_means"],
                "param_stds": results_lin_sym["param_stds"],
            },
            "linear_negative": {
                "logZ": results_lin_neg["logZ"],
                "logZerr": results_lin_neg["logZerr"],
                "param_means": results_lin_neg["param_means"],
                "param_stds": results_lin_neg["param_stds"],
            },
            "quadratic": {
                "logZ": results_quad["logZ"],
                "logZerr": results_quad["logZerr"],
                "param_means": results_quad["param_means"],
                "param_stds": results_quad["param_stds"],
            },
            "bayes_factors": {
                "delta_logZ_linear_symmetric_minus_const": float(delta_logZ_LsymC),
                "delta_logZ_linear_negative_minus_const": float(delta_logZ_LnegC),
                "delta_logZ_quadratic_minus_const": float(delta_logZ_QC),
                "log10_B_linear_symmetric_over_const": float(logB10_LsymC),
                "log10_B_linear_negative_over_const": float(logB10_LnegC),
                "log10_B_quadratic_over_const": float(logB10_QC),
                "jeffreys_strength_linear_symmetric_over_const": jeffreys_strength(logB10_LsymC),
                "jeffreys_strength_linear_negative_over_const": jeffreys_strength(logB10_LnegC),
                "jeffreys_strength_quadratic_over_const": jeffreys_strength(logB10_QC),
            },
        },
    }

    if results_hier is not None:
        out["hierarchical_linear"] = {
            "logZ": results_hier["logZ"],
            "logZerr": results_hier["logZerr"],
            "param_means": results_hier["param_means"],
            "param_stds": results_hier["param_stds"],
            "note": "A hierarchikus modell más likelihoodot használ (összes fájl x skála), "
                    "ezért logZ közvetlenül nem hasonlítható a fenti log(Z)-ekhez.",
        }

    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"💾 Eredmények elmentve: {outfile}")


def plot_delta_mi_with_models(lmax_grid: np.ndarray,
                              avg_delta_mi: np.ndarray,
                              std_delta_mi: np.ndarray,
                              results_const: Dict[str, Any],
                              results_lin_sym: Dict[str, Any],
                              results_lin_neg: Dict[str, Any],
                              results_quad: Dict[str, Any],
                              outfile: str = "delta_mi_scale_dependence_bayes.pdf") -> None:
    """
    ΔMI(ℓ_max) + konstans, lineáris, COS-lineáris és kvadratikus fit.
    """
    x_index = np.arange(len(avg_delta_mi))

    mu_C = results_const["param_means"][0]

    mu_L_sym = results_lin_sym["param_means"][0]
    m_sym = results_lin_sym["param_means"][1]

    mu_L_neg = results_lin_neg["param_means"][0]
    m_neg = results_lin_neg["param_means"][1]

    mu_Q = results_quad["param_means"][0]
    a_Q = results_quad["param_means"][1]
    b_Q = results_quad["param_means"][2]

    y_const = mu_C * np.ones_like(x_index)
    y_lin_sym = mu_L_sym + m_sym * x_index
    y_lin_neg = mu_L_neg + m_neg * x_index
    y_quad = mu_Q + a_Q * x_index + b_Q * x_index**2

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        lmax_grid,
        avg_delta_mi,
        yerr=std_delta_mi,
        fmt="o",
        capsize=4,
        label="Átlagolt COS-tengely ΔMI (±1σ)",
    )

    ax.plot(lmax_grid, y_const, label="Konstans modell (M_C)", linestyle=":", linewidth=2)
    ax.plot(lmax_grid, y_lin_sym, label="Lineáris modell (M_lin, m∈[-0.1,0.1])", linestyle="--", linewidth=2)
    ax.plot(lmax_grid, y_lin_neg, label="COS-spec. lineáris (M_COS, m<0)", linestyle="-.", linewidth=2)
    ax.plot(lmax_grid, y_quad, label="Kvadratikus modell (M_quad)", linestyle="-", linewidth=1.5)

    ax.set_title("ΔMI skálafüggés és Bayes-i modellek illesztése", fontsize=14)
    ax.set_xlabel("ℓ_max (skála)", fontsize=12)
    ax.set_ylabel("ΔMI (COS-tengely)", fontsize=12)
    ax.set_xticks(lmax_grid)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    print(f"📈 Ábra elmentve: {outfile}")


def plot_logZ_bar(results_const: Dict[str, Any],
                  results_lin_sym: Dict[str, Any],
                  results_lin_neg: Dict[str, Any],
                  results_quad: Dict[str, Any],
                  outfile: str = "bayes_logZ_comparison.pdf") -> None:
    """
    log(Z) oszlopdiagram a négy modellre (azonos adat, eltérő modell).
    """
    labels = ["Konstans", "Lin. szimmetrikus", "Lin. negatív", "Kvadratikus"]
    logZ_vals = [
        results_const["logZ"],
        results_lin_sym["logZ"],
        results_lin_neg["logZ"],
        results_quad["logZ"],
    ]
    logZ_errs = [
        results_const["logZerr"],
        results_lin_sym["logZerr"],
        results_lin_neg["logZerr"],
        results_quad["logZerr"],
    ]

    x = np.arange(len(labels))
    width = 0.6

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, logZ_vals, yerr=logZ_errs, width=width, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("log(Z)")
    ax.set_title("Bayes-i modellbizonyítékok összehasonlítása\n(azonos adat, különböző modellek)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    print(f"📊 Ábra elmentve: {outfile}")


# ----------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------

def main():
    print("--- COS Bayes-i Modell Összehasonlító Készlet (robosztus + kvadratikus + hierarchikus) ---")
    print(f"Feldolgozandó JSON fájlok száma: {len(JSON_FILES)}")

    # 1) Adatok
    random_monos, measured_monos, delta_mi_matrix = load_cmb_time_data(JSON_FILES)
    lmax_grid = extract_lmax_grid_from_example(JSON_FILES)

    print(f"\n[1] Összes véletlen tengely száma: {len(random_monos)}")
    print(f"[1] COS tengelyek száma (fájlok száma): {len(measured_monos)}")
    print(f"[1] Delta_MI mátrix alakja: {delta_mi_matrix.shape}")

    # 2) Globális monotonitás
    global_stats = compute_global_monotonicity_stats(random_monos, measured_monos)
    print("\n[2] Globális monotonitás statisztika (ΛCDM vs. COS):")
    print(f"    Random eloszlás középérték (μ): {global_stats['mu_h0']:.4f}")
    print(f"    Random eloszlás szórás (σ₀): {global_stats['std_h0']:.4f}")
    print(f"    Mért COS monotonitás átlag: {global_stats['measured_mean_mono']:.4f}")
    print(f"    Egyoldali p-érték: {global_stats['p_value_one_sided']:.4e}")
    print(f"    Szigma-eltérés: {global_stats['sigma']:.2f} σ")

    # 2/b) Monotonitás-eloszlás ábra
    plot_monotonicity_distribution(
        random_monos,
        global_stats,
        outfile="mi_monotonicity_significance.pdf",
    )

    # 3) ΔMI skálafüggés
    avg_delta_mi, std_delta_mi = compute_delta_mi_stats(delta_mi_matrix)
    x_index = np.arange(len(avg_delta_mi))
    print("\n[3] ΔMI skálafüggés (átlag ± szórás):")
    for i, (ell, val, err) in enumerate(zip(lmax_grid, avg_delta_mi, std_delta_mi)):
        print(f"    i={i}, ℓ_max={ell:3d}: ΔMI={val:.4f} ± {err:.4f}")

    # 4) Bayes-i modellek: konstans, lineáris, COS-lineáris, kvadratikus
    loglike_C = make_loglike_const(x_index, avg_delta_mi, std_delta_mi)
    loglike_Lsym = make_loglike_linear(x_index, avg_delta_mi, std_delta_mi)
    loglike_Lneg = make_loglike_linear(x_index, avg_delta_mi, std_delta_mi)
    loglike_Q = make_loglike_quadratic(x_index, avg_delta_mi, std_delta_mi)

    results_const = run_nested_sampling(loglike_C, prior_transform_const, ndim=2,
                                        model_name="Konstans modell (M_C)")
    results_lin_sym = run_nested_sampling(loglike_Lsym, prior_transform_linear_symmetric, ndim=3,
                                          model_name="Lineáris modell (M_lin, szimmetrikus prior)")
    results_lin_neg = run_nested_sampling(loglike_Lneg, prior_transform_linear_negative, ndim=3,
                                          model_name="Lineáris COS-specifikus modell (M_COS, m<0)")
    results_quad = run_nested_sampling(loglike_Q, prior_transform_quadratic, ndim=4,
                                       model_name="Kvadratikus modell (M_quad)")

    # 5) Bayes-tényezők (azonos adat!)
    logZ_C = results_const["logZ"]
    logZ_Lsym = results_lin_sym["logZ"]
    logZ_Lneg = results_lin_neg["logZ"]
    logZ_Q = results_quad["logZ"]

    delta_logZ_LsymC = logZ_Lsym - logZ_C
    delta_logZ_LnegC = logZ_Lneg - logZ_C
    delta_logZ_QC = logZ_Q - logZ_C

    logB10_LsymC = delta_logZ_LsymC / np.log(10.0)
    logB10_LnegC = delta_logZ_LnegC / np.log(10.0)
    logB10_QC = delta_logZ_QC / np.log(10.0)

    print("\n[6] Bayes-i modellösszehasonlítás (azonos adat, különböző modellek):")
    print(f"    ΔlogZ (M_lin - M_C)  = {delta_logZ_LsymC:.3f} -> log10(B) = {logB10_LsymC:.3f} -> {jeffreys_strength(logB10_LsymC)}")
    print(f"    ΔlogZ (M_COS - M_C)  = {delta_logZ_LnegC:.3f} -> log10(B) = {logB10_LnegC:.3f} -> {jeffreys_strength(logB10_LnegC)}")
    print(f"    ΔlogZ (M_quad - M_C) = {delta_logZ_QC:.3f} -> log10(B) = {logB10_QC:.3f} -> {jeffreys_strength(logB10_QC)}")

    # 6) Hierarchikus lineáris modell: teljes delta_mi_matrix
    print("\n[7] Hierarchikus lineáris modell futtatása az összes (fájl, skála) pontra...")
    loglike_hier = make_loglike_hierarchical_linear(x_index, delta_mi_matrix)
    prior_hier, ndim_hier = make_prior_transform_hierarchical_linear(delta_mi_matrix.shape[0])
    results_hier = run_nested_sampling(loglike_hier, prior_hier, ndim=ndim_hier,
                                       model_name="Hierarchikus lineáris modell (M_hier)")

    # 7) JSON + ábrák
    save_results_json(
        outfile="cos_bayes_MI_results.json",
        lmax_grid=lmax_grid,
        avg_delta_mi=avg_delta_mi,
        std_delta_mi=std_delta_mi,
        global_stats=global_stats,
        results_const=results_const,
        results_lin_sym=results_lin_sym,
        results_lin_neg=results_lin_neg,
        results_quad=results_quad,
        results_hier=results_hier,
    )

    plot_delta_mi_with_models(
        lmax_grid=lmax_grid,
        avg_delta_mi=avg_delta_mi,
        std_delta_mi=std_delta_mi,
        results_const=results_const,
        results_lin_sym=results_lin_sym,
        results_lin_neg=results_lin_neg,
        results_quad=results_quad,
        outfile="delta_mi_scale_dependence_bayes.pdf",
    )

    plot_logZ_bar(
        results_const=results_const,
        results_lin_sym=results_lin_sym,
        results_lin_neg=results_lin_neg,
        results_quad=results_quad,
        outfile="bayes_logZ_comparison.pdf",
    )

    print("\nMegjegyzés: a hierarchikus modell log(Z) értéke eltérő likelihoodon alapul, "
          "ezért nem közvetlenül hasonlítható a fenti log(Z)-ekhez.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Import hiba:", e)
        print("Ellenőrizze, hogy a szükséges csomagok telepítve vannak (numpy, matplotlib, dynesty, scipy).")
    except Exception as e:
        print("❌ Váratlan hiba történt a futás során:", e)
