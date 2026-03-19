# Curvature table at central grid point: discrete vs continuous with 95% BCa CI
# ASCII-only, reproducible.

import math
import numpy as np

# ---------- "True" Godel slice metric ----------
def true_metric_components(x, a=1.0):
    # g_xx = a^2, g_yy = a^2 * 0.5 * exp(2x), g_xy = 0
    return a * a, a * a * 0.5 * np.exp(2.0 * x), 0.0

# ---------- Discrete reconstruction from noisy edge-lengths ----------
def reconstruct_metric_point(x, h, sigma_rel=0.03, a=1.0, rng=np.random.default_rng()):
    gxx, gyy, gxy = true_metric_components(x, a=a)
    x_mid = min(x + 0.5 * h, 1.0)
    gxx_m, gyy_m, _ = true_metric_components(x_mid, a=a)

    Lx = math.sqrt(gxx) * h
    Ly = math.sqrt(gyy) * h
    Ld = math.sqrt(gxx_m + gyy_m - 2.0 * gxy) * h

    Lx *= (1.0 + rng.normal(0.0, sigma_rel))
    Ly *= (1.0 + rng.normal(0.0, sigma_rel))
    Ld *= (1.0 + rng.normal(0.0, sigma_rel))

    d_x2 = max(Lx, 0.0) ** 2
    d_y2 = max(Ly, 0.0) ** 2
    d_xy2 = max(Ld, 0.0) ** 2

    gxx_rec = d_x2 / (h * h)
    gyy_rec = d_y2 / (h * h)
    gxy_rec = 0.5 * (d_x2 + d_y2 - d_xy2) / (h * h)
    return gxx_rec, gyy_rec, gxy_rec

def deriv_center(arr, h, i_c):
    n = len(arr)
    if 0 < i_c < n - 1:
        return (arr[i_c + 1] - arr[i_c - 1]) / (2.0 * h)
    elif i_c == 0:
        return (arr[1] - arr[0]) / h
    else:
        return (arr[-1] - arr[-2]) / h

# ---------- Discrete/continuous Gamma and R2 at central index ----------
def curvature_triplet_once(n=5, sigma_rel=0.03, a=1.0, seed=4242):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1) if n > 1 else 1.0
    i = n // 2

    rec = np.array([reconstruct_metric_point(x, h, sigma_rel, a=a, rng=rng) for x in xs])
    tru = np.array([true_metric_components(x, a=a) for x in xs])

    def gamma_x_yy(gx, gy, dgy):
        gdet = gx * gy
        return -0.5 * (gy / gdet) * dgy

    def gamma_y_xy(gx, gy, dgy):
        gdet = gx * gy
        return 0.5 * (gx / gdet) * dgy

    dgy_d_disc = deriv_center(rec[:, 1], h, i)
    dgy_d_cont = deriv_center(tru[:, 1], h, i)

    Gx_disc = gamma_x_yy(rec[i, 0], rec[i, 1], dgy_d_disc)
    Gy_disc = gamma_y_xy(rec[i, 0], rec[i, 1], dgy_d_disc)
    Gx_cont = gamma_x_yy(tru[i, 0], tru[i, 1], dgy_d_cont)
    Gy_cont = gamma_y_xy(tru[i, 0], tru[i, 1], dgy_d_cont)

    # R2 ~ - d/dx(Gamma^x_{yy}) - Gamma^x_{yy} * Gamma^y_{xy}
    def gamma_x_series(field):
        nloc = len(field)
        out = np.zeros(nloc)
        for k in range(nloc):
            gx, gy = field[k, 0], field[k, 1]
            if 0 < k < nloc - 1:
                dgy = (field[k + 1, 1] - field[k - 1, 1]) / (2.0 * h)
            elif k == 0:
                dgy = (field[1, 1] - field[0, 1]) / h
            else:
                dgy = (field[-1, 1] - field[-2, 1]) / h
            out[k] = gamma_x_yy(gx, gy, dgy)
        return out

    Gx_series_disc = gamma_x_series(rec)
    if 0 < i < n - 1:
        dGx_dx_disc = (Gx_series_disc[i + 1] - Gx_series_disc[i - 1]) / (2.0 * h)
    elif i == 0:
        dGx_dx_disc = (Gx_series_disc[1] - Gx_series_disc[0]) / h
    else:
        dGx_dx_disc = (Gx_series_disc[-1] - Gx_series_disc[-2]) / h
    R2_disc = - dGx_dx_disc - (Gx_disc * Gy_disc)

    Gx_series_cont = gamma_x_series(tru)
    if 0 < i < n - 1:
        dGx_dx_cont = (Gx_series_cont[i + 1] - Gx_series_cont[i - 1]) / (2.0 * h)
    elif i == 0:
        dGx_dx_cont = (Gx_series_cont[1] - Gx_series_cont[0]) / h
    else:
        dGx_dx_cont = (Gx_series_cont[-1] - Gx_series_cont[-2]) / h
    R2_cont = - dGx_dx_cont - (Gx_cont * Gy_cont)

    return (Gx_disc, Gy_disc, R2_disc), (Gx_cont, Gy_cont, R2_cont)

# ---------- BCa bootstrap (same as before; ASCII, no scipy) ----------
def ndtr(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ndtri(p):
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return num / den

def bca_ci_mean(vals, B=10000, seed=1234, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n < 3:
        raise ValueError("BCa requires at least 3 observations.")
    theta_hat = float(vals.mean())

    boots = np.empty(B, dtype=float)
    for b in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        boots[b] = sample.mean()
    boots.sort()

    prop = (boots < theta_hat).mean()
    eps = 1e-12
    prop = min(max(prop, eps), 1.0 - eps)
    z0 = ndtri(prop)

    s = vals.sum()
    theta_jack = (s - vals) / (n - 1)
    theta_bar = float(theta_jack.mean())
    u = theta_bar - theta_jack
    num = float(np.sum(u ** 3))
    den = 6.0 * (float(np.sum(u ** 2)) ** 1.5)
    a = (num / den) if den != 0.0 else 0.0

    z_lo = ndtri(alpha / 2.0)
    z_hi = ndtri(1.0 - alpha / 2.0)

    def adj_quant(z):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return alpha / 2.0 if z == z_lo else 1.0 - alpha / 2.0
        return ndtr(z0 + (z0 + z) / denom)

    a1 = min(max(adj_quant(z_lo), 0.0), 1.0)
    a2 = min(max(adj_quant(z_hi), 0.0), 1.0)

    lo = float(np.quantile(boots, a1, method="nearest"))
    hi = float(np.quantile(boots, a2, method="nearest"))
    return lo, hi

# ---------- Multi-trial wrapper and LaTeX printing ----------
def curvature_table_with_ci(n=5, sigma_rel=0.03, a=1.0,
                            trials=100, seed_sim=7777,
                            B_boot=10000, seed_boot=8888):
    base = np.random.default_rng(seed_sim)
    gx_disc = []
    gy_disc = []
    r2_disc = []

    # Continuous (deterministic) reference from a single noiseless call
    (disc_once, cont_once) = curvature_triplet_once(n=n, sigma_rel=sigma_rel, a=a, seed=base.integers(0, 1 << 31))
    # recompute continuous with zero noise path (same function computes both)
    # we keep cont_once as the reference
    Gx_cont, Gy_cont, R2_cont = cont_once

    for _ in range(trials):
        seed = base.integers(0, 1 << 31)
        disc, cont = curvature_triplet_once(n=n, sigma_rel=sigma_rel, a=a, seed=seed)
        gx_disc.append(disc[0])
        gy_disc.append(disc[1])
        r2_disc.append(disc[2])

    gx_disc = np.asarray(gx_disc, dtype=float)
    gy_disc = np.asarray(gy_disc, dtype=float)
    r2_disc = np.asarray(r2_disc, dtype=float)

    def mean_ci(arr):
        m = float(arr.mean())
        lo, hi = bca_ci_mean(arr, B=B_boot, seed=seed_boot, alpha=0.05)
        return m, lo, hi

    m_gx, lo_gx, hi_gx = mean_ci(gx_disc)
    m_gy, lo_gy, hi_gy = mean_ci(gy_disc)
    m_r2, lo_r2, hi_r2 = mean_ci(r2_disc)

    def perr(d, c):
        denom = max(1.0, abs(c))
        return 100.0 * abs(d - c) / denom

    # LaTeX-friendly rows
    print("LATEX_ROWS (Quantity & Discrete(mean) & 95% CI & Continuous & Diff[%]):")
    print(f"$\\Gamma^x_{{\\;yy}}$ & {m_gx:.3f} & [{lo_gx:.3f}, {hi_gx:.3f}] & {Gx_cont:.3f} & {perr(m_gx, Gx_cont):.1f} \\\\")
    print(f"$\\Gamma^y_{{\\;xy}}$ & {m_gy:.3f} & [{lo_gy:.3f}, {hi_gy:.3f}] & {Gy_cont:.3f} & {perr(m_gy, Gy_cont):.1f} \\\\")
    print(f"$R_{{\\!(2)}}$       & {m_r2:.3f} & [{lo_r2:.3f}, {hi_r2:.3f}] & {R2_cont:.3f} & {perr(m_r2, R2_cont):.1f} \\\\")

# Example run (match paper defaults if you like)
if __name__ == "__main__":
    curvature_table_with_ci(n=5, sigma_rel=0.03, a=1.0,
                            trials=100, seed_sim=7777,
                            B_boot=10000, seed_boot=8888)
