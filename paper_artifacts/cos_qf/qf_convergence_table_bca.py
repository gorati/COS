# Godel-fit convergence table: mean, std and 95% BCa CI
# Reproducible (seeded) synthetic simulation.

import math
import numpy as np

# ---------- Physics / simulation ----------

def true_metric_components(x, a=1.0):
    # Example "true" metric components
    # g_xx = a^2, g_yy = a^2 * 0.5 * exp(2x), g_xy = 0
    return a * a, a * a * 0.5 * np.exp(2.0 * x), 0.0

def reconstruct_metric_point(x, h, sigma_rel=0.03, a=1.0, rng=np.random.default_rng()):
    # Reconstruct metric point from noisy "edge lengths"
    gxx, gyy, gxy = true_metric_components(x, a=a)
    x_mid = min(x + 0.5 * h, 1.0)
    gxx_m, gyy_m, _ = true_metric_components(x_mid, a=a)

    # Deterministic lengths
    Lx = math.sqrt(gxx) * h
    Ly = math.sqrt(gyy) * h
    Ld = math.sqrt(gxx_m + gyy_m - 2.0 * gxy) * h

    # Relative Gaussian noise
    Lx *= (1.0 + rng.normal(0.0, sigma_rel))
    Ly *= (1.0 + rng.normal(0.0, sigma_rel))
    Ld *= (1.0 + rng.normal(0.0, sigma_rel))

    d_x2 = max(Lx, 0.0) ** 2
    d_y2 = max(Ly, 0.0) ** 2
    d_xy2 = max(Ld, 0.0) ** 2

    # Simple inverse reconstruction
    gxx_rec = d_x2 / (h * h)
    gyy_rec = d_y2 / (h * h)
    gxy_rec = 0.5 * (d_x2 + d_y2 - d_xy2) / (h * h)
    return gxx_rec, gyy_rec, gxy_rec

def epsilon_for_grid(n, sigma_rel=0.03, a=1.0, rng=np.random.default_rng()):
    # Relative metric error epsilon on an n-point 1D grid in x
    xs = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1) if n > 1 else 1.0
    diff2 = 0.0
    tru2 = 0.0
    for x in xs:
        gxx, gyy, gxy = true_metric_components(x, a=a)
        rxx, ryy, rxy = reconstruct_metric_point(x, h, sigma_rel, a=a, rng=rng)
        diff2 += (rxx - gxx) ** 2 + (ryy - gyy) ** 2 + 2.0 * (rxy - gxy) ** 2
        tru2 += (gxx ** 2) + (gyy ** 2) + 2.0 * (gxy ** 2)
    return math.sqrt(diff2) / math.sqrt(tru2)

def epsilon_stats(N_list, sigma_rel=0.03, trials=100, seed=2025, a=1.0):
    # Repeat "trials" times for each N; return (N, mean, sd, vals)
    out = []
    base = np.random.default_rng(seed)
    for N in N_list:
        n = int(round(N ** 0.5))
        vals = []
        for _ in range(trials):
            rng = np.random.default_rng(base.integers(0, 1 << 31))
            vals.append(epsilon_for_grid(n, sigma_rel=sigma_rel, a=a, rng=rng))
        vals = np.asarray(vals, dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        out.append((N, mean, sd, vals))
    return out

def fit_alpha(points):
    # Fit <epsilon> ~ N^{-alpha}; points is list of (N, mean, sd, vals)
    x = np.log([N for (N, _, _, _) in points])
    y = np.log([m for (_, m, _, _) in points])
    A = np.vstack([np.ones_like(x), -x]).T
    c, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(alpha)

# ---------- Standard normal CDF and inverse (Acklam) ----------

def ndtr(z):
    # Standard normal CDF using erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ndtri(p):
    # Inverse standard normal CDF (Acklam approximation)
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

# ---------- BCa bootstrap CI for the mean ----------

def bca_ci_mean(vals, B=10000, seed=1234, alpha=0.05):
    # BCa (bias-corrected and accelerated) CI for mean(vals)
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n < 3:
        raise ValueError("BCa requires at least 3 observations.")
    theta_hat = float(vals.mean())

    # Bootstrap sample means
    boots = np.empty(B, dtype=float)
    for b in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        boots[b] = sample.mean()
    boots.sort()

    # Bias-correction z0
    prop = (boots < theta_hat).mean()
    eps = 1e-12
    prop = min(max(prop, eps), 1.0 - eps)
    z0 = ndtri(prop)

    # Jackknife acceleration a
    s = vals.sum()
    theta_jack = (s - vals) / (n - 1)  # leave-one-out means
    theta_bar = float(theta_jack.mean())
    u = theta_bar - theta_jack
    num = float(np.sum(u ** 3))
    den = 6.0 * (float(np.sum(u ** 2)) ** 1.5)
    a = (num / den) if den != 0.0 else 0.0

    # Adjusted quantiles
    z_lo = ndtri(alpha / 2.0)
    z_hi = ndtri(1.0 - alpha / 2.0)

    def adj_quant(z):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            # fallback to percentile
            return alpha / 2.0 if z == z_lo else 1.0 - alpha / 2.0
        return ndtr(z0 + (z0 + z) / denom)

    a1 = min(max(adj_quant(z_lo), 0.0), 1.0)
    a2 = min(max(adj_quant(z_hi), 0.0), 1.0)

    lo = float(np.quantile(boots, a1, method="nearest"))
    hi = float(np.quantile(boots, a2, method="nearest"))
    return lo, hi

# ---------- Driver ----------

def make_rows(N_LIST, TRIALS=100, SIGMA_REL=0.03, SEED=2026, B=10000, SEED_BOOT=4242, a=1.0):
    pts = epsilon_stats(N_LIST, sigma_rel=SIGMA_REL, trials=TRIALS, seed=SEED, a=a)
    rows = []
    for (N, mean, sd, vals) in pts:
        ci_low, ci_high = bca_ci_mean(vals, B=B, seed=SEED_BOOT, alpha=0.05)
        rows.append((N, mean, sd, ci_low, ci_high, vals))
    return rows

def maybe_save_csv(rows, path="godel_conv.csv"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("N,mean,sd,ci95\n")
            for (N, m, s, l, u, _vals) in rows:
                f.write(f"{N},{m:.6f},{s:.6f},\"[{l:.6f}, {u:.6f}]\"\n")
        return True, path
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # Parameters (keep consistent with the paper)
    N_LIST = [25, 49, 100]
    TRIALS = 100
    SIGMA_REL = 0.03
    SEED_SIM = 2026
    B_BOOT = 10000
    SEED_BOOT = 4242
    A_SCALE = 1.0

    rows = make_rows(
        N_LIST,
        TRIALS=TRIALS,
        SIGMA_REL=SIGMA_REL,
        SEED=SEED_SIM,
        B=B_BOOT,
        SEED_BOOT=SEED_BOOT,
        a=A_SCALE,
    )

    print("\nTABLE_ROWS (N, mean_eps, std_eps, [BCa 95% CI]):")
    for (N, m, s, l, u, _vals) in rows:
        print(f"{N} & {m:.4f} & {s:.4f} & [{l:.4f}, {u:.4f}] \\\\")

    coords = " ".join(
        [f"({N},{m:.4f}) +- (0,{s:.4f})" for (N, m, s, _l, _u, _v) in rows]
    )
    print("\nPGFPLOTS_COORDS:\n" + coords)

    alpha = fit_alpha([(r[0], r[1], r[2], r[5]) for r in rows])
    print("\nALPHA_FIT:\n" + f"{alpha:.3f}")

    ok, info = maybe_save_csv(rows, path="godel_conv.csv")
    if ok:
        print(f"\nCSV saved: {info}")
    else:
        print(f"\nCSV save failed: {info}")
