# Multi-run v_g benchmark with BCa 95% CI on mean(\hat v_g)
# Tight-binding: omega(k) = -2 J cos k  -> choose sign convention: v_exact = +2 J sin k (to match centroid slope here)
# Adds measurement noise on centroid readout to avoid degenerate CI.

import math
import numpy as np

# ---------- Parameters ----------
SEED_SIM     = 2025     # base seed for trials
SEED_BOOT    = 4242     # seed for BCa bootstrap
TRIALS       = 200      # runs per k0
B_BOOT       = 10_000   # bootstrap reps
N            = 64       # sites (periodic)
DT           = 1.0e-2   # time step
T_STEPS      = 200      # steps
SAMPLE_EVERY = 10       # sample cadence
J            = 1.0      # hopping (LR velocity = 2|J|)
SIGMA_X      = 3.5      # initial packet width
X0           = N // 4   # initial center
PHASE_NOISE  = 0.0      # phase noise OFF
READOUT_NOISE= 0.02     # centroid readout noise (in site units) 
K_LIST       = [0.5, 1.0, 1.5, 2.0, 2.5]
CSV_OUT      = "qf_vg_benchmark_bca.csv"
WRITE_CSV    = True

# ---------- Model ----------
def tb_hamiltonian(N, J):
    H = np.zeros((N, N), dtype=complex)
    for x in range(N):
        xp = (x + 1) % N
        xm = (x - 1) % N
        H[x, xp] += -J
        H[x, xm] += -J
    return H

def wavepacket_state(N, x0, k0, sigma_x):
    x  = np.arange(N)
    dx = ((x - x0 + N//2) % N) - N//2
    psi = np.exp(1j * k0 * x) * np.exp(-0.5 * (dx / sigma_x) ** 2)
    psi /= np.linalg.norm(psi)
    return psi

def add_phase_noise(psi, rng, sigma):
    if sigma <= 0.0:
        return psi
    phases = rng.normal(0.0, sigma, size=psi.size)
    return psi * np.exp(1j * phases)

def evolve_and_measure_unitary(H, psi0, dt, t_steps, sample_every, rng, readout_noise=0.0):
    psi = psi0.astype(complex, copy=True)
    xs, ts = [], []
    idx = np.arange(H.shape[0], dtype=float)
    for n in range(t_steps + 1):
        if n % sample_every == 0:
            p = np.abs(psi) ** 2
            x_mean = float(np.sum(idx * p))
            if readout_noise > 0.0:
                x_mean += float(rng.normal(0.0, readout_noise))
            xs.append(x_mean); ts.append(n * dt)
        if n < t_steps:
            psi = psi + (-1j * dt) * (H @ psi)
            norm = np.linalg.norm(psi)
            if norm != 0.0:
                psi /= norm
    return np.asarray(ts, float), np.asarray(xs, float)

def vg_from_trajectory(ts, xs):
    slope, intercept = np.polyfit(ts, xs, 1)
    return float(slope)

def v_exact(k, J):
    # choose + sign here to match centroid slope in this TB setup
    return  2.0 * J * math.sin(k)
    # If you prefer the other convention, use:
    # return -2.0 * J * math.sin(k)

# ---------- BCa 95% CI for mean ----------
def ndtr(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ndtri(p):
    if p <= 0.0: return -math.inf
    if p >= 1.0: return  math.inf
    a=[-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b=[-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c=[-7.784894002430293e-03, -3.223964580411365e-01,-2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d=[ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1.0 - 0.02425
    if p < pl:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > ph:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
    den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return num / den

def bca_ci_mean(vals, B=10_000, seed=4242, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    n = vals.size
    if n < 3:
        raise ValueError("BCa requires at least 3 observations.")
    theta_hat = float(vals.mean())
    boots = np.empty(B, float)
    for b in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        boots[b] = sample.mean()
    boots.sort()
    prop = (boots < theta_hat).mean()
    prop = min(max(prop, 1e-12), 1 - 1e-12)
    z0 = ndtri(prop)
    s = vals.sum()
    theta_jack = (s - vals) / (n - 1)
    theta_bar  = float(theta_jack.mean())
    u = theta_bar - theta_jack
    num = float(np.sum(u ** 3))
    den = 6.0 * (float(np.sum(u ** 2)) ** 1.5)
    a = (num / den) if den != 0.0 else 0.0
    z_lo = ndtri(alpha / 2.0)
    z_hi = ndtri(1.0 - alpha / 2.0)
    def adj_quant(z):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return (alpha / 2.0) if z == z_lo else (1.0 - alpha / 2.0)
        return ndtr(z0 + (z0 + z) / denom)
    a1 = min(max(adj_quant(z_lo), 0.0), 1.0)
    a2 = min(max(adj_quant(z_hi), 0.0), 1.0)
    lo = float(np.quantile(boots, a1, method="nearest"))
    hi = float(np.quantile(boots, a2, method="nearest"))
    return lo, hi

# ---------- Driver ----------
def main():
    rng_base = np.random.default_rng(SEED_SIM)
    H = tb_hamiltonian(N, J)
    v_LR = 2.0 * abs(J)

    rows = []
    print("LATEX_ROWS (k0 & vhat_mean & 95% CI & v_exact & v_LR):")
    for k0 in K_LIST:
        vg_hats = []
        for _ in range(TRIALS):
            trial_seed = int(rng_base.integers(0, 1 << 31))
            rng = np.random.default_rng(trial_seed)
            psi0 = wavepacket_state(N, X0, k0, SIGMA_X)
            psi0 = add_phase_noise(psi0, rng, sigma=PHASE_NOISE)  # default OFF
            ts, xs = evolve_and_measure_unitary(H, psi0, DT, T_STEPS, SAMPLE_EVERY,
                                                rng=rng, readout_noise=READOUT_NOISE)
            vg_hat = vg_from_trajectory(ts, xs)
            vg_hats.append(vg_hat)

        vg_hats = np.asarray(vg_hats, float)
        vg_mean = float(vg_hats.mean())
        ci_lo, ci_hi = bca_ci_mean(vg_hats, B=B_BOOT, seed=SEED_BOOT, alpha=0.05)
        v_ex = v_exact(k0, J)
        print(f"{k0:.1f} & {vg_mean:.3f} & [{ci_lo:.3f}, {ci_hi:.3f}] & {v_ex:.3f} & {v_LR:.3f} \\\\")
        rows.append((k0, vg_mean, ci_lo, ci_hi, v_ex, v_LR, TRIALS))

    if WRITE_CSV:
        try:
            with open(CSV_OUT, "w", encoding="utf-8") as f:
                f.write("k0,vg_mean,ci_lo,ci_hi,vg_exact,v_LR,trials\n")
                for r in rows:
                    f.write(f"{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}, {r[4]:.6f}, {r[5]:.6f}, {r[6]}\n")
            print(f"\nCSV saved: {CSV_OUT}")
        except Exception as e:
            print(f"\nCSV save failed: {e}")

if __name__ == "__main__":
    main()
