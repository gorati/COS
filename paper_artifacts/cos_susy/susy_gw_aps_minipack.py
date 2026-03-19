# susy_gw_aps_minipack.py
# ASCII-only, NumPy-only. Produces LaTeX rows + CSV for SUSY pairing & closure sanity.
# Includes controlled SUSY-breaking perturbation delta for benchmarking.

import math
import numpy as np
import csv

# ----------------------------- Utilities -----------------------------

def ndtr(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ndtri(p):
    if p <= 0.0: return -math.inf
    if p >= 1.0: return math.inf
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
        m = float(vals.mean())
        return m, m, m
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
    return theta_hat, lo, hi

# ----------------------------- Model pieces -----------------------------

def superpotential_m_phi(phi, lam=1.0, mu=1.0):
    phi_star = +mu / math.sqrt(lam)
    m = 2.0 * lam * phi_star
    return phi_star, m

def twist_diff_matrix(N, a, theta):
    D = np.zeros((N, N), dtype=complex)
    for x in range(N):
        xp = (x + 1) % N
        xm = (x - 1) % N
        phase_p = np.exp(1j * (theta if xp == 0 else 0.0))
        phase_m = np.exp(-1j * (theta if x == 0 else 0.0)) if xm == N - 1 else 1.0
        D[x, xp] += +0.5 * phase_p
        D[x, xm] += -0.5 * phase_m
    return D / a

def gw_correction(D, a, tau=0.5, steps=1):
    N = D.shape[0]
    Z = np.zeros_like(D)
    Dchi = np.block([[Z, D], [D.conj().T, Z]])
    I = np.eye(2*N, dtype=complex)
    gamma5 = np.block([[np.eye(N), Z],[Z, -np.eye(N)]])
    X = gamma5 @ Dchi + Dchi @ gamma5 - a * Dchi @ gamma5 @ Dchi
    Dchi_corr = Dchi - tau * X
    return Dchi_corr[0:N, N:2*N]

def aps_projector(N):
    if N < 4:
        return np.eye(N)
    e0 = np.zeros((N,1)); e0[0,0] = 1.0
    eL = np.zeros((N,1)); eL[N-1,0] = 1.0
    E = np.hstack([e0, eL])
    B = np.array([[0.0, +1.0], [-1.0, 0.0]], dtype=float)
    w, U = np.linalg.eig(B)
    keep = [i for i in range(len(w)) if np.real(w[i]) >= 0.0]
    dis = [i for i in range(len(w)) if i not in keep]
    if len(dis) == 0:
        return np.eye(N)
    Udis = U[:, dis]
    Wdis = E @ Udis
    Q, _ = np.linalg.qr(Wdis)
    P = np.eye(N) - Q @ Q.conj().T
    Ufull, S, Vh = np.linalg.svd(P)
    rank = int(round(np.sum(S > 1e-12)))
    return Ufull[:, :rank]

def build_blocks(N, a, theta, gw_step=True, aps=True):
    D0 = twist_diff_matrix(N, a, theta)
    D = gw_correction(D0, a, tau=0.5, steps=1) if gw_step else D0
    P = aps_projector(N) if aps else np.eye(N)
    Dp = P.conj().T @ D @ P
    return Dp, P

# ----------------------------- Core computation -----------------------------

def spectra_pairing_and_closure(N, a, theta, m, aps=True, gw_step=True,
                                zero_tol=1e-10, eps=1e-3, rng=None):
    """
    Compute SUSY pairing and closure metrics with *imaginary* antisymmetric
    SUSY-breaking delta, ensuring small but real spectral splitting.
    """
    # --- Build Dirac block and projector ---
    D, P = build_blocks(N, a, theta, gw_step=gw_step, aps=aps)
    HB = (D @ D.conj().T) + (m*m) * np.eye(D.shape[0])
    HF = (D.conj().T @ D) + (m*m) * np.eye(D.shape[1])

    # --- Controlled SUSY-breaking perturbation delta (asymmetric blocks) ---
    if eps > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        DeltaB = eps * (rng.standard_normal(HB.shape) + 1j * rng.standard_normal(HB.shape))
        DeltaF = eps * (rng.standard_normal(HF.shape) + 1j * rng.standard_normal(HF.shape))
        # Make them Hermitian to keep H selfadjoint
        DeltaB = (DeltaB + DeltaB.conj().T) / 2.0
        DeltaF = (DeltaF + DeltaF.conj().T) / 2.0
        HB = HB + DeltaB
        HF = HF - DeltaF  # opposite phase -> soft SUSY breaking

    # --- Supercharges ---
    ZB = np.zeros_like(HB)
    ZF = np.zeros_like(HF)
    Q = np.block([[ZB, D],
                  [np.zeros((D.shape[1], D.shape[0])), ZF]])
    Qdg = np.block([[ZB, np.zeros((D.shape[0], D.shape[1]))],
                    [D.conj().T, ZF]])

    # --- Physical SUSY Hamiltonian (block form) ---
    H_phys = np.block([[HB, np.zeros((HB.shape[0], HF.shape[1]))],
                       [np.zeros((HF.shape[0], HB.shape[1])), HF]])

    # --- Closure residual ---
    R = (Q @ Qdg + Qdg @ Q) - 2.0 * H_phys
    num = np.linalg.norm(R, ord='fro')
    den = max(1e-16, np.linalg.norm(H_phys, ord='fro'))
    closure = float(num / den)

    # --- Spectra and pairing metric ---
    eB = np.linalg.eigvalsh(HB)
    eF = np.linalg.eigvalsh(HF)
    eB_pos = eB[eB > zero_tol]
    eF_pos = eF[eF > zero_tol]
    K = min(eB_pos.size, eF_pos.size)
    pairing = float('nan') if K == 0 else float(np.max(np.abs(eB_pos[:K] - eF_pos[:K])))

    n_zero_B = int(np.sum(eB <= zero_tol))
    n_zero_F = int(np.sum(eF <= zero_tol))
    return pairing, closure, n_zero_B, n_zero_F

# ----------------------------- Main experiment -----------------------------

def run_benchmark(N=64, a=1.0, trials=40, seed=2025,
                  lam=1.0, mu=1.0,
                  aps=True, gw_step=True,
                  eps=1e-3,
                  ci_B=10000, ci_seed=4242,
                  csv_out="susy_gw_aps_minipack.csv"):
    rng = np.random.default_rng(seed)
    phi_star, m = superpotential_m_phi(0.0, lam=lam, mu=mu)
    pair_vals, clos_vals, zeros_B, zeros_F, thetas = [], [], [], [], []

    for _ in range(trials):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        thetas.append(theta)
        p, c, zB, zF = spectra_pairing_and_closure(
            N=N, a=a, theta=theta, m=m,
            aps=aps, gw_step=gw_step,
            zero_tol=1e-10,
            eps=eps, rng=rng
        )
        pair_vals.append(p)
        clos_vals.append(c)
        zeros_B.append(zB)
        zeros_F.append(zF)

    pair_arr = np.array([v for v in pair_vals if np.isfinite(v)], dtype=float)
    clos_arr = np.array(clos_vals, dtype=float)
    p_mean, p_lo, p_hi = bca_ci_mean(pair_arr, B=ci_B, seed=ci_seed, alpha=0.05)
    c_mean, c_lo, c_hi = bca_ci_mean(clos_arr, B=ci_B, seed=ci_seed+1, alpha=0.05)

    print("LATEX_ROWS (metric & N & seed & mean & [95% CI]):")
    print("$\\max_n\\,|E_B-E_F|$ & {} & {} & {:.3e} & [{:.3e}, {:.3e}] \\\\".format(
        N, seed, p_mean, p_lo, p_hi))
    row = "$\\|\\{{Q,Q^\\dagger\\}}\\!-\\!2H\\|_\\mathrm{{F}}/\\|H\\|_\\mathrm{{F}}$ & {} & {} & {:.3e} & [{:.3e}, {:.3e}] \\\\"
    print(row.format(N, seed, c_mean, c_lo, c_hi))

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N","seed","trials","lam","mu","m","aps","gw_step","eps",
                    "pair_mean","pair_lo","pair_hi",
                    "closure_mean","closure_lo","closure_hi",
                    "zerosB_mean","zerosF_mean"])
        w.writerow([N, seed, trials, lam, mu, m, int(aps), int(gw_step), eps,
                    f"{p_mean:.6e}", f"{p_lo:.6e}", f"{p_hi:.6e}",
                    f"{c_mean:.6e}", f"{c_lo:.6e}", f"{c_hi:.6e}",
                    float(np.mean(zeros_B)), float(np.mean(zeros_F))])

    print("\nDIAGNOSTICS:")
    print(f"phi_star = {phi_star:.6f}, mass m = {m:.6f}, eps = {eps:.1e}")
    print(f"avg zero modes: B = {np.mean(zeros_B):.2f}, F = {np.mean(zeros_F):.2f}")
    print("csv written:", csv_out)

if __name__ == "__main__":
    run_benchmark(N=128, a=1.0, trials=50, seed=2025,
                  lam=1.0, mu=1.0,
                  aps=True, gw_step=True,
                  eps=1e-3,
                  ci_B=10000, ci_seed=4242,
                  csv_out="susy_gw_aps_minipack.csv")
