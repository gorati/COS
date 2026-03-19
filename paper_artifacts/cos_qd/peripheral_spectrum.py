import numpy as np

try:
    from scipy.sparse.linalg import eigs as arpack_eigs
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def peripheral_spectrum(C_tau: np.ndarray, k_max: int = 8, eps: float = 1e-8):
    """
    Peripheral spectrum extraction:
      - Prefer ARPACK (largest magnitude),
      - Fallback to dense eigen if needed.
    Returns: (per, all_eigs) where per = [(lambda, v), ...]
    """
    n = C_tau.shape[0]
    # ARPACK only for 1 <= k < n
    k = max(1, min(k_max, max(1, n - 1)))

    eigvals = None
    eigvecs = None

    if _HAVE_SCIPY and n >= 2:
        try:
            eigvals, eigvecs = arpack_eigs(C_tau, k=k, which="LM")
        except Exception:
            eigvals, eigvecs = None, None

    if eigvals is None:
        # Fallback to full eigen-decomposition
        eigvals, eigvecs = np.linalg.eig(C_tau)

    mask = np.abs(np.abs(eigvals) - 1.0) < eps
    per = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals)) if mask[i]]
    return per, eigvals

def stability_checks(C_tau: np.ndarray, Pi_phys_tau: np.ndarray):
    """
    (1) Projective norm test: || C^* C - P_phys ||_F
    (2) Gershgorin estimate (by row sums).
    """
    proj_norm = np.linalg.norm(C_tau.conj().T @ C_tau - Pi_phys_tau, ord="fro")

    diag = np.diag(C_tau)
    R = np.abs(C_tau).sum(axis=1) - np.abs(diag)
    gersh_min = np.min(np.abs(diag) - R)
    gersh_max = np.max(np.abs(diag) + R)

    return {
        "proj_norm_fro": proj_norm,
        "gershgorin_min": gersh_min,
        "gershgorin_max": gersh_max,
    }

# ------------------------- DEMO -------------------------
if __name__ == "__main__":
    np.random.seed(7)
    n = 6
    # Nearly unitary demo block to have eigenvalues on the unit circle
    A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2 * n)
    Q, _ = np.linalg.qr(A)
    C_tau_demo = Q
    Pi_phys_tau_demo = np.eye(n, dtype=complex)

    per, all_eigs = peripheral_spectrum(C_tau_demo, k_max=4, eps=1e-6)
    print("[Peripheral] count with |lambda| ~ 1:", len(per))
    for idx, (lam, _) in enumerate(per, 1):
        print(
            "{:2d}. lambda ~= {: .6f}{:+.6f}j  |lambda|={:.6f}  arg(lambda)={:.6f}".format(
                idx, lam.real, lam.imag, abs(lam), np.angle(lam)
            )
        )

    stats = stability_checks(C_tau_demo, Pi_phys_tau_demo)
    print("[Stability] projective norm test (Fro):", stats["proj_norm_fro"])
    print(
        "[Gershgorin] estimated |lambda|-range ~ [{:.6f}, {:.6f}]".format(
            stats["gershgorin_min"], stats["gershgorin_max"]
        )
    )
