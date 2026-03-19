# build_step_operator_ts2.py
import numpy as np
from typing import Dict, Iterable, List, Tuple, Optional

Array = np.ndarray
LocalBlock = Tuple[Iterable[int], Array]  # (local indices, g_x block)

def _exp_i_dt_half_g(g: Array, dt: float) -> Array:
    """Compute exp(i * dt/2 * g) for Hermitian g via eigen-decomposition."""
    w, V = np.linalg.eigh(g)
    return V @ np.diag(np.exp(1j * (dt / 2.0) * w)) @ V.conj().T

def _apply_local(U_global: Array, U_local: Array, idx: Iterable[int]) -> Array:
    """Right-multiply the (idx x idx) block of U_global by U_local."""
    idx = list(idx)
    sub = U_global[np.ix_(idx, idx)]
    U_global[np.ix_(idx, idx)] = U_local @ sub
    return U_global

def build_step_operator_ts2(
    dim: int,
    local_blocks: List[LocalBlock],
    dt: float,
    proj_phys: Optional[Array] = None,
    proj_tau: Optional[Dict[str, Array]] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    TS2 scheme:
      C_TS2 = P_phys * [ prod_x exp(i * dt/2 * g_x) ] * [ prod_x^rev exp(i * dt/2 * g_x) ] * P_phys
      C_tau[tau] = P_tau * C_TS2 * P_tau
    """
    if proj_phys is None:
        proj_phys = np.eye(dim, dtype=complex)

    # 1) Local exponentials
    halves: List[Tuple[List[int], Array]] = []
    for idx, g in local_blocks:
        Ux_half = _exp_i_dt_half_g(g, dt)
        halves.append((list(idx), Ux_half))

    # 2) First half-product (forward)
    U_forward = np.eye(dim, dtype=complex)
    for idx, Ux_half in halves:
        U_forward = _apply_local(U_forward, Ux_half, idx)

    # 3) Second half-product (reverse order)
    U_backward = np.eye(dim, dtype=complex)
    for idx, Ux_half in reversed(halves):
        U_backward = _apply_local(U_backward, Ux_half, idx)

    # 4) Projective assembly
    C_TS2 = proj_phys @ (U_forward @ U_backward) @ proj_phys

    # 5) Sector blocks
    C_tau: Dict[str, Array] = {}
    if proj_tau is not None:
        for tau, P in proj_tau.items():
            C_tau[tau] = P @ C_TS2 @ P

    return C_TS2, C_tau

# --- Mini demo ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dim = 6

    # Example: three local Hermitian blocks (with overlaps)
    def herm(n):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (A + A.conj().T) / 2

    H1, H2, H3 = herm(2), herm(2), herm(3)

    local_blocks = [
        ([0, 1], H1),
        ([2, 3], H2),
        ([3, 4, 5], H3),
    ]

    dt = 0.1
    P_phys = np.eye(dim, dtype=complex)
    P_tau = {
        "tau_A": np.diag([1, 1, 1, 0, 0, 0]).astype(complex),
        "tau_B": np.diag([0, 0, 0, 1, 1, 1]).astype(complex),
    }

    C, blocks = build_step_operator_ts2(dim, local_blocks, dt, proj_phys=P_phys, proj_tau=P_tau)
    print("[TS2] ||C||_F =", np.linalg.norm(C, "fro"))
    for k, M in blocks.items():
        print(f"[TS2] ||C^{k}||_F =", np.linalg.norm(M, "fro"))
