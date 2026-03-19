import numpy as np
from typing import List, Tuple, Callable

# Types
State = int
Path = List[State]

def propose_local_step(state: State, n_states: int, rng: np.random.Generator):
    """
    Simple symmetric nearest-neighbor proposal on a ring of n_states states.
    Returns: next_state, q_forward, q_reverse (both 1/2 here).
    """
    step = rng.choice([-1, 1])
    next_state = (state + step) % n_states
    q_forward = 0.5
    q_reverse = 0.5
    return next_state, q_forward, q_reverse

def local_matrix_element(C_alpha: np.ndarray, s_from: State, s_to: State) -> complex:
    """
    Toy: return matrix element <s_to| C_alpha | s_from>.
    """
    return C_alpha[s_to, s_from]

def path_sum_mh(
    Gamma_i: State,
    T: int,
    n_samples: int,
    C_alphas: List[np.ndarray],
    phase_of_alpha: Callable[[int], float],
    rng: np.random.Generator = np.random.default_rng(123),
) -> Tuple[complex, List[Tuple[Path, float, float]]]:
    """
    Simple Metropolis-Hastings sampler for path-sum over a toy finite state space.
    Returns the phase-reweighted estimator of K_T and collected samples.
    Each sample: (path, abs_weight, phase).
    """
    n_states = C_alphas[0].shape[0]
    path: Path = [Gamma_i]
    A_abs = 1.0
    samples = []

    for _ in range(n_samples):
        # propose a local move (choose alpha and next state)
        alpha_idx = rng.integers(0, len(C_alphas))
        s_from = path[-1]
        s_to, q_fwd, q_rev = propose_local_step(s_from, n_states, rng)

        # acceptance ratio based on matrix element magnitude
        num = abs(local_matrix_element(C_alphas[alpha_idx], s_from, s_to))
        den = 1.0  # identity reference in the toy
        r = (num / max(den, 1e-15)) * (q_rev / max(q_fwd, 1e-15))

        if rng.random() < min(1.0, r):
            path.append(s_to)
            A_abs *= num
        # when we reach T steps, record and reset
        if len(path) == T + 1:
            phase = phase_of_alpha(alpha_idx)
            samples.append((path.copy(), A_abs, phase))
            path = [Gamma_i]
            A_abs = 1.0

    # phase-reweighted estimator
    if samples:
        num = sum(w * np.exp(1j * ph) for _, w, ph in samples)
        den = sum(w for _, w, _ in samples)
        K_T_hat = num / max(den, 1e-30)
    else:
        K_T_hat = 0.0 + 0.0j

    return K_T_hat, samples

# --- mini demo ---
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 5
    # build a few toy "Kraus-like" matrices
    C0 = np.eye(n, dtype=complex)
    C1 = np.roll(np.eye(n, dtype=complex), 1, axis=0)  # shift-down
    C2 = np.roll(np.eye(n, dtype=complex), -1, axis=0) # shift-up
    C_alphas = [C0, 0.7*C1, 0.7*C2]

    def phase_of_alpha(alpha_idx: int) -> float:
        return 0.2 * alpha_idx

    K_T, samps = path_sum_mh(Gamma_i=0, T=4, n_samples=2000, C_alphas=C_alphas,
                             phase_of_alpha=phase_of_alpha, rng=rng)
    phases = np.array([ph for _, _, ph in samps]) if samps else np.array([0.0])
    mags = np.array([w for _, w, _ in samps]) if samps else np.array([0.0])
    print("[PathSumMH] K_T estimate:", K_T)
    print("[PathSumMH] |K_T| =", abs(K_T))
    print("[PathSumMH] mean |A| =", float(mags.mean()), " std |A| =", float(mags.std()))
    print("[PathSumMH] mean phase =", float(phases.mean()), " std phase =", float(phases.std()))
