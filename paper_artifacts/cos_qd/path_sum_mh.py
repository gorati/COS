# path_sum_mh.py
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

Array = np.ndarray
RNG = np.random.Generator

@dataclass
class MHPathSample:
    path: List[Any]      # sequence of states Gamma_0 -> ... -> Gamma_T
    A_abs: float         # accumulated absolute amplitude (importance weight)
    phase: float         # accumulated phase S[gamma] (real), total factor = exp(i * phase)

def path_sum_mh(
    Gamma_i: Any,
    T: int,
    n_samples: int,
    propose_local_step,          # callable(state, rng) -> (Gamma_new, alpha_id, q_fwd, q_rev)
    local_abs_ratio,             # callable(Gamma_old, Gamma_new, alpha_id) -> r >= 0
    phase_increment,             # callable(Gamma_old, Gamma_new, alpha_id) -> delta_phase (float)
    measure_factor_abs=None,     # callable(alpha_id) -> positive float; default 1.0
    rng: Optional[RNG] = None,
) -> Tuple[complex, List[MHPathSample]]:
    """
    Metropolis-Hastings path-sum with phase reweighting.

    We sample paths gamma of fixed length T using importance weights |A[gamma]|.
    Estimator (phase reweighting):
        K_T_hat = sum_j |A_j| * exp(i * S_j) / sum_j |A_j|
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    if measure_factor_abs is None:
        measure_factor_abs = lambda alpha_id: 1.0

    samples: List[MHPathSample] = []
    gamma: List[Any] = [Gamma_i]
    A_abs: float = 1.0
    phase: float = 0.0

    while len(samples) < n_samples:
        Gamma_end = gamma[-1]
        Gamma_new, alpha_id, q_fwd, q_rev = propose_local_step(Gamma_end, rng)

        r = local_abs_ratio(Gamma_end, Gamma_new, alpha_id)
        acc_prob = min(1.0, r * (q_rev / max(q_fwd, 1e-300)))
        if rng.random() < acc_prob:
            gamma.append(Gamma_new)
            A_abs *= (r * measure_factor_abs(alpha_id))
            phase += phase_increment(Gamma_end, Gamma_new, alpha_id)

            if len(gamma) - 1 == T:
                samples.append(MHPathSample(path=list(gamma), A_abs=A_abs, phase=phase))
                gamma = [Gamma_i]
                A_abs = 1.0
                phase = 0.0

    num = np.sum([s.A_abs * np.exp(1j * s.phase) for s in samples], dtype=complex)
    den = np.sum([s.A_abs for s in samples], dtype=float)
    K_T_hat = num / max(den, 1e-300)
    return K_T_hat, samples

class ToyModel:
    """
    Minimal toy model to demonstrate the MH path-sum estimator.
    States: integers {0,1,2,...,N-1}
    Allowed moves: +/- 1 modulo N (alpha = +1 or -1)
    Abs ratio favors forward with exp(beta)
    Phase increment adds +/- phi0 per step
    Proposal is symmetric => q_rev/q_fwd = 1
    """
    def __init__(self, N: int = 8, beta: float = 0.3, phi0: float = 0.12):
        self.N = N; self.beta = beta; self.phi0 = phi0

    def propose_local_step(self, state: int, rng: RNG) -> Tuple[int, int, float, float]:
        alpha = 1 if rng.random() < 0.5 else -1
        new_state = (state + alpha) % self.N
        return new_state, alpha, 0.5, 0.5

    def local_abs_ratio(self, old: int, new: int, alpha: int) -> float:
        if (new - old) % self.N == 1:      # forward
            return float(np.exp(self.beta))
        elif (new - old) % self.N == self.N - 1:  # backward
            return float(np.exp(-self.beta))
        return 0.0

    def phase_increment(self, old: int, new: int, alpha: int) -> float:
        return self.phi0 if alpha == 1 else -self.phi0

    def measure_factor_abs(self, alpha: int) -> float:
        return 1.0

def demo():
    rng = np.random.default_rng(2025)
    model = ToyModel(N=8, beta=0.3, phi0=0.12)
    K_T_hat, samples = path_sum_mh(
        Gamma_i=0, T=12, n_samples=3000,
        propose_local_step=model.propose_local_step,
        local_abs_ratio=model.local_abs_ratio,
        phase_increment=model.phase_increment,
        measure_factor_abs=model.measure_factor_abs,
        rng=rng,
    )
    A_abs_vals = np.array([s.A_abs for s in samples], dtype=float)
    phases = np.array([s.phase for s in samples], dtype=float)
    print("[PathSumMH] K_T estimate:", K_T_hat)
    print("[PathSumMH] |K_T| =", np.abs(K_T_hat))
    print("[PathSumMH] mean |A| =", A_abs_vals.mean(), " std |A| =", A_abs_vals.std())
    print("[PathSumMH] mean phase =", phases.mean(), " std phase =", phases.std())

if __name__ == "__main__":
    demo()
