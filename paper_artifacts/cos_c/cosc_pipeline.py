# COS-C pipeline: de Sitter, tensor modes - constant Delta-tau, midpoint omega^2, windowed late freeze-out
"""
Key ideas (ASCII-safe):
- Uniform grid: Delta_tau = constant -> the centered 2nd-order scheme is more consistent.
- omega^2(tau) = k^2 - 2/tau^2 evaluated at the cell midpoint (tau_{i+1/2} = tau_i + Delta_tau/2).
- Do not measure P_t at a single instant; instead, on superhorizon scales, average |h|^2 within a window:
  H_conf/k in [R1, R2] (e.g., R1=50, R2=200).
- Correct normalization (M_pl = 1):  P_t(k) = (4 k^3 / pi^2) * |v|^2 / a^2.
  (Sketch: mu'' + (k^2 - a''/a) mu = 0, with mu = (a M_pl / sqrt(2)) h_A; then sum over the two polarizations.)

Run: python cosc_pipeline.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np
import math

# -----------------------------
# Parameters and state
# -----------------------------

@dataclass
class Params:
    Hc: float = 1e-5
    k_modes: np.ndarray = field(default_factory=lambda: np.array([1e-6, 3e-6, 1e-5]))
    n_steps: int = 600000
    tau0: float = -1.0e7
    dtau: float = 5.0e1           # CONSTANT step
    R1: float = 50.0              # freeze window lower edge: H_conf/k >= R1
    R2: float = 200.0             # freeze window upper edge: H_conf/k >= R2

@dataclass
class BackgroundState:
    i: int
    tau: float
    a: float
    H: float       # H_conf := -1/tau
    dtau: float

@dataclass
class ModeState:
    v_prev: complex
    v_curr: complex
    # Accumulate values inside the freeze window
    window_vals: List[float] = field(default_factory=list)
    frozen: bool = False
    Pt_store: Optional[float] = None
    tau_cross: Optional[float] = None
    tau_freeze1: Optional[float] = None
    tau_freeze2: Optional[float] = None

@dataclass
class RunState:
    bg: BackgroundState
    modes: Dict[float, ModeState]
    spectra_tensor: Dict[float, float]           # P_t(k) from window average
    tau_crossings: Dict[float, float]
    tau_freeze_window: Dict[float, Tuple[float, float]]

# -----------------------------
# Helper functions
# -----------------------------

def a_of_tau(Hc: float, tau: float) -> float:
    return -1.0 / (Hc * tau)

def H_conf_of_tau(tau: float) -> float:
    return -1.0 / tau

def omega2_mid(k: float, tau_mid: float) -> float:
    """ de Sitter: a''/a = 2/tau^2, evaluated at tau_mid """
    return k*k - 2.0 / (tau_mid * tau_mid)

def bd_tensor(k: float, tau: float) -> complex:
    """ v_k(tau) = exp(-i k tau) * (1 - i/(k tau)) / sqrt(2 k) """
    amp = 1.0 / math.sqrt(2.0 * max(k, 1e-30))
    phase = complex(math.cos(-k*tau), math.sin(-k*tau))
    return amp * (1.0 - 1j/(k*tau)) * phase

def centered_step_mid(v_im1: complex, v_i: complex, dt: float, omega2_mid_val: float) -> complex:
    """ Centered 2nd-order step: v_{i+1} = (2 - omega_mid^2 * dt^2) * v_i - v_{i-1} """
    return (2.0 - omega2_mid_val * dt * dt) * v_i - v_im1

# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline(params: Params) -> RunState:
    # Initial background on a constant-Delta_tau grid
    a0 = a_of_tau(params.Hc, params.tau0)
    H0 = H_conf_of_tau(params.tau0)
    bg = BackgroundState(i=0, tau=params.tau0, a=a0, H=H0, dtau=params.dtau)

    # Initialize modes (analytic BD; no rescaling)
    modes: Dict[float, ModeState] = {}
    for k in params.k_modes:
        v0 = bd_tensor(k, bg.tau)
        v_1 = bd_tensor(k, bg.tau - bg.dtau)
        modes[k] = ModeState(v_prev=v_1, v_curr=v0)

    spectra_tensor: Dict[float, float] = {}
    tau_crossings: Dict[float, float] = {}
    tau_freeze_window: Dict[float, Tuple[float, float]] = {}

    for step in range(params.n_steps):
        # Exit if all modes are frozen
        if all(m.frozen for m in modes.values()):
            break

        # Background step and local parameters
        dt = bg.dtau
        tau_mid = bg.tau + 0.5 * dt
        tau_ip1 = bg.tau + dt
        if tau_ip1 >= -1e-12:
            tau_ip1 = -1e-12
            tau_mid = (bg.tau + tau_ip1) * 0.5
        a_ip1 = a_of_tau(params.Hc, tau_ip1)
        H_ip1 = H_conf_of_tau(tau_ip1)

        # Modes
        for k, m in modes.items():
            if m.frozen:
                continue

            # Log horizon crossing: k ? H_conf
            if (k - bg.H) * (k - H_ip1) <= 0.0 and k not in tau_crossings:
                tau_crossings[k] = bg.tau

            # Midpoint omega^2 and centered step
            om2 = omega2_mid(k, tau_mid)
            v_ip1 = centered_step_mid(m.v_prev, m.v_curr, dt, om2)

            # Window condition: R1 <= H_conf/k <= R2
            ratio_next = H_ip1 / k
            if ratio_next >= params.R1:
                if m.tau_freeze1 is None:
                    m.tau_freeze1 = bg.tau  # window start
                if ratio_next <= params.R2:
                    # h = v/a; here v -> v_ip1, a -> a_ip1 (close in time)
                    h_sq = abs(v_ip1)**2 / max(a_ip1 * a_ip1, 1e-30)
                    m.window_vals.append(h_sq)
                else:
                    # Leaving the window -> finalize
                    if not m.window_vals:
                        # If empty, at least push one last sample
                        h_sq = abs(v_ip1)**2 / max(a_ip1 * a_ip1, 1e-30)
                        m.window_vals.append(h_sq)
                    mean_hsq = float(np.mean(m.window_vals))
                    Pt = (4.0 * k**3 / (math.pi**2)) * mean_hsq
                    m.frozen = True
                    m.Pt_store = Pt
                    m.tau_freeze2 = bg.tau
                    spectra_tensor[k] = Pt
                    tau_freeze_window[k] = (m.tau_freeze1, m.tau_freeze2)

            # Advance
            m.v_prev, m.v_curr = m.v_curr, v_ip1
            modes[k] = m

        # Background update
        bg = BackgroundState(i=bg.i + 1, tau=tau_ip1, a=a_ip1, H=H_ip1, dtau=dt)

    return RunState(
        bg=bg,
        modes=modes,
        spectra_tensor=spectra_tensor,
        tau_crossings=tau_crossings,
        tau_freeze_window=tau_freeze_window
    )

# -----------------------------
# Minimal demo
# -----------------------------

def main():
    p = Params(
        Hc=1e-5,
        k_modes=np.array([1e-6, 3e-6, 1e-5]),
        n_steps=600000,
        tau0=-1.0e7,
        dtau=5.0e1,
        R1=50.0,
        R2=200.0,
    )
    run = run_pipeline(p)

    Pt_th = 2.0 * p.Hc**2 / (math.pi**2)

    print(f"Final state: i={run.bg.i}, tau={run.bg.tau:.4e}, a={run.bg.a:.4e}, H_conf={run.bg.H:.4e}")
    print("\nHorizon crossings (tau_cross):")
    for k in sorted(run.tau_crossings.keys()):
        print(f"  k={k:.3e}  tau_cross~{run.tau_crossings[k]:.3e}")
    print(f"\nFreeze-out windows (tau_freeze in [tau1, tau2], H_conf/k in [{p.R1}, {p.R2}]):")
    for k in sorted(run.tau_freeze_window.keys()):
        t1, t2 = run.tau_freeze_window[k]
        print(f"  k={k:.3e}  tau_freeze~[{t1:.3e}, {t2:.3e}]")
    print("\nSpectra (tensor, dimensionless P_t with window-average):")
    for k in sorted(run.spectra_tensor.keys()):
        Pt = run.spectra_tensor[k]
        print(f"  k={k:.3e}  P_t~{Pt:.6e}   [ P_t / P_t^(th) ~ {Pt / Pt_th:.3f} ]")
    print("\nTheoretical value: P_t^(th) = 2 H_c^2 / pi^2 ~ {:.6e}".format(Pt_th))

if __name__ == "__main__":
    main()
