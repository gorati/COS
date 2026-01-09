#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cos_core_sim_combined.py
------------------------
COS-CORE Simulation + COS-STAB audit logger (COMBINED / reference integration)

What this script does:
- Runs a simple COS-CORE toy evolution (graph/geometry + fermion-gap proxy)
- Produces COS-STAB-auditable artifacts:
    1) run_meta.json   (run-level metadata)
    2) metrics.jsonl   (per-step audit log; one valid JSON object per line)
- Enforces the audit schema at write-time via CosStabLogger:
  if required fields are missing or inconsistent -> the run fails immediately.

This is the recommended pattern for COS-NUM:
- keep your physics/simulator code (COSState) separate from the audit contract
- use CosStabLogger to enforce COS-STAB logging requirements

Dependencies:
- Python 3.9+
- numpy
- cos_stab_logger.py in the same folder (provided separately)

Validate outputs:
    python validate_metrics.py metrics.jsonl --schema extended --strict
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from cos_stab_logger import CosStabLogger

# --- COS-STAB Specifikáció szerinti konstansok ---
DELTA_MIN = 1e-4  # Minimális megengedett spektrális rés
DT_PLANCK = 1.0   # Időlépés Planck-egységben

REQUIRED_FIELDS_MINIMAL = (
    "step", "t_phys", "p_geom", "p_phys", "g_HW", "gap_event",
)

REQUIRED_FIELDS_EXTENDED = REQUIRED_FIELDS_MINIMAL + (
    "move_type",
    "bench_policy_v", "bench_u_id", "bench_gamma", "O_bench",
    "seed", "dt", "lambda_geom",
)

@dataclass
class COSState:
    """
    Reprezentálja a rendszer pillanatnyi állapotát (gráf + mezők).

    Fizikai definíciók (audit-szinten):
      - g_HW (Gap of H_W): A Hermitikus Wilson-Dirac operátor (H_W = gamma5 * D_W)
        legkisebb abszolút sajátértéke: min(|lambda|).
        Ez védi a rendszert a fermion-duplikációtól (doubling problem).
      - p_geom: A geometriai projektor (Pi_geom) várható értéke.
    """
    seed: int
    lambda_geom: float

    def __post_init__(self) -> None:
        # Reprodukálhatóság
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Futási paraméterek
        self.dt: float = DT_PLANCK

        # Állapot
        self.step_idx: int = 0
        self.t_phys: float = 0.0
        self.num_vertices: int = 100
        self.is_geometric: bool = True

        # Kezdeti gap (biztonságos)
        self.gap_hw: float = 0.1

        # Benchmark horgony (COS-STAB D.3 jellegű)
        self.bench_u_id: int = 42
        self.bench_gamma_hash: str = "init_hash_0000"

    def evolve_step(self) -> None:
        """Egy elemi időlépés (Trotter-step) végrehajtása."""
        self.step_idx += 1
        self.t_phys += self.dt

        # --- 1. Dinamika (Pachner moves & Geometry) ---
        stability_prob = 0.99 if self.lambda_geom > 0 else 0.90
        self.is_geometric = (random.random() < stability_prob)

        # --- 2. Fermion Gap Dinamika (g_HW) ---
        drift = np.random.normal(0, 0.005)
        self.gap_hw += drift

        # Gap protection: a demonstrációban csak levágjuk a negatív tartományt
        if self.gap_hw < 0.0:
            self.gap_hw = 1e-5

    def get_metrics(self) -> Dict[str, Any]:
        """Összeállítja a COS-STAB (v10) 'extended' sémához illeszkedő rekordot."""
        gap_event = 1 if self.gap_hw < DELTA_MIN else 0
        p_geom = 1.0 if self.is_geometric else 0.0
        o_bench = [self.num_vertices, float(self.gap_hw)]

        record: Dict[str, Any] = {
            # --- run context (kept in every line for audit robustness) ---
            "seed": int(self.seed),
            "dt": float(self.dt),
            "lambda_geom": float(self.lambda_geom),

            # --- dynamics ---
            "step": int(self.step_idx),
            "t_phys": float(self.t_phys),
            "p_geom": float(p_geom),
            "p_phys": 1.0,

            # --- bench & gap ---
            "bench_policy_v": "v1.1-core+logger",
            "bench_u_id": int(self.bench_u_id),
            "bench_gamma": str(self.bench_gamma_hash),
            "O_bench": o_bench,
            "g_HW": float(self.gap_hw),
            "gap_event": int(gap_event),

            # --- move bookkeeping (toy) ---
            "move_type": "4-1" if random.random() > 0.5 else "2-3",
        }
        return record


def main() -> None:
    ap = argparse.ArgumentParser(description="COS-CORE + COS-STAB audit logger (combined reference)")
    ap.add_argument("--steps", type=int, default=1000, help="Szimulációs lépések száma")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed")
    ap.add_argument("--lambda-geom", type=float, default=1.0, help="Geometriai csatolás (demó paraméter)")
    ap.add_argument("--output", type=str, default="metrics.jsonl", help="Kimeneti audit-log (JSONL)")
    ap.add_argument("--run-meta", type=str, default="run_meta.json", help="Futás metaadatok (JSON)")
    ap.add_argument("--stop-on-gap-event", action="store_true", help="Állj meg az első gap_event=1 esetén")
    args = ap.parse_args()

    sim = COSState(seed=args.seed, lambda_geom=args.lambda_geom)

    run_id = f"cos-core-{int(time.time())}"
    run_meta = {
        "run_id": run_id,
        "timestamp_unix": time.time(),
        "seed": args.seed,
        "dt": DT_PLANCK,
        "T": args.steps,
        "lambda_geom": args.lambda_geom,
        "Delta_min": DELTA_MIN,
        "policy": "COS-STAB-extended",
        "output": args.output,
    }

    print(f"[*] Starting COS-CORE (seed={args.seed}, lambda_geom={args.lambda_geom})")
    print(f"[*] Writing: {args.output}")
    print(f"[*] Run meta: {args.run_meta} (run_id={run_id})")

    with CosStabLogger(
        output_path=args.output,
        run_meta_path=args.run_meta,
        run_meta=run_meta,
        required_fields=REQUIRED_FIELDS_EXTENDED,
        delta_min=DELTA_MIN,
        strict=True,
    ) as logger:
        for _ in range(args.steps):
            sim.evolve_step()
            rec = sim.get_metrics()
            logger.write(rec)

            if args.stop_on_gap_event and rec["gap_event"] == 1:
                print("[WARN] gap_event=1 encountered; stopping early per policy.")
                break

    print(f"[OK] Simulation finished. Last g_HW: {sim.gap_hw:.6f}")
    print("[OK] Next: validate with:")
    print(f"     python validate_metrics.py {args.output} --schema extended --strict")


if __name__ == "__main__":
    main()
