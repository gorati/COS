#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cos_stab_logger.py
------------------
COS-STAB compatible JSONL logger for COS-NUM runs.

Purpose:
- Enforce a *logging contract* (schema) for auditability.
- Write one valid JSON object per line to metrics.jsonl.
- Optionally write run-level metadata to run_meta.json.
- Fail fast (strict mode) if required fields are missing, ill-typed, or inconsistent.

Core idea:
- The simulator computes step-varying quantities (p_geom, p_phys, g_HW, etc.).
- The logger validates and persists them.

This module provides:
- REQUIRED_FIELDS_MINIMAL: normative minimum fields for COS-STAB audit
- REQUIRED_FIELDS_EXTENDED: recommended extended audit fields for COS-NUM
- CosStabLogger: schema-enforcing JSONL writer

Usage (extended schema example):
    from cos_stab_logger import CosStabLogger, REQUIRED_FIELDS_EXTENDED

    with CosStabLogger(
        output_path="metrics.jsonl",
        run_meta_path="run_meta.json",
        run_meta={"run_id":"run-0001","seed":2025,"dt":1.0,"T":1000,"lambda_geom":1.0,"Delta_min":1e-4},
        required_fields=REQUIRED_FIELDS_EXTENDED,
        delta_min=1e-4,
        strict=True,
    ) as logger:
        for step in range(1, T+1):
            logger.write({
                "step": step,
                "t_phys": step*dt,
                "p_geom": p_geom,
                "p_phys": p_phys,
                "g_HW": g_hw,
                "move_type": move_type,
                "bench_policy_v": bench_policy_v,
                "bench_u_id": bench_u_id,
                "bench_gamma": bench_gamma,
                "O_bench": O_bench,
                "seed": seed,
                "dt": dt,
                "lambda_geom": lambda_geom,
                # gap_event optional: auto-computed if missing
            })

Validate offline:
    python validate_metrics.py metrics.jsonl --schema extended --strict
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


# --- Normative minimum audit schema (COS-STAB core) ---
REQUIRED_FIELDS_MINIMAL: Tuple[str, ...] = (
    "step",
    "t_phys",
    "p_geom",
    "p_phys",
    "g_HW",
    "gap_event",
)

# --- Recommended extended audit schema (COS-NUM) ---
REQUIRED_FIELDS_EXTENDED: Tuple[str, ...] = REQUIRED_FIELDS_MINIMAL + (
    "move_type",
    "bench_policy_v",
    "bench_u_id",
    "bench_gamma",
    "O_bench",
    "seed",
    "dt",
    "lambda_geom",
)

# Default: be conservative; users can pass EXTENDED explicitly
DEFAULT_REQUIRED_FIELDS: Tuple[str, ...] = REQUIRED_FIELDS_MINIMAL


def _is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def compute_gap_event(g_hw: float, delta_min: float) -> int:
    """Return 1 if g_HW < Delta_min, else 0."""
    return 1 if float(g_hw) < float(delta_min) else 0


@dataclass(frozen=True)
class LoggerConfig:
    delta_min: float = 1e-4
    strict: bool = True
    required_fields: Iterable[str] = DEFAULT_REQUIRED_FIELDS
    float_round: Optional[int] = 6  # stable diffs; set None to disable


class CosStabLogger:
    """
    JSONL logger that enforces a COS-STAB-style audit schema.

    Writes:
      - metrics.jsonl: one JSON object per line
      - run_meta.json (optional): run-level metadata (not step-varying)
    """

    def __init__(
        self,
        output_path: str = "metrics.jsonl",
        *,
        delta_min: float = 1e-4,
        strict: bool = True,
        required_fields: Iterable[str] = DEFAULT_REQUIRED_FIELDS,
        float_round: Optional[int] = 6,
        run_meta_path: Optional[str] = None,
        run_meta: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ) -> None:
        self.output_path = output_path
        self.cfg = LoggerConfig(
            delta_min=float(delta_min),
            strict=bool(strict),
            required_fields=tuple(required_fields),
            float_round=float_round,
        )

        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._fh = open(output_path, mode, encoding="utf-8", newline="\n")

        self._last_step: Optional[int] = None
        self._last_t: Optional[float] = None

        if run_meta_path is not None:
            os.makedirs(os.path.dirname(run_meta_path) or ".", exist_ok=True)
            meta = dict(run_meta or {})
            meta.setdefault("created_unix", time.time())
            meta.setdefault("metrics_path", os.path.basename(output_path))
            meta.setdefault("Delta_min", float(delta_min))
            meta.setdefault("schema_required_fields", list(self.cfg.required_fields))
            with open(run_meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2, sort_keys=True, ensure_ascii=False)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "CosStabLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, record: Dict[str, Any]) -> None:
        """Validate + write a single metrics record as one JSON line."""
        rec = dict(record)

        # Compute gap_event if not provided
        if "gap_event" not in rec:
            if "g_HW" not in rec:
                raise ValueError("Missing 'g_HW'; cannot compute 'gap_event'.")
            rec["gap_event"] = compute_gap_event(rec["g_HW"], self.cfg.delta_min)

        self._validate_record(rec)

        if self.cfg.float_round is not None:
            rec = self._round_floats(rec, self.cfg.float_round)

        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

        self._last_step = int(rec["step"])
        self._last_t = float(rec["t_phys"])

    def _round_floats(self, obj: Any, nd: int) -> Any:
        if isinstance(obj, float):
            return round(obj, nd)
        if isinstance(obj, dict):
            return {k: self._round_floats(v, nd) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._round_floats(v, nd) for v in obj]
        return obj

    def _validate_record(self, rec: Dict[str, Any]) -> None:
        # Required fields check
        missing = [k for k in self.cfg.required_fields if k not in rec]
        if missing:
            raise ValueError(f"Missing required fields in metrics record: {missing}")

        # Basic types/ranges
        if not isinstance(rec["step"], int):
            raise TypeError(f"'step' must be int, got {type(rec['step']).__name__}")

        if not _is_finite_number(rec["t_phys"]):
            raise TypeError("'t_phys' must be a finite number.")

        for pkey in ("p_geom", "p_phys"):
            if not _is_finite_number(rec[pkey]):
                raise TypeError(f"'{pkey}' must be a finite number in [0,1].")
            pv = float(rec[pkey])
            if pv < 0.0 or pv > 1.0:
                raise ValueError(f"'{pkey}' out of range [0,1]: {pv}")

        if not _is_finite_number(rec["g_HW"]):
            raise TypeError("'g_HW' must be a finite number (fermion gap proxy).")

        if not isinstance(rec["gap_event"], int) or rec["gap_event"] not in (0, 1):
            raise TypeError("'gap_event' must be int in {0,1}.")

        # gap_event consistency (strict mode)
        if self.cfg.strict:
            expected_gap_event = compute_gap_event(rec["g_HW"], self.cfg.delta_min)
            if int(rec["gap_event"]) != expected_gap_event:
                raise ValueError(
                    f"Inconsistent gap_event={rec['gap_event']} for g_HW={rec['g_HW']} "
                    f"with Delta_min={self.cfg.delta_min} (expected {expected_gap_event})."
                )

        # Extended-field validations (only if present/required)
        if "dt" in rec:
            if not _is_finite_number(rec["dt"]) or float(rec["dt"]) <= 0.0:
                raise TypeError("'dt' must be a finite positive number.")
        if "seed" in rec:
            if not _is_finite_number(rec["seed"]):
                raise TypeError("'seed' must be numeric.")
        if "lambda_geom" in rec:
            if not _is_finite_number(rec["lambda_geom"]) or float(rec["lambda_geom"]) < 0.0:
                raise TypeError("'lambda_geom' must be a finite non-negative number.")

        if "bench_policy_v" in rec:
            if not isinstance(rec["bench_policy_v"], str) or not rec["bench_policy_v"]:
                raise TypeError("'bench_policy_v' must be a non-empty string.")
        if "bench_gamma" in rec:
            if not isinstance(rec["bench_gamma"], str) or not rec["bench_gamma"]:
                raise TypeError("'bench_gamma' must be a non-empty string.")
        if "bench_u_id" in rec:
            if not _is_finite_number(rec["bench_u_id"]):
                raise TypeError("'bench_u_id' must be numeric (int preferred).")
        if "O_bench" in rec:
            if not isinstance(rec["O_bench"], (list, tuple)):
                raise TypeError("'O_bench' must be a list/tuple of benchmark observables.")
        if "move_type" in rec:
            if not isinstance(rec["move_type"], str) or not rec["move_type"]:
                raise TypeError("'move_type' must be a non-empty string.")

        # Monotonicity checks (strict mode)
        if self.cfg.strict:
            if self._last_step is not None and rec["step"] != self._last_step + 1:
                raise ValueError(f"Non-consecutive step index: got {rec['step']} after {self._last_step}.")
            if self._last_t is not None and float(rec["t_phys"]) < float(self._last_t):
                raise ValueError(f"Non-monotone t_phys: got {rec['t_phys']} after {self._last_t}.")


if __name__ == "__main__":
    # Self-test (dummy values) — demonstrates that the logger works standalone.
    seed = 2025
    dt = 1.0
    lambda_geom = 1.0
    T = 3

    bench_policy_v = "v1.0-selftest"
    bench_u_id = 42
    bench_gamma = "init_hash_0000"

    with CosStabLogger(
        output_path="metrics.jsonl",
        run_meta_path="run_meta.json",
        run_meta={
            "run_id": "cos-stab-logger-selftest",
            "seed": seed,
            "dt": dt,
            "T": T,
            "lambda_geom": lambda_geom,
            "Delta_min": 1e-4,
        },
        required_fields=REQUIRED_FIELDS_EXTENDED,
        delta_min=1e-4,
        strict=True,
    ) as logger:
        g_hw = 0.1
        for step in range(1, T + 1):
            p_geom = 1.0
            p_phys = 1.0
            move_type = "selftest"
            O_bench = [100, g_hw]

            logger.write({
                "step": step,
                "t_phys": step * dt,
                "p_geom": p_geom,
                "p_phys": p_phys,
                "g_HW": g_hw,
                "move_type": move_type,
                "bench_policy_v": bench_policy_v,
                "bench_u_id": bench_u_id,
                "bench_gamma": bench_gamma,
                "O_bench": O_bench,
                "seed": seed,
                "dt": dt,
                "lambda_geom": lambda_geom,
                # gap_event optional: auto-computed if missing
            })

    print("[OK] wrote metrics.jsonl and run_meta.json (selftest)")
