#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_metrics.py
-------------------
Offline validator for COS-STAB-style `metrics.jsonl` audit logs.

Default behaviour follows the COS-STAB *minimal* contract:
- step, t_phys, p_geom, p_phys, g_HW, gap_event

You can enforce a stricter schema with --schema extended.

Usage:
  python validate_metrics.py metrics.jsonl --delta-min 1e-4 --strict
  python validate_metrics.py metrics.jsonl --schema extended --require-no-gap-events
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple


REQUIRED_MINIMAL = ["step", "t_phys", "p_geom", "p_phys", "g_HW", "gap_event"]

REQUIRED_EXTENDED = REQUIRED_MINIMAL + [
    "move_type",
    "bench_policy_v",
    "bench_u_id",
    "bench_gamma",
    "O_bench",
    "seed",
    "dt",
    "lambda_geom",
]


def is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def compute_gap_event(g_hw: float, delta_min: float) -> int:
    return 1 if float(g_hw) < float(delta_min) else 0


def validate_record(rec: Dict[str, Any], *, delta_min: float, strict: bool, required: List[str]) -> List[str]:
    errs: List[str] = []

    for k in required:
        if k not in rec:
            errs.append(f"missing field: {k}")
    if errs:
        return errs

    # Core
    if not isinstance(rec["step"], int):
        errs.append(f"step must be int, got {type(rec['step']).__name__}")
    if not is_finite_number(rec["t_phys"]):
        errs.append("t_phys must be finite number")

    for pkey in ("p_geom", "p_phys"):
        if not is_finite_number(rec[pkey]):
            errs.append(f"{pkey} must be finite number in [0,1]")
        else:
            pv = float(rec[pkey])
            if pv < 0.0 or pv > 1.0:
                errs.append(f"{pkey} out of range [0,1]: {pv}")

    if not is_finite_number(rec["g_HW"]):
        errs.append("g_HW must be finite number")
    if not isinstance(rec["gap_event"], int) or rec["gap_event"] not in (0, 1):
        errs.append("gap_event must be int in {0,1}")

    if strict:
        expected = compute_gap_event(rec["g_HW"], delta_min)
        if int(rec["gap_event"]) != expected:
            errs.append(
                f"gap_event inconsistent (gap_event={rec['gap_event']}, g_HW={rec['g_HW']}, "
                f"Delta_min={delta_min}, expected={expected})"
            )

    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate COS-STAB metrics.jsonl")
    ap.add_argument("metrics_path", help="Path to metrics.jsonl")
    ap.add_argument("--delta-min", type=float, default=1e-4, help="Delta_min threshold for gap_event consistency")
    ap.add_argument("--strict", action="store_true", help="Enable strict checks (gap_event consistency + sequencing)")
    ap.add_argument("--schema", choices=["minimal", "extended"], default="minimal", help="Which required schema to enforce")
    ap.add_argument("--require-no-gap-events", action="store_true", help="Fail if any gap_event==1 is present")
    ap.add_argument("--min-p-geom", type=float, default=None, help="Optional: fail if min(p_geom) < threshold")
    ap.add_argument("--max-lines", type=int, default=None, help="Optional: validate only first N lines (debug)")
    args = ap.parse_args()

    required = REQUIRED_MINIMAL if args.schema == "minimal" else REQUIRED_EXTENDED

    n = 0
    n_gap = 0
    min_gap = None
    min_pgeom = None
    last_step = None
    last_t = None

    errors: List[Tuple[int, str]] = []

    with open(args.metrics_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if args.max_lines is not None and n >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                errors.append((line_idx, f"invalid JSON: {e}"))
                continue

            if not isinstance(rec, dict):
                errors.append((line_idx, "record is not a JSON object"))
                continue

            rec_errs = validate_record(rec, delta_min=args.delta_min, strict=args.strict, required=required)
            if rec_errs:
                for e in rec_errs:
                    errors.append((line_idx, e))
                continue

            # Optional sequencing checks (strict)
            if args.strict:
                if last_step is not None and rec["step"] != last_step + 1:
                    errors.append((line_idx, f"non-consecutive step: got {rec['step']} after {last_step}"))
                if last_t is not None and float(rec["t_phys"]) < float(last_t):
                    errors.append((line_idx, f"non-monotone t_phys: got {rec['t_phys']} after {last_t}"))
            last_step = rec["step"]
            last_t = float(rec["t_phys"])

            n += 1
            g = float(rec["g_HW"])
            pg = float(rec["p_geom"])
            min_gap = g if (min_gap is None or g < min_gap) else min_gap
            min_pgeom = pg if (min_pgeom is None or pg < min_pgeom) else min_pgeom
            if int(rec["gap_event"]) == 1:
                n_gap += 1

    if args.require_no_gap_events and n_gap > 0:
        errors.append((0, f"gap_event present: count={n_gap} (require-no-gap-events)"))

    if args.min_p_geom is not None and min_pgeom is not None and min_pgeom < float(args.min_p_geom):
        errors.append((0, f"min(p_geom)={min_pgeom:.6g} < {args.min_p_geom}"))

    if errors:
        print("[FAIL] metrics validation failed.")
        print(f"  schema: {args.schema}")
        print(f"  records checked: {n}")
        if min_gap is not None:
            print(f"  min g_HW: {min_gap:.6g}")
        if min_pgeom is not None:
            print(f"  min p_geom: {min_pgeom:.6g}")
        print(f"  gap_event count: {n_gap}")
        print("  errors (first 50):")
        for i, (ln, msg) in enumerate(errors[:50], start=1):
            prefix = f"line {ln}" if ln > 0 else "global"
            print(f"    {i:02d}. {prefix}: {msg}")
        return 1

    print("[OK] metrics validation passed.")
    print(f"  schema: {args.schema}")
    print(f"  records checked: {n}")
    if min_gap is not None:
        print(f"  min g_HW: {min_gap:.6g}")
    if min_pgeom is not None:
        print(f"  min p_geom: {min_pgeom:.6g}")
    print(f"  gap_event count: {n_gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
