#!/usr/bin/env python3
"""Print a compact summary of bundled Bilby result JSON files."""
from pathlib import Path
import json, math
import numpy as np

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "results" / "current").glob("*/*_result.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    post = d.get("posterior", {}).get("content", {})
    print(f"\n{d.get('label')}  logZ={d.get('log_evidence'):.6f} +/- {d.get('log_evidence_err'):.6f}")
    for name in ["Omega0", "Omega_astro0", "A_disc", "omega", "phi"]:
        vals = post.get(name)
        if vals:
            a = np.asarray(vals, dtype=float)
            print(f"  {name:14s} median={np.quantile(a,0.5):.3e}  90%=[{np.quantile(a,0.05):.3e}, {np.quantile(a,0.95):.3e}]")
