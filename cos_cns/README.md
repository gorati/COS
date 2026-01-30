# COS-CNS: Causality, Signal-Locality (No-Signaling), and Finite-Speed Influence in Non-Unitary Discrete Spacetime Dynamics.

## Files

- `cos_cns_pipeline_trajectories_graph.py`  
  Main reproducible pipeline (trajectory-based / open quantum system style).

- `cos_cns_pipeline_trajectories_graph.bat`  
  Windows runner for the **final** (publication) configuration.

Optional (recommended):
- `cos_cns_autotune_multi.bat`  
  Windows runner for calibration/diagnostics (e.g. multi-topology and/or autotune).

## Quick start (Windows)

### Final run (publication configuration)
Double-click:
- `cos_cns_pipeline_trajectories_graph.bat`

Or run in CMD from this folder:
```bat
python cos_cns_pipeline_trajectories_graph.py --topology er --publish_topology er --N 15 --steps 24 --trials 60 --ntraj 1200 --gamma 0.05 --seed 42 --edge_gate sqrt_swap --sched_steps 24 --sched_B 7
