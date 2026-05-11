@echo off
setlocal
cd /d "%~dp0"

REM COS-CNS multi-topology + autotune (calibration / diagnostics)
REM - Autotune runs only on publish_topology (per script logic)
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 60 --ntraj 1200 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --eps_sched 1e-10 --eps_ns 1e-3 ^
  --autotune_nc3 ^
  --autotune_trials 30 ^
  --autotune_ntraj 600 ^
  --autotune_steps 4,6,8,10,12,16,20,24

endlocal