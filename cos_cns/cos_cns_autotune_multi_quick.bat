@echo off
setlocal
cd /d "%~dp0"

REM COS-CNS autotune / diagnostics "quick"
REM - Multiple topologies, publish chain
REM - Uses --autotune_nc3 to search a schedule-dependent (non-confluent) negative control
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 20 --ntraj 300 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --eps_sched 1e-10 --eps_ns 1e-3 ^
  --autotune_nc3

endlocal