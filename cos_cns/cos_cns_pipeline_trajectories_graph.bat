@echo off
setlocal
cd /d "%~dp0"

REM COS-CNS trajectories pipeline
REM - publish_topology = chain (avoid "N=15 ER" optics)
REM - eps_sched tightened: treat scheduling invariance as exact; any residual is numerical/approximation error
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 40 --ntraj 600 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --sched_steps 24 --sched_B 9 ^
  --eps_sched 1e-10 --eps_ns 1e-3

endlocal