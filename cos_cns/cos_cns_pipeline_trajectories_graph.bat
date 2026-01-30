@echo off
setlocal

REM Ensure we run from this script's directory
cd /d "%~dp0"

REM Run the COS-CNS trajectories pipeline (ER, final run parameters)
python cos_cns_pipeline_trajectories_graph.py --topology er --publish_topology er --N 15 --steps 24 --trials 60 --ntraj 1200 --gamma 0.05 --seed 42 --edge_gate sqrt_swap --sched_steps 24 --sched_B 7

endlocal
