@echo off
setlocal
cd /d "%~dp0"

REM (Optional) deterministic hashing if your script ever uses Python hash() for seeds:
REM set PYTHONHASHSEED=0

REM Multi-topology + autotune (calibration / diagnostics)
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er --publish_topology er ^
  --N 15 --steps 24 --trials 60 --ntraj 1200 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --autotune_nc3

endlocal
