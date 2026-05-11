@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Optional: isolate this run so we never pick up an older runs\*_chain by accident
if exist runs (
  if not exist runs_prep mkdir runs_prep
  for /f "delims=" %%D in ('dir /b /ad "runs\*" 2^>nul') do move "runs\%%D" "runs_prep\%%D" >nul
)

REM 1) AUTOTUNE (quick) — run only on the publish topology (chain) to keep it fast
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 20 --ntraj 300 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --eps_sched 1e-10 --eps_ns 1e-3 ^
  --autotune_nc3

REM 2) Locate the most recent chain run directory (safe now because we staged old runs away)
set "RUN_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "runs\*_chain" 2^>nul') do (
  set "RUN_DIR=%%D"
  goto :got_run
)

:got_run
if "%RUN_DIR%"=="" (
  echo ERROR: Could not find a fresh runs\*_chain directory.
  exit /b 1
)

echo Using autotune run: runs\%RUN_DIR%

REM 3) Read steps_sched and B_sched from the run's config.json
for /f "tokens=1,2 delims=," %%a in ('
  python -c "import json; d=json.load(open(r'runs/%RUN_DIR%/config.json','r',encoding='utf-8')); sp=d['scheduling_params']['steps_sched']; bs=d['scheduling_params']['B_sched']; print(f'{sp},{bs}')"
') do (
  set "SCHED_STEPS=%%a"
  set "SCHED_B=%%b"
)

echo Autotune picked: sched_steps=%SCHED_STEPS%  sched_B=%SCHED_B%

REM 4) PAPER RUN — fixed scheduling parameters (no autotune here)
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 60 --ntraj 1200 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --sched_steps %SCHED_STEPS% --sched_B %SCHED_B% ^
  --eps_sched 1e-10 --eps_ns 1e-3

endlocal