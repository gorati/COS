@echo off
setlocal EnableExtensions

rem Resolve repository root from this script location.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

pushd "%REPO_ROOT%" >nul

if not exist "results\current" mkdir "results\current"
if "%N_JOBS%"=="" set "N_JOBS=4"
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"

"%PYTHON_BIN%" "src\cos_pantheon\axis_scan.py" ^
  --data "data\pantheon\Pantheon+SH0ES.dat" ^
  --cov "data\pantheon\Pantheon+SH0ES_STAT+SYS.cov" ^
  --out "results\current\pantheon_axis_scan_q0_fixed.json" ^
  --zmin 0.01 ^
  --zmax 0.10 ^
  --cos-coords gal ^
  --cos-lon 0 ^
  --cos-lat 90 ^
  --statistic abs_delta_q0 ^
  --run-null ^
  --null-mode sky-scramble ^
  --n-random-axes 1000 ^
  --n-null 500 ^
  --axis-seed 12345 ^
  --null-seed 24680 ^
  --progress-every 10 ^
  --n-jobs %N_JOBS%

popd >nul
endlocal
