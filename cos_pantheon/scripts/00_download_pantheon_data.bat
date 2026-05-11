@echo off
setlocal EnableExtensions

rem Resolve repository root from this script location.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

pushd "%REPO_ROOT%" >nul

if not exist "data\pantheon" mkdir "data\pantheon"

curl -L -o "data\pantheon\Pantheon+SH0ES.dat" "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%%2BSH0ES.dat"
curl -L -o "data\pantheon\Pantheon+SH0ES_STAT+SYS.cov" "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%%2BSH0ES_STAT%%2BSYS.cov"

popd >nul
endlocal
