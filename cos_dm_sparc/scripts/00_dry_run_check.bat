@echo off
setlocal
cd /d "%~dp0\.."
python src\cos_dm_sparc\fit.py data\sparc\Rotmod_LTG data\sparc\SPARC_Lelli2016c.mrt --pattern "**/*.dat" --dry-run
endlocal
