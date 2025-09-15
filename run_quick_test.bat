@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_quick_test.ps1" -TimeoutSeconds 120 -MaxEpochs 1 -MaxBatches 5
set ERR=%ERRORLEVEL%
if %ERR%==124 (
  echo [run_quick_test] TIMED_OUT_120S_TERMINATED
) else (
  echo [run_quick_test] Exit code %ERR%.
)
exit /b %ERR%
