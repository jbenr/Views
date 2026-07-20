@echo off
rem tens_10s30s research funnel: --predict -> --exit -> --sweep
rem stops at the first step that fails
cd /d "%~dp0"

call mamba run -n 2s10s python tens_10s30s.py --predict
if errorlevel 1 goto :fail

call mamba run -n 2s10s python tens_10s30s.py --exit
if errorlevel 1 goto :fail

call mamba run -n 2s10s python tens_10s30s.py --sweep
if errorlevel 1 goto :fail

echo.
echo funnel complete.
exit /b 0

:fail
echo.
echo funnel FAILED at the step above.
exit /b 1
