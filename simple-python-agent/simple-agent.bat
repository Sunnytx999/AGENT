@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

echo Checking python from PATH...
where python
python -c "import sys; print(sys.executable)"
echo.

python --version >nul 2>nul
if %errorlevel%==0 (
  echo Using python:
  python -c "import sys; print(sys.executable)"
  echo.
  python -m simple_agent_cli %*
  exit /b %errorlevel%
)

echo Checking py launcher...
where py
py -3 -c "import sys; print(sys.executable)"
echo.

py -3 --version >nul 2>nul
if %errorlevel%==0 (
  echo Using py -3:
  py -3 -c "import sys; print(sys.executable)"
  echo.
  py -3 -m simple_agent_cli %*
  exit /b %errorlevel%
)

echo Python was not found or cannot run. Please install Python and make sure python or py is available in PATH.
exit /b 1
