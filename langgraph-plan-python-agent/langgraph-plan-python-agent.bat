@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"

if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" -m langchain_agent_cli %*
  exit /b %errorlevel%
)

python -m langchain_agent_cli %*
exit /b %errorlevel%
