@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --cache-dir .cache/pip -r requirements-desktop.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts/setup_models.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" launcher.py --download-model
exit /b %errorlevel%
