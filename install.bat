@echo off
rem Thin wrapper. All the logic lives in install.py.
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo python not found. Install Python 3.10+ from python.org
    echo and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)
python install.py %*
pause
