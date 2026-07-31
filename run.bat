@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
    echo Run install.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m lmcluster %1
pause
