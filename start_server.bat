@echo off
cd /d "%~dp0"
echo ============================================================
echo   CrowdCast AI - One-Click Setup ^& Server Launcher
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment (.venv) for this laptop...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH on this laptop.
        echo Please install Python 3.10+ from https://www.python.org/downloads/ and check "Add Python to PATH".
        pause
        exit /b 1
    )
    echo [2/3] Installing required packages from requirements.txt...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\pip.exe" install -r requirements.txt
    echo [3/3] Training ML model for this laptop's environment...
    ".venv\Scripts\python.exe" ml\train_model.py
)

echo Starting CrowdCast AI server at http://127.0.0.1:5000 ...
".venv\Scripts\python.exe" app.py
pause
