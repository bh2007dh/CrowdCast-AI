@echo off
cd /d "%~dp0"
echo ============================================================
echo   CrowdCast AI - Automatic Setup ^& Server Launcher
echo ============================================================

REM Detect available Python command (python or py launcher)
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
) else (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3
    )
)

if "%PYTHON_CMD%"=="" (
    echo ERROR: Python is not installed or not added to PATH on this laptop.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: Check the box "Add Python to PATH" during installation!
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment (.venv) using %PYTHON_CMD%...
    %PYTHON_CMD% -m venv .venv
    echo [2/3] Installing required packages from requirements.txt...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\pip.exe" install -r requirements.txt
    echo [3/3] Building ML model for this laptop...
    ".venv\Scripts\python.exe" ml\train_model.py
)

echo Starting CrowdCast AI server at http://127.0.0.1:5000 ...
".venv\Scripts\python.exe" app.py
pause
