@echo off
cd /d "%~dp0"
echo Starting CrowdCast AI at http://127.0.0.1:5000 ...
".venv\Scripts\python.exe" app.py
pause
