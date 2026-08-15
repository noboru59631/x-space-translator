@echo off
setlocal
cd /d "%~dp0"
title X Space Translator

if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment was not found. Run setup.bat first.
  pause
  exit /b 1
)

if not exist ".env" copy ".env.example" ".env" >nul
echo X Space Translator: http://127.0.0.1:8765
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765'"
call .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765

echo.
echo The application stopped.
pause
