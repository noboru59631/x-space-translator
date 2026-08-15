@echo off
setlocal
cd /d "%~dp0"
title X Space Translator - Setup

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3.11"
) else (
  set "PYTHON_CMD=python"
)

%PYTHON_CMD% --version
if errorlevel 1 goto :python_error

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating Python virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo [2/4] Updating pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [3/4] Installing packages. This can take several minutes...
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

if not exist ".env" copy ".env.example" ".env" >nul
echo [4/4] Checking FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo WARNING: FFmpeg was not found.
  echo Install it with: winget install Gyan.FFmpeg
  echo Then open a new terminal and run setup.bat again.
) else (
  echo FFmpeg found.
)

echo.
echo Setup completed. Double-click start.bat to launch.
pause
exit /b 0

:python_error
echo Python 3.11 was not found. Install it from https://www.python.org/downloads/
pause
exit /b 1

:failed
echo Setup failed. Review the message above, then run setup.bat again.
pause
exit /b 1
