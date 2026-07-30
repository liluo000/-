@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Qwen Project Diagnostics

echo ========================================
echo Qwen Project Diagnostics
echo ========================================
echo Project folder: %CD%
echo.

echo [Files]
if exist ".env" (echo .env: FOUND) else (echo .env: MISSING)
if exist "requirements.txt" (echo requirements.txt: FOUND) else (echo requirements.txt: MISSING)
if exist "app.py" (echo app.py: FOUND) else (echo app.py: MISSING)
echo.

echo [Python launcher]
where py 2>nul
py --list 2>nul
echo.

echo [python command]
where python 2>nul
python --version 2>nul
echo.

echo [Virtual environment]
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version
    ".venv\Scripts\python.exe" -m pip --version
) else (
    echo .venv not created yet.
)
echo.

echo Copy the text in this window if you need help.
pause
