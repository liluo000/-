@echo off
cd /d "%~dp0"
echo This will delete only the local Python virtual environment.
choice /M "Continue"
if errorlevel 2 exit /b 0
if exist ".venv" rmdir /S /Q ".venv"
echo Done. Run START_HERE.bat again.
pause
