@echo off
cd /d "%~dp0"
echo This window will stay open after the launcher exits.
echo.
call run_windows.bat
echo.
echo Launcher exit code: %ERRORLEVEL%
pause
