@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Salt Price Matcher - Qwen API

set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "LOG_FILE=%~dp0startup.log"
>"%LOG_FILE%" echo [%date% %time%] Startup started.

echo ========================================
echo  12-SKU Product and Price Matcher
echo  Qwen3.7-Flash API Edition
echo ========================================
echo.

echo [1/4] Preparing configuration ...
if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
    ) else (
        >".env" echo DASHSCOPE_API_KEY=
    )
)
echo Configuration file is ready.
echo.

echo [2/4] Finding Python 3.10 or newer ...
set "PYTHON_CMD="
py -3.13 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PYTHON_CMD=py -3.13"
if not defined PYTHON_CMD py -3.12 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD py -3.11 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD py -3.10 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PYTHON_CMD=py -3.10"
if not defined PYTHON_CMD python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    echo ERROR: Python 3.10 or newer was not found.
    echo Install 64-bit Python from python.org and select Add Python to PATH.
    echo.
    echo Log: %LOG_FILE%
    pause
    exit /b 30
)

echo Using: %PYTHON_CMD%
%PYTHON_CMD% --version
%PYTHON_CMD% --version >>"%LOG_FILE%" 2>&1

echo.
echo [3/4] Preparing Python environment ...
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Creating .venv. This can take a minute ...
    %PYTHON_CMD% -m venv ".venv" >>"%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        echo Open startup.log for details.
        pause
        exit /b 31
    )
)

if not exist "%VENV_PY%" (
    echo ERROR: .venv Python is missing.
    pause
    exit /b 32
)

echo Installing or checking dependencies.
echo The first run can take 3 to 15 minutes depending on your network.
"%VENV_PY%" -m pip install --prefer-binary --progress-bar off -r "requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    echo Check your network, proxy, antivirus, and startup.log.
    pause
    exit /b 40
)

echo.
echo [4/4] Starting the web application ...
echo Open this address in your browser: http://127.0.0.1:8501
echo Keep this window open while using the program.
echo.
start "" http://127.0.0.1:8501
"%VENV_PY%" -m streamlit run "app.py" --server.address 127.0.0.1 --server.port 8501
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo ERROR: The web application stopped with code %APP_EXIT%.
    echo Open startup.log or run RUN_DEBUG.bat.
    pause
    exit /b %APP_EXIT%
)

exit /b 0
