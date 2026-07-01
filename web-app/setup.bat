@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Start AI Watermark Web App (Windows)

set "SCRIPT_DIR=%~dp0"
set "WEBAPP_DIR=%SCRIPT_DIR%"
set "PROJECT_DIR=%SCRIPT_DIR%.."

REM Prefer a project virtual environment if one exists
set "PYTHON_CMD=python"
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_DIR%\.venv\Scripts\python.exe"
) else if exist "%PROJECT_DIR%\..\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_DIR%\..\.venv\Scripts\python.exe"
)

echo.
echo ================================
echo AI Watermark Web Application
echo ================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/
    exit /b 1
)

REM Check if Python is installed
"%PYTHON_CMD%" --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.10+ is not available. Please install Python from https://www.python.org/
    exit /b 1
)

echo Using Python: %PYTHON_CMD%

REM Install frontend dependencies
echo.
echo [1/5] Installing frontend dependencies...
pushd "%WEBAPP_DIR%frontend"
if exist package-lock.json (
    call npm ci
    if %ERRORLEVEL% NEQ 0 (
        echo npm ci failed, retrying with npm install...
        call npm install
    )
) else (
    call npm install
)
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    exit /b 1
)
popd

REM Install backend dependencies
echo.
echo [2/5] Installing backend dependencies...
"%PYTHON_CMD%" -m pip install -r "%WEBAPP_DIR%backend\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install backend dependencies
    exit /b 1
)

REM Install project (watermark core) dependencies
echo.
echo [3/5] Installing watermark core dependencies...
"%PYTHON_CMD%" -m pip install -r "%PROJECT_DIR%\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install watermark core dependencies
    exit /b 1
)

REM Install main package
echo.
echo [4/5] Installing main watermark package...
"%PYTHON_CMD%" -m pip install -e "%PROJECT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not install main package. Make sure you have installed it separately.
)

REM Create environment file if it doesn't exist
if not exist "%WEBAPP_DIR%frontend\.env" (
    echo.
    echo [5/5] Creating .env file...
    copy "%WEBAPP_DIR%.env.example" "%WEBAPP_DIR%frontend\.env" >nul
    echo.
    echo Created .env file. Update REACT_APP_API_URL if needed.
)

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo To start the application:
echo.
echo Terminal 1 - Backend:
echo   cd /d "%WEBAPP_DIR%backend"
echo   %PYTHON_CMD% main.py
echo.
echo Terminal 2 - Frontend:
echo   cd /d "%WEBAPP_DIR%frontend"
echo   npm start
echo.
echo Then open http://localhost:3000 in your browser
echo.
pause
