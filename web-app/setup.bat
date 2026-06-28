@echo off
REM Start AI Watermark Web App (Windows)

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
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed. Please install Python 3.10+ from https://www.python.org/
    exit /b 1
)

REM Install frontend dependencies
echo.
echo [1/4] Installing frontend dependencies...
cd frontend
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    exit /b 1
)

REM Install backend dependencies
echo.
echo [2/4] Installing backend dependencies...
cd ..\backend
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install backend dependencies
    exit /b 1
)

REM Install main package
echo.
echo [3/4] Installing main watermark package...
pip install -e ..\..
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not install main package. Make sure you have installed it separately.
)

REM Create environment file if it doesn't exist
if not exist "..\frontend\.env" (
    echo.
    echo [4/4] Creating .env file...
    copy ..\..env.example ..\frontend\.env
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
echo   cd backend
echo   python main.py
echo.
echo Terminal 2 - Frontend:
echo   cd frontend
echo   npm start
echo.
echo Then open http://localhost:3000 in your browser
echo.
pause
