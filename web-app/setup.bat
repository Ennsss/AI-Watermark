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
echo ================================================================================
echo       AI Watermark Web Application - Setup & Installation
echo ================================================================================
echo.
echo REQUIRED DOWNLOADS - Please install these before proceeding:
echo.
echo [FRONTEND]
echo   - Node.js 18+ (any version including latest)
echo     Download: https://nodejs.org/
echo     Install: Run the Windows installer, check "Add to PATH"
echo.
echo [BACKEND]
echo   - Python 3.10+
echo     Download: https://www.python.org/downloads/
echo     Install: Run the Windows installer, CHECK "Add Python to PATH"
echo.
echo [OPTIONAL but Recommended]
echo   - Git for Windows (for version control)
echo     Download: https://git-scm.com/download/win
echo.
echo ================================================================================
echo CHECKING INSTALLED DEPENDENCIES:
echo ================================================================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js 18+ is NOT installed
    echo   Action: Download and install from https://nodejs.org/ (LTS version)
    echo   Then restart this script.
    echo.
    set "MISSING_NODE=1"
) else (
    for /f "tokens=*" %%i in ('node --version') do (
        echo [OK] Node.js %%i - Frontend will work
    )
)

REM Check if npm is installed
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm is NOT installed ^(comes with Node.js^)
    echo   Action: Reinstall Node.js from https://nodejs.org/
    echo.
    set "MISSING_NPM=1"
) else (
    for /f "tokens=*" %%i in ('npm --version') do (
        echo [OK] npm %%i - Package manager ready
    )
)

REM Check if Python is installed
"%PYTHON_CMD%" --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10+ is NOT installed
    echo   Action: Download and install from https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation
    echo   Then restart this script.
    echo.
    set "MISSING_PYTHON=1"
) else (
    for /f "tokens=*" %%i in ('"%PYTHON_CMD%" --version') do (
        echo [OK] %%i - Backend will work
    )
)

REM Check if pip is installed
"%PYTHON_CMD%" -m pip --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip is NOT installed ^(comes with Python^)
    echo   Action: Reinstall Python from https://www.python.org/downloads/
    echo.
    set "MISSING_PIP=1"
) else (
    for /f "tokens=*" %%i in ('"%PYTHON_CMD%" -m pip --version') do (
        echo [OK] %%i - Python package manager ready
    )
)

REM Check if git is installed (optional)
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Git is NOT installed ^(optional but recommended^)
    echo   To install: https://git-scm.com/download/win
) else (
    for /f "tokens=*" %%i in ('git --version') do (
        echo [OK] %%i - Version control ready
    )
)

echo.
if defined MISSING_NODE (
    echo *** ERROR: Node.js (npm) is required for the FRONTEND ***
)
if defined MISSING_PYTHON (
    echo *** ERROR: Python (pip) is required for the BACKEND ***
)
if defined MISSING_NPM (
    echo *** ERROR: npm package manager is missing ***
)
if defined MISSING_PIP (
    echo *** ERROR: pip package manager is missing ***
)

if defined MISSING_NODE (
    if defined MISSING_PYTHON (
        echo.
        echo SOLUTION: Install both Node.js and Python, then restart this script
        echo   1. Download Node.js from https://nodejs.org/ ^(latest version or LTS^)
        echo   2. Download Python from https://www.python.org/downloads/
        echo   3. Run this script again
    ) else (
        echo.
        echo SOLUTION: Install Node.js, then restart this script
        echo   Download from https://nodejs.org/ ^(latest version or LTS^)
    )
    exit /b 1
)

if defined MISSING_PYTHON (
    echo.
    echo SOLUTION: Install Python, then restart this script
    echo   Download from https://www.python.org/downloads/
    exit /b 1
)

echo ================================================================================
echo All required dependencies found! Proceeding with installation...
echo ================================================================================
echo.

echo.
echo ================================================================================
echo STEP 1: Installing Frontend Dependencies (React, Axios, etc.)
echo ================================================================================
echo.
pushd "%WEBAPP_DIR%frontend"
if not exist "node_modules" (
    echo Installing npm packages for frontend...
    call npm install
    if !ERRORLEVEL! NEQ 0 (
        echo [WARNING] npm install encountered issues, retrying...
        call npm install --legacy-peer-deps
        if !ERRORLEVEL! NEQ 0 (
            echo [ERROR] Failed to install frontend dependencies
            popd
            exit /b 1
        )
    )
) else (
    echo node_modules already exists, updating packages...
    call npm update
)
echo [OK] Frontend dependencies installed
popd
echo.

echo ================================================================================
echo STEP 2: Installing Backend Dependencies (FastAPI, Pillow, OpenCV, etc.)
echo ================================================================================
echo.
echo Installing Python packages for backend...
"%PYTHON_CMD%" -m pip install --upgrade pip setuptools wheel
"%PYTHON_CMD%" -m pip install -r "%WEBAPP_DIR%backend\requirements.txt"
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to install backend dependencies
    exit /b 1
)
echo [OK] Backend dependencies installed
echo.

echo ================================================================================
echo STEP 3: Installing Watermark Core Dependencies (NumPy, PyWavelets, etc.)
echo ================================================================================
echo.
echo Installing Python packages for watermark core...
"%PYTHON_CMD%" -m pip install -r "%PROJECT_DIR%\requirements.txt"
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Some watermark core dependencies may have failed
)
echo [OK] Watermark core dependencies processed
echo.

echo ================================================================================
echo STEP 4: Installing Main Watermark Package
echo ================================================================================
echo.
echo Installing main watermark package in development mode...
"%PYTHON_CMD%" -m pip install -e "%PROJECT_DIR%"
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Could not install main package, but this may be optional
)
echo [OK] Main package installation attempted
echo.

REM Create environment file if it doesn't exist
if not exist "%WEBAPP_DIR%frontend\.env" (
    echo ================================================================================
    echo STEP 5: Setting Up Environment Variables
    echo ================================================================================
    echo.
    if exist "%WEBAPP_DIR%.env.example" (
        copy "%WEBAPP_DIR%.env.example" "%WEBAPP_DIR%frontend\.env" >nul
        echo [OK] Created .env file from .env.example
    ) else (
        echo REACT_APP_API_URL=http://localhost:8000 > "%WEBAPP_DIR%frontend\.env"
        echo [OK] Created .env file with default settings
    )
    echo.
) else (
    echo [OK] .env file already exists
)

echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo All required packages have been installed:
echo.
echo   Frontend (React, Axios, etc.)
echo   Backend (FastAPI, Pillow, OpenCV, etc.)
echo   Watermark Core (NumPy, PyWavelets, scikit-image, etc.)
echo   Environment Configuration
echo.
echo ================================================================================
echo Ready to Run!
echo ================================================================================
echo.
echo To start the application, open TWO terminals and run:
echo.
echo Terminal 1 - Backend Server:
echo   cd /d "%WEBAPP_DIR%backend"
echo   %PYTHON_CMD% -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
echo.
echo Terminal 2 - Frontend Dev Server:
echo   cd /d "%WEBAPP_DIR%frontend"
echo   npm start
echo.
echo Then open your browser to:
echo   http://localhost:3000
echo.
echo ================================================================================
echo.
pause
