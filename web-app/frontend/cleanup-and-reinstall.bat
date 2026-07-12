@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo.
echo ================================================================================
echo  Frontend Cleanup & Reinstall
echo ================================================================================
echo.

REM Check Node version
echo Checking Node.js version...
for /f "tokens=*" %%i in ('node --version') do (
    set "NODE_VERSION=%%i"
    echo Current Node.js: !NODE_VERSION!
)

echo.
echo React requires Node 18 or higher
echo You can use any version 18+ including the latest version
echo.

echo Proceeding with cleanup...
echo.

REM Delete node_modules
echo Step 1: Deleting corrupted node_modules...
if exist node_modules (
    echo   Removing node_modules directory...
    rmdir /s /q node_modules
    echo [OK] node_modules deleted
) else (
    echo [INFO] node_modules not found
)

REM Delete package-lock.json
echo.
echo Step 2: Deleting package-lock.json...
if exist package-lock.json (
    echo   Removing package-lock.json...
    del /q package-lock.json
    echo [OK] package-lock.json deleted
) else (
    echo [INFO] package-lock.json not found
)

REM Clear npm cache
echo.
echo Step 3: Clearing npm cache...
call npm cache clean --force
echo [OK] npm cache cleared

REM Reinstall dependencies
echo.
echo Step 4: Reinstalling dependencies...
call npm install
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] npm install encountered issues, retrying with legacy peer deps...
    call npm install --legacy-peer-deps
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to reinstall
        exit /b 1
    )
)
echo [OK] Dependencies reinstalled

echo.
echo ================================================================================
echo Cleanup Complete!
echo ================================================================================
echo.
echo *** IMPORTANT ***
echo.
echo You must downgrade Node.js to a supported version (18, 20, or 22):
echo.
echo 1. Download Node.js v22 LTS from https://nodejs.org/
echo 2. Uninstall your current Node.js 25.x
echo 3. Install the new Node.js v22 LTS
echo 4. Restart your terminal
echo 5. Run: npm start
echo.
echo ================================================================================
echo.
pause
