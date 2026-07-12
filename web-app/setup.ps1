# Development startup script for Windows PowerShell

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "AI Watermark Web Application" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[Checking Prerequisites]" -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/"
    exit 1
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js is not installed" -ForegroundColor Red
    Write-Host "Please install Node.js 16+ from https://nodejs.org/"
    exit 1
}

Write-Host "✓ Python found: $(python --version)" -ForegroundColor Green
Write-Host "✓ Node.js found: $(node --version)" -ForegroundColor Green
Write-Host ""

# Install backend dependencies
Write-Host "[1/3] Installing Backend Dependencies..." -ForegroundColor Yellow
Push-Location backend
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install backend dependencies" -ForegroundColor Red
    exit 1
}
Pop-Location

# Install frontend dependencies
Write-Host "[2/3] Installing Frontend Dependencies..." -ForegroundColor Yellow
Push-Location frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install frontend dependencies" -ForegroundColor Red
    exit 1
}
Pop-Location

# Create .env if needed
Write-Host "[3/3] Configuring Environment..." -ForegroundColor Yellow
if (-not (Test-Path "frontend/.env")) {
    Copy-Item ".env.example" "frontend/.env"
    Write-Host "✓ Created frontend/.env" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the application:" -ForegroundColor White
Write-Host ""
Write-Host "Backend (Terminal 1):" -ForegroundColor Cyan
Write-Host "  cd backend`n  python main.py" -ForegroundColor White
Write-Host ""
Write-Host "Frontend (Terminal 2):" -ForegroundColor Cyan
Write-Host "  cd frontend`n  npm start" -ForegroundColor White
Write-Host ""
Write-Host "Then open http://localhost:3000 in your browser" -ForegroundColor Yellow
Write-Host ""
