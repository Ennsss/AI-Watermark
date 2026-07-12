#!/bin/bash
# Start AI Watermark Web App (macOS/Linux)

set -e

echo ""
echo "================================"
echo "AI Watermark Web Application"
echo "================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed. Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

# Install frontend dependencies
echo ""
echo "[1/4] Installing frontend dependencies..."
cd frontend
npm install

# Install backend dependencies
echo ""
echo "[2/4] Installing backend dependencies..."
cd ../backend
pip install -r requirements.txt

# Install main package
echo ""
echo "[3/4] Installing main watermark package..."
pip install -e ../..

# Create environment file if it doesn't exist
if [ ! -f "../frontend/.env" ]; then
    echo ""
    echo "[4/4] Creating .env file..."
    cp ../.env.example ../frontend/.env
    echo ""
    echo "Created .env file. Update REACT_APP_API_URL if needed."
fi

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To start the application:"
echo ""
echo "Terminal 1 - Backend:"
echo "  cd backend"
echo "  python main.py"
echo ""
echo "Terminal 2 - Frontend:"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "Then open http://localhost:3000 in your browser"
echo ""
