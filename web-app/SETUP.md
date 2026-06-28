# Comprehensive Setup Guide

## Quick Start

### Windows
```bash
cd web-app
setup.bat
```

### macOS/Linux
```bash
cd web-app
chmod +x setup.sh
./setup.sh
```

## Step-by-Step Setup

### 1. Backend Setup

```bash
cd web-app/backend

# Install Python dependencies
pip install -r requirements.txt

# Install the main AI Watermark package
pip install -e ../..

# Start the FastAPI server
python main.py
```

The backend will start at `http://localhost:8000`

Verify it's working:
```bash
curl http://localhost:8000/health
```

### 2. Frontend Setup

In a new terminal:

```bash
cd web-app/frontend

# Install Node dependencies
npm install

# Create environment file
cp ../.env.example .env

# Start the React development server
npm start
```

The app will open at `http://localhost:3000`

## Docker Setup (Optional)

### Build Docker Image

```bash
docker build -t watermark-app -f Dockerfile .
```

### Run Docker Container

```bash
docker run -p 8000:8000 -p 3000:3000 watermark-app
```

## Configuration

### Environment Variables

Create `web-app/frontend/.env`:

```env
REACT_APP_API_URL=http://localhost:8000
```

For production deployment, update API URL to your backend server.

### Backend Configuration

Edit `web-app/backend/main.py` to customize:
- Default watermark parameters (delta, wavelet, DWT level)
- Image processing constraints
- Upload directory location

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'watermark'"

**Solution**: Install the main package
```bash
cd AI-Watermark
pip install -e .
```

### Issue: Port 8000 or 3000 already in use

**Backend (change port in main.py):**
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed to 8001
```

Then update frontend `.env`:
```env
REACT_APP_API_URL=http://localhost:8001
```

**Frontend:**
```bash
PORT=3001 npm start
```

### Issue: CORS errors in browser

Make sure backend is running and `.env` has correct API URL:
```env
REACT_APP_API_URL=http://localhost:8000
```

### Issue: Image processing is slow

- Check image size (should be < 2MB)
- Verify backend has enough CPU resources
- Try with smaller images first

## Testing

### Backend Tests

```bash
cd backend
# Add pytest to requirements and run
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Manual Testing

1. Open `http://localhost:3000`
2. Select a test image
3. Click "Embed Watermark"
4. Download the watermarked image
5. Upload the same image
6. Click "Extract Watermark"
7. Verify extraction results

## Performance Tips

1. **Images**: Use PNG for lossless watermarking, JPEG for lossy testing
2. **Delta**: Start with default 16, increase for better robustness
3. **Size**: Larger images process slower; 512x512 is optimal
4. **Concurrency**: Backend supports multiple concurrent requests

## Production Deployment

### Prerequisites
- Docker and Docker Compose (optional)
- Cloud hosting (AWS, GCP, Azure)
- SSL certificate (for HTTPS)

### Deployment Options

#### Option 1: VPS (AWS EC2, DigitalOcean, etc.)

```bash
# SSH into server
ssh user@your_server

# Clone repo
git clone <your-repo> AI-Watermark
cd AI-Watermark/web-app

# Install dependencies
./setup.sh

# Run with systemd/supervisord for persistence
# Backend
python backend/main.py

# Frontend (build and serve with nginx)
cd frontend
npm run build
# Serve 'build' directory with nginx
```

#### Option 2: Docker + Cloud Run

```bash
# Build image
docker build -t watermark-app .

# Push to registry
docker tag watermark-app gcr.io/your-project/watermark-app
docker push gcr.io/your-project/watermark-app

# Deploy to Cloud Run
gcloud run deploy watermark-app \
  --image gcr.io/your-project/watermark-app \
  --platform managed \
  --region us-central1
```

#### Option 3: Kubernetes

```bash
# Apply manifests
kubectl apply -f deployment.yaml
```

### Environment Variables for Production

```env
REACT_APP_API_URL=https://api.youromain.com
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_WORKERS=4
```

### Security Recommendations

1. Enable HTTPS/SSL
2. Rate limiting on API
3. Input validation (already implemented)
4. Remove debug logging in production
5. Implement authentication if needed
6. Regular security updates

## Next Steps

1. ✅ Start backend and frontend servers
2. 📱 Open app in browser
3. 🎨 Upload an image
4. 🔒 Embed a watermark
5. 📥 Download watermarked image
6. ✨ Extract watermark to verify

For more information, see `README.md` in this directory.
