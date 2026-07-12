# Web App Structure Overview

## Project Layout

```
AI-Watermark/
├── web-app/                          # NEW: Web application directory
│   ├── backend/                      # FastAPI backend server
│   │   ├── main.py                   # Main API application
│   │   └── requirements.txt          # Python dependencies
│   │
│   ├── frontend/                     # React frontend application
│   │   ├── public/
│   │   │   └── index.html            # HTML entry point
│   │   ├── src/
│   │   │   ├── App.jsx               # Main React component
│   │   │   ├── App.css               # Styling (mobile-responsive)
│   │   │   ├── index.jsx             # React root
│   │   │   └── index.css             # Global styles
│   │   ├── package.json              # Node dependencies
│   │   └── .env                      # Environment config
│   │
│   ├── README.md                     # Web app documentation
│   ├── SETUP.md                      # Detailed setup guide
│   ├── API_TESTING.md                # API testing guide
│   │
│   ├── .env.example                  # Environment template
│   ├── .gitignore                    # Git ignore rules
│   ├── setup.bat                     # Windows setup script
│   ├── setup.sh                      # Linux/macOS setup script
│   ├── setup.ps1                     # PowerShell setup script
│   ├── Dockerfile                    # Docker configuration
│   └── docker-compose.yml            # Docker Compose setup
│
├── src/                              # Existing: Main watermark library
└── ...other existing files...
```

## Features Implemented

### Backend (FastAPI)
✅ **RESTful API** with endpoints:
- `GET /health` - Health check
- `POST /api/embed` - Embed watermark in image
- `POST /api/extract` - Extract watermark from image  
- `GET /api/config` - Get configuration parameters

✅ **Image Processing**
- Automatic image resizing to 512x512
- Support for JPEG and PNG formats
- File size validation (max 50MB)
- Base64 encoding for image transmission

✅ **Watermarking Integration**
- Integrates with existing DWT-QIM implementation
- Configurable Delta (quantization step)
- Custom payload support (128-bit hex)
- BER and confidence metrics on extraction

✅ **Error Handling**
- Comprehensive validation
- Clear error messages
- CORS enabled for development

### Frontend (React)
✅ **Mobile Responsive Design**
- Touch-friendly interface
- Optimized for mobile, tablet, desktop
- Single-column layout on mobile
- Grid-based on larger screens

✅ **Tab Interface**
- Embed Watermark tab
- Extract Watermark tab
- Easy switching between modes

✅ **Image Upload**
- Drag-and-drop support
- File type validation
- Preview before processing
- Image selection history

✅ **Processing Controls**
- Delta slider (4-64 range)
- Optional custom payload input
- Real-time parameter adjustment
- Visual feedback during processing

✅ **Results Display**
- Embedded watermark preview
- Download functionality
- Extraction results (payload, BER, confidence)
- Parameter display

✅ **User Experience**
- Loading states with spinner
- Error alerts
- Success notifications
- Responsive button states

## Getting Started

### Quick Start (Windows)
```bash
cd AI-Watermark/web-app
setup.bat
```

### Quick Start (Linux/macOS)
```bash
cd AI-Watermark/web-app
chmod +x setup.sh
./setup.sh
```

### Manual Start

**Terminal 1 - Backend:**
```bash
cd web-app/backend
pip install -r requirements.txt
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd web-app/frontend
npm install
npm start
```

Then open: **http://localhost:3000**

## API Specification

### Base URL
`http://localhost:8000` (development)

### Embed Endpoint
```
POST /api/embed

Request:
- file: multipart/form-data (image file)
- delta: integer (4-64, optional, default: 16)
- payload: hex string (optional, 32 chars for 128-bit)

Response (200):
{
  "status": "success",
  "image": "base64_png_data",
  "format": "png",
  "size": [512, 512],
  "parameters": {
    "delta": 16,
    "wavelet": "haar",
    "dwt_level": 2,
    "subbands": ["LH2", "HL2"]
  }
}

Response (400/500):
{ "detail": "error_message" }
```

### Extract Endpoint
```
POST /api/extract

Request:
- file: multipart/form-data (watermarked image)
- delta: integer (4-64, optional, default: 16)

Response (200):
{
  "status": "success",
  "extracted_payload": "hex_string",
  "bit_error_rate": 0.0,
  "confidence": 0.95,
  "parameters": {
    "delta": 16,
    "wavelet": "haar",
    "dwt_level": 2
  }
}

Response (400/500):
{ "detail": "error_message" }
```

## Configuration

### Frontend Environment (`.env`)
```env
REACT_APP_API_URL=http://localhost:8000
```

### Backend Configuration
Edit `backend/main.py`:
- `DEFAULT_PARAMS`: Default watermarking parameters
- `MAX_FILE_SIZE`: Maximum upload size (default: 50MB)
- `UPLOAD_DIR`: Where to store temporary files

## Development Workflow

### File Structure
- **Frontend**: React with no build config (uses create-react-app standard)
- **Backend**: FastAPI application with modular design
- **Shared**: Uses existing watermark library from `src/`

### Making Changes

**Backend Changes:**
1. Edit `backend/main.py`
2. Backend auto-reloads with Uvicorn
3. Test with API_TESTING.md examples

**Frontend Changes:**
1. Edit files in `frontend/src/`
2. React dev server auto-refreshes
3. Check browser console for errors

### Debugging

**Backend Logs:**
- Printed to console during `python main.py`
- Check for error messages and stack traces

**Frontend Logs:**
- Browser console (F12 → Console tab)
- Network tab for API requests/responses

## Testing

### Manual Testing Steps
1. Open http://localhost:3000
2. Select a PNG/JPEG image
3. Adjust Delta slider if desired
4. Click "Embed Watermark"
5. Download the watermarked image
6. Upload the watermarked image
7. Click "Extract Watermark"
8. Verify extracted payload

### Test Cases
- ✓ Embed with default payload
- ✓ Embed with custom payload
- ✓ Extract with matching delta
- ✓ Extract with different delta (high BER expected)
- ✓ Different image formats
- ✓ Error handling (invalid files, oversized, etc.)

## Performance Metrics

- **Image Processing**: 500-1000ms per image
- **API Response**: <2s typical
- **Frontend Load**: <1s
- **Concurrent Requests**: Limited by backend workers

## Deployment Options

### Local Development
- Both services run on localhost
- No configuration needed

### Docker
```bash
docker-compose up
```

### Production (VPS)
- Backend: WSGI server (Gunicorn) + Nginx
- Frontend: Static build served by Nginx
- See SETUP.md for details

### Cloud (AWS, GCP, Azure)
- Backend: Cloud Run / Lambda / AppEngine
- Frontend: S3 + CloudFront / Cloud Storage
- See SETUP.md for cloud-specific guides

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Change port in `backend/main.py` and update `.env` |
| Port 3000 in use | Run `PORT=3001 npm start` in frontend |
| Module not found | Run `pip install -e ../..` in backend directory |
| CORS errors | Check `.env` has correct API URL |
| Slow processing | Try smaller images (512x512 optimal) |
| Extraction high BER | Use same delta as embedding |

## Tech Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Server**: Uvicorn
- **Image Processing**: OpenCV, PIL, NumPy
- **Watermarking**: Existing DWT/QIM implementation
- **Language**: Python 3.10+

### Frontend
- **Framework**: React 18
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Styling**: CSS3 (no build tool needed)
- **Node**: 16+

### DevOps
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Version Control**: Git

## Future Enhancements

- [ ] Batch processing multiple images
- [ ] User authentication and image gallery
- [ ] Advanced CNN-assisted extraction
- [ ] Watermark analytics and reporting
- [ ] Image comparison tools
- [ ] Robustness testing suite
- [ ] Export capabilities (CSV, PDF)
- [ ] Webhook notifications
- [ ] API key authentication

## Support & Resources

- **Main Documentation**: [README.md](README.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **API Testing**: [API_TESTING.md](API_TESTING.md)
- **Project Root**: See `src/` for watermarking implementation
- **Paper Reference**: See `RESEARCH_ALIGNMENT.md` in project root

## License

Same as main AI Watermark project.

---

**Last Updated**: 2026-06-27
**Web App Version**: 1.0.0
