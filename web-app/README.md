# AI Watermark Web Application

A mobile-responsive web application for embedding and extracting DWT-QIM watermarks on digital art.

## Features

- 🎨 **Embed Watermarks** - Add invisible DWT-QIM watermarks to digital images
- 🔍 **Extract Watermarks** - Detect and extract embedded watermarks from images
- 📱 **Mobile Responsive** - Works seamlessly on desktop, tablet, and mobile devices
- ⚡ **Real-time Processing** - Fast image processing with instant feedback
- 🔒 **Provenance Tracking** - Protect digital art ownership with robust watermarking

## Architecture

```
web-app/
├── backend/          # FastAPI server
│   ├── main.py      # API endpoints
│   └── requirements.txt
└── frontend/         # React web app
    ├── src/
    │   ├── App.jsx      # Main component
    │   ├── App.css      # Styles
    │   └── index.jsx    # Entry point
    ├── public/
    │   └── index.html
    └── package.json
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
pip install -e ../../  # Install the main AI Watermark package
```

2. Run the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Install Node dependencies:
```bash
cd frontend
npm install
```

2. Create `.env` file:
```bash
cp ../.env.example .env
# Update REACT_APP_API_URL if needed
```

3. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## API Endpoints

### Health Check
```
GET /health
```

### Embed Watermark
```
POST /api/embed
- file: Image file (multipart/form-data)
- delta: Quantization step (optional, default: 16)
- payload: 128-bit hex payload (optional)

Response:
{
  "status": "success",
  "image": "base64_encoded_image",
  "format": "png",
  "size": [512, 512],
  "parameters": {...}
}
```

### Extract Watermark
```
POST /api/extract
- file: Image file (multipart/form-data)
- delta: Quantization step used for embedding (default: 16)

Response:
{
  "status": "success",
  "extracted_payload": "hex_string",
  "bit_error_rate": 0.0,
  "confidence": 0.95,
  "parameters": {...}
}
```

### Get Configuration
```
GET /api/config

Response:
{
  "defaults": {...},
  "constraints": {...},
  "supported_formats": ["JPEG", "PNG"]
}
```

## Parameters

### Quantization Step (Delta)
- **Range**: 4 - 64
- **Default**: 16
- **Effect**: Higher values = more robust but more visible; Lower values = less visible but less robust

### Image Size
- **Expected**: 512 x 512 pixels
- **Supported**: 256 x 256 to 2048 x 2048 (auto-resized)
- **Max file size**: 50 MB

### Payload
- **Size**: 128 bits (16 bytes)
- **Format**: Hexadecimal string (32 characters)
- **Default**: Auto-generated test payload if not specified

## Usage Examples

### Embed a Watermark

1. Select an image from your device
2. Adjust Delta slider (robustness vs visibility)
3. Optionally enter a custom 128-bit payload
4. Click "Embed Watermark"
5. Download the watermarked image

### Extract a Watermark

1. Select a watermarked image
2. Set Delta to the value used during embedding
3. Click "Extract Watermark"
4. View the extracted payload and metrics:
   - **Bit Error Rate (BER)**: Percentage of bits that differ
   - **Confidence**: Extraction confidence score

## Performance

- Image processing: ~500-1000ms per image (depends on size)
- API response time: <2s for typical images
- Supported concurrent requests: Limited by backend workers

## Troubleshooting

### "Failed to embed watermark"
- Check image format (JPEG/PNG)
- Verify image size is reasonable
- Ensure backend is running at correct address

### "Payload must be 128 bits"
- Enter exactly 32 hexadecimal characters
- Example: `aabbccddeeff00112233445566778899`

### CORS errors
- Ensure `REACT_APP_API_URL` matches backend URL
- Backend CORS is enabled for development

### Extraction shows high BER
- Image may have been degraded (compression, resize, crop)
- Try increasing Delta value used during embedding
- Original image may not have been watermarked

## Development

### Run Both Frontend and Backend

```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend  
cd frontend
npm start
```

### Build for Production

```bash
cd frontend
npm run build
```

Output will be in `frontend/build/`

## Deployment

### Docker (Optional)

```dockerfile
# Build image
docker build -f Dockerfile -t watermark-app .

# Run container
docker run -p 8000:8000 -p 3000:3000 watermark-app
```

### Cloud Deployment

For AWS/GCP/Azure deployment, containerize both services and deploy to:
- Backend: Cloud Run, Lambda, AppEngine
- Frontend: S3 + CloudFront, Cloud Storage, or static hosting

## Limitations

- Single-user tool (no authentication)
- Watermark detection only works on images watermarked with this application
- Does not protect against all types of image degradation
- Research implementation, not production-grade DRM

## Future Enhancements

- [ ] Batch processing
- [ ] User authentication and image library
- [ ] Advanced extraction (CNN-assisted)
- [ ] Watermark robustness analytics
- [ ] Image comparison tools
- [ ] Export analytics reports

## License

See main project LICENSE

## Support

For issues or questions, refer to the main project documentation in `docs/codebase_guide.md`
