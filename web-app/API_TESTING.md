# API Testing Guide

## Using cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Configuration
```bash
curl http://localhost:8000/api/config
```

### Embed Watermark
```bash
curl -X POST http://localhost:8000/api/embed \
  -F "file=@path/to/image.jpg" \
  -F "delta=16" \
  -F "payload=aabbccddeeff00112233445566778899"
```

### Extract Watermark
```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@path/to/watermarked.jpg" \
  -F "delta=16"
```

## Using Python

```python
import requests

# Configuration
API_URL = "http://localhost:8000"

# Embed watermark
with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"delta": 16}
    response = requests.post(f"{API_URL}/api/embed", files=files, data=data)
    print(response.json())

# Extract watermark
with open("watermarked.jpg", "rb") as f:
    files = {"file": f}
    data = {"delta": 16}
    response = requests.post(f"{API_URL}/api/extract", files=files, data=data)
    print(response.json())
```

## Using JavaScript/Fetch

```javascript
// Embed watermark
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("delta", 16);

const response = await fetch("http://localhost:8000/api/embed", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result);

// Extract watermark
const formData2 = new FormData();
formData2.append("file", fileInput.files[0]);
formData2.append("delta", 16);

const response2 = await fetch("http://localhost:8000/api/extract", {
  method: "POST",
  body: formData2,
});

const result2 = await response2.json();
console.log(result2);
```

## Postman Collection

1. Open Postman
2. Create a new collection "AI Watermark"
3. Add the following requests:

### Request 1: Health Check
- Method: GET
- URL: http://localhost:8000/health

### Request 2: Get Config
- Method: GET
- URL: http://localhost:8000/api/config

### Request 3: Embed
- Method: POST
- URL: http://localhost:8000/api/embed
- Body (form-data):
  - file: (select image file)
  - delta: 16

### Request 4: Extract
- Method: POST
- URL: http://localhost:8000/api/extract
- Body (form-data):
  - file: (select watermarked image file)
  - delta: 16

## Response Examples

### Embed Success
```json
{
  "status": "success",
  "image": "iVBORw0KGgoAAAANS...",
  "format": "png",
  "size": [512, 512],
  "parameters": {
    "delta": 16,
    "wavelet": "haar",
    "dwt_level": 2,
    "subbands": ["LH2", "HL2"]
  }
}
```

### Extract Success
```json
{
  "status": "success",
  "extracted_payload": "aabbccddeeff00112233445566778899",
  "bit_error_rate": 0.0,
  "confidence": 0.95,
  "parameters": {
    "delta": 16,
    "wavelet": "haar",
    "dwt_level": 2
  }
}
```

### Error Response
```json
{
  "detail": "Image must be grayscale or RGB/BGR"
}
```

## Test Scenarios

### Scenario 1: Basic Watermarking
1. Embed watermark with default parameters
2. Download watermarked image
3. Extract watermark from downloaded image
4. Verify extracted payload matches

### Scenario 2: Different Delta Values
1. Embed with delta=8, 16, 32, 64
2. Extract each and compare visibility and BER
3. Note robustness trade-offs

### Scenario 3: Image Degradation
1. Embed watermark (delta=16)
2. Process watermarked image: JPEG compression, resize, crop
3. Extract and check BER
4. Verify robustness to degradation

### Scenario 4: Custom Payload
1. Generate random 128-bit hex payload
2. Embed with custom payload
3. Extract and verify payload matches
