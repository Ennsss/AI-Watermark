"""FastAPI backend for AI Watermark web application."""

import os
import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Add parent src directory to path to import watermark modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from watermark.embedding import embed_watermark, extract_watermark
from watermark.preprocessor import rgb_to_ycbcr, extract_y_channel, pad_to_multiple, ycbcr_to_rgb
import cv2

app = FastAPI(
    title="AI Watermark Web API",
    description="DWT-QIM watermarking API for digital art",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Default watermark parameters (from your main_experiment.yaml)
DEFAULT_PARAMS = {
    "wavelet": "haar",
    "dwt_level": 2,
    "target_subbands": ["LH2", "HL2"],
    "delta": 16.0,  # Default quantization step
    "payload": bytes([0xAA] * 16),  # 128-bit test payload
}


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def validate_image(image_array: np.ndarray, max_size: int = 2048) -> None:
    """Validate image dimensions and format."""
    if len(image_array.shape) not in (2, 3):
        raise ValueError("Image must be grayscale or RGB/BGR")
    
    height, width = image_array.shape[:2]
    if height > max_size or width > max_size:
        raise ValueError(f"Image dimensions exceed maximum {max_size}x{max_size}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Watermark API"}


@app.post("/api/embed")
async def embed_watermark_endpoint(
    file: UploadFile = File(...),
    delta: float = Form(DEFAULT_PARAMS["delta"]),
    payload: Optional[str] = Form(None),
):
    """
    Embed watermark into an image.
    
    Parameters:
    - file: Image file (JPEG, PNG)
    - delta: Quantization step (default: 16)
    - payload: Optional 128-bit payload as hex string (default: test payload)
    
    Returns:
    - Watermarked image as base64-encoded PNG
    - Embedding parameters used
    """
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if delta < 4 or delta > 64:
        raise HTTPException(status_code=400, detail="Delta must be between 4 and 64")
    
    try:
        # Read image
        image_data = await file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate
        validate_image(image_array)
        
        # Resize to expected dimensions if needed
        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            image_array = cv2.resize(image_array, (512, 512))
        
        # Parse payload - convert hex string to bit array
        watermark_payload = DEFAULT_PARAMS["payload"]
        if payload:
            try:
                watermark_payload = bytes.fromhex(payload)
                if len(watermark_payload) != 16:
                    raise ValueError("Payload must be 128 bits (16 bytes)")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
        
        # Convert bytes to bit array
        bits = np.unpackbits(np.frombuffer(watermark_payload, dtype=np.uint8))
        
        # Convert RGB to YCbCr and extract Y channel
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Embed watermark into Y channel
        y_watermarked = embed_watermark(
            y_padded,
            bits,
            seed=42,  # Fixed seed for reproducibility
            delta=float(delta),
            wavelet=DEFAULT_PARAMS["wavelet"],
            level=DEFAULT_PARAMS["dwt_level"],
            target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
        )
        
        # Remove padding
        y_watermarked = y_watermarked[:y_channel.shape[0], :y_channel.shape[1]]
        
        # Replace Y channel in YCbCr
        ycbcr[:, :, 0] = y_watermarked
        
        # Convert back to RGB
        watermarked_rgb = ycbcr_to_rgb(ycbcr)
        watermarked_rgb = np.clip(watermarked_rgb, 0, 255).astype(np.uint8)
        
        # Convert to base64
        result_base64 = image_to_base64(watermarked_rgb)
        
        return JSONResponse({
            "status": "success",
            "image": result_base64,
            "format": "png",
            "size": image_array.shape,
            "parameters": {
                "delta": float(delta),
                "wavelet": DEFAULT_PARAMS["wavelet"],
                "dwt_level": DEFAULT_PARAMS["dwt_level"],
                "subbands": DEFAULT_PARAMS["target_subbands"],
                "payload_hex": watermark_payload.hex(),
            },
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/extract")
async def extract_watermark_endpoint(
    file: UploadFile = File(...),
    delta: float = Form(DEFAULT_PARAMS["delta"]),
    use_cnn: bool = Form(False),
):
    """
    Extract watermark from an image.
    
    Parameters:
    - file: Watermarked image file (JPEG, PNG)
    - delta: Quantization step used for embedding (default: 16)
    - use_cnn: Use CNN-assisted extraction (if model available)
    
    Returns:
    - Extracted bitstream (as hex)
    - Bit Error Rate (BER)
    - Confidence metrics
    """
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if delta < 4 or delta > 64:
        raise HTTPException(status_code=400, detail="Delta must be between 4 and 64")
    
    try:
        # Read image
        image_data = await file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate
        validate_image(image_array)
        
        # Resize if needed
        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            image_array = cv2.resize(image_array, (512, 512))
        
        # Convert RGB to YCbCr and extract Y channel
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Extract watermark from Y channel
        num_bits = 128  # Fixed payload size
        extracted_bits = extract_watermark(
            y_padded,
            num_bits=num_bits,
            seed=42,  # Must match embedding
            delta=float(delta),
            wavelet=DEFAULT_PARAMS["wavelet"],
            level=DEFAULT_PARAMS["dwt_level"],
            target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
        )
        
        # Convert bit array to hex
        extracted_bytes = np.packbits(extracted_bits)
        extracted_hex = extracted_bytes.tobytes().hex()
        
        # Calculate BER against default payload (simple confidence metric)
        default_bits = np.unpackbits(np.frombuffer(DEFAULT_PARAMS["payload"], dtype=np.uint8))
        ber = float(np.sum(extracted_bits != default_bits)) / len(extracted_bits)
        confidence = max(0, 1.0 - ber)  # Confidence decreases with BER
        
        return JSONResponse({
            "status": "success",
            "extracted_payload": extracted_hex,
            "bit_error_rate": ber,
            "confidence": confidence,
            "parameters": {
                "delta": float(delta),
                "wavelet": DEFAULT_PARAMS["wavelet"],
                "dwt_level": DEFAULT_PARAMS["dwt_level"],
            },
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/detect")
async def detect_watermark_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """
    Detect if an image contains a watermark by trying multiple Delta values.
    
    Parameters:
    - file: Image file (JPEG, PNG)
    - threshold: Confidence threshold for detection (0-1, default: 0.5)
    
    Returns:
    - watermark_detected: Boolean indicating if watermark was found
    - confidence: Confidence score (0-1)
    - detection_probability: Probability that watermark is present
    - delta_used: The Delta value that gave the best detection result
    """
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    try:
        # Read image
        image_data = await file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate
        validate_image(image_array)
        
        # Resize if needed
        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            image_array = cv2.resize(image_array, (512, 512))
        
        # Convert RGB to YCbCr and extract Y channel
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Try multiple common Delta values and find best detection
        delta_values = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
        best_confidence = 0
        best_delta = 16
        best_ber = 1.0
        default_bits = np.unpackbits(np.frombuffer(DEFAULT_PARAMS["payload"], dtype=np.uint8))
        
        for delta in delta_values:
            try:
                num_bits = 128  # Fixed payload size
                extracted_bits = extract_watermark(
                    y_padded,
                    num_bits=num_bits,
                    seed=42,  # Must match embedding
                    delta=float(delta),
                    wavelet=DEFAULT_PARAMS["wavelet"],
                    level=DEFAULT_PARAMS["dwt_level"],
                    target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
                )
                
                # Calculate BER against default payload
                ber = float(np.sum(extracted_bits != default_bits)) / len(extracted_bits)
                confidence = max(0, 1.0 - ber)  # Confidence decreases with BER
                
                # Track best result
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_delta = delta
                    best_ber = ber
            except Exception:
                # If extraction fails for this delta, continue trying others
                continue
        
        # Determine if watermark is detected based on threshold
        watermark_detected = best_confidence >= threshold
        
        return JSONResponse({
            "status": "success",
            "watermark_detected": watermark_detected,
            "confidence": best_confidence,
            "detection_probability": best_confidence,
            "bit_error_rate": best_ber,
            "threshold_used": threshold,
            "parameters": {
                "delta": float(best_delta),
                "wavelet": DEFAULT_PARAMS["wavelet"],
                "dwt_level": DEFAULT_PARAMS["dwt_level"],
            },
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/config")
async def get_config():
    """Get default configuration parameters."""
    return {
        "defaults": DEFAULT_PARAMS,
        "constraints": {
            "min_delta": 4,
            "max_delta": 64,
            "min_image_size": 256,
            "max_image_size": 2048,
            "payload_bits": 128,
        },
        "supported_formats": ["JPEG", "PNG"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
