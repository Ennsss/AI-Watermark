"""FastAPI backend for AI Watermark web application."""

import os
import io
import base64
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session

# Add parent src directory to path to import watermark modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from watermark.embedding import embed_watermark, extract_watermark
from watermark.preprocessor import rgb_to_ycbcr, extract_y_channel, pad_to_multiple, ycbcr_to_rgb
import cv2

# Import database and services
from database import init_db, get_db
from artwork_service import ArtworkService
from verification_service import VerificationService
from report_service import ReportService
from utils import bytes_from_hex_payload, hex_from_bytes_payload

app = FastAPI(
    title="AI Watermark Web API",
    description="Digital Artwork Provenance Management Platform",
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
STORAGE_DIR = Path(__file__).parent / "storage"
UPLOAD_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Initialize database
init_db()

# Initialize services
artwork_service = ArtworkService(str(STORAGE_DIR))
verification_service = VerificationService()
report_service = ReportService()

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


def file_to_base64(file_path: Path) -> Optional[str]:
    """Convert a file to base64 if it exists."""
    if not file_path.exists():
        return None

    return base64.b64encode(file_path.read_bytes()).decode()


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


# ============================================================================
# PRODUCT API ENDPOINTS - Digital Artwork Provenance Management
# ============================================================================

@app.get("/api/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary statistics."""
    stats = verification_service.get_dashboard_stats(db)
    return {
        "status": "success",
        "data": stats,
    }


@app.get("/api/dashboard/recent-activity")
async def get_recent_activity(db: Session = Depends(get_db), limit: int = 10):
    """Get recent activity for dashboard."""
    events = verification_service.get_recent_activity(db, limit)
    return {
        "status": "success",
        "data": events,
    }


@app.post("/api/artworks/register")
async def register_artwork(
    title: str = Form(...),
    creator_name: str = Form(...),
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Register a new artwork and embed watermark.
    
    Flow:
    1. Validate image
    2. Generate artwork ID
    3. Generate unique payload
    4. Embed watermark
    5. Save registry record
    6. Return watermarked image download
    """
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and validate image
        image_data = await file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate dimensions
        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            image_array = cv2.resize(image_array, (512, 512))
        
        # Create artwork record in database
        artwork = artwork_service.create_artwork(
            db,
            title=title,
            creator_name=creator_name,
            original_filename=file.filename,
            original_file_path=str(STORAGE_DIR / "originals" / file.filename),
            notes=notes,
        )
        
        # Save original file
        original_path = STORAGE_DIR / "originals" / f"{artwork.artwork_id}_{file.filename}"
        original_path.write_bytes(image_data)
        artwork.original_file_path = str(original_path)
        
        # Get payload from database
        payload_hex = artwork.payload
        watermark_payload = bytes_from_hex_payload(payload_hex)
        bits = np.unpackbits(np.frombuffer(watermark_payload, dtype=np.uint8))
        
        # Convert RGB to YCbCr and extract Y channel
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Embed watermark
        y_watermarked = embed_watermark(
            y_padded,
            bits,
            seed=42,
            delta=DEFAULT_PARAMS["delta"],
            wavelet=DEFAULT_PARAMS["wavelet"],
            level=DEFAULT_PARAMS["dwt_level"],
            target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
        )
        
        # Remove padding
        y_watermarked = y_watermarked[:y_channel.shape[0], :y_channel.shape[1]]
        
        # Replace Y channel
        ycbcr[:, :, 0] = y_watermarked
        
        # Convert back to RGB
        watermarked_rgb = ycbcr_to_rgb(ycbcr)
        watermarked_rgb = np.clip(watermarked_rgb, 0, 255).astype(np.uint8)
        
        # Save watermarked image
        watermarked_filename = f"{artwork.artwork_id}_watermarked.png"
        watermarked_path = STORAGE_DIR / "watermarked" / watermarked_filename
        watermarked_image = Image.fromarray(watermarked_rgb)
        watermarked_image.save(watermarked_path)
        
        # Update artwork record
        artwork_service.update_watermark_status(
            db,
            artwork.artwork_id,
            watermarked_filename,
            str(watermarked_path),
        )
        
        # Convert to base64 for immediate download
        result_base64 = image_to_base64(watermarked_rgb)
        
        return JSONResponse({
            "status": "success",
            "artwork_id": artwork.artwork_id,
            "title": artwork.title,
            "creator_name": artwork.creator_name,
            "registration_date": artwork.registration_date.isoformat(),
            "watermark_status": "embedded",
            "image": result_base64,
            "format": "png",
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/artworks")
async def get_artworks(db: Session = Depends(get_db)):
    """Get all registered artworks."""
    artworks = artwork_service.get_all_artworks(db)
    return {
        "status": "success",
        "data": [
            {
                "artwork_id": art.artwork_id,
                "title": art.title,
                "creator_name": art.creator_name,
                "registration_date": art.registration_date.isoformat(),
                "watermark_status": art.watermark_status,
                "notes": art.notes,
            }
            for art in artworks
        ],
    }


@app.get("/api/artworks/{artwork_id}")
async def get_artwork(artwork_id: str, db: Session = Depends(get_db)):
    """Get a specific artwork."""
    artwork = artwork_service.get_artwork(db, artwork_id)
    if not artwork:
        raise HTTPException(status_code=404, detail="Artwork not found")

    watermarked_path = Path(artwork.watermarked_file_path) if artwork.watermarked_file_path else None
    watermarked_image_base64 = file_to_base64(watermarked_path) if watermarked_path else None
    
    return {
        "status": "success",
        "data": {
            "artwork_id": artwork.artwork_id,
            "title": artwork.title,
            "creator_name": artwork.creator_name,
            "registration_date": artwork.registration_date.isoformat(),
            "watermark_status": artwork.watermark_status,
            "notes": artwork.notes,
            "watermarked_filename": artwork.watermarked_filename,
            "watermarked_download_url": f"/api/artworks/{artwork.artwork_id}/watermarked",
            "watermarked_image_base64": watermarked_image_base64,
        },
    }


@app.get("/api/artworks/{artwork_id}/watermarked")
async def download_watermarked_artwork(artwork_id: str, db: Session = Depends(get_db)):
    """Download the watermarked artwork image."""
    artwork = artwork_service.get_artwork(db, artwork_id)
    if not artwork or not artwork.watermarked_file_path:
        raise HTTPException(status_code=404, detail="Watermarked image not found")

    watermarked_path = Path(artwork.watermarked_file_path)
    if not watermarked_path.exists():
        raise HTTPException(status_code=404, detail="Watermarked image not found")

    return FileResponse(
        path=str(watermarked_path),
        media_type="image/png",
        filename=artwork.watermarked_filename or f"{artwork.artwork_id}_watermarked.png",
    )


@app.post("/api/verifications")
async def verify_image(
    artwork_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Verify an image against a selected artwork.
    
    Flow:
    1. Fetch artwork and get expected payload
    2. Validate suspected image
    3. Extract payload
    4. Calculate BER
    5. Classify result
    6. Save verification event
    """
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Get artwork and expected payload
        artwork = artwork_service.get_artwork(db, artwork_id)
        if not artwork:
            raise HTTPException(status_code=404, detail="Artwork not found")
        
        expected_payload_hex = artwork.payload
        expected_payload = bytes_from_hex_payload(expected_payload_hex)
        
        # Read and validate suspected image
        image_data = await file.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate dimensions
        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            image_array = cv2.resize(image_array, (512, 512))
        
        # Convert RGB to YCbCr
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Extract watermark
        start_time = time.time()
        num_bits = 128  # Fixed payload size
        extracted_bits = extract_watermark(
            y_padded,
            num_bits=num_bits,
            seed=42,
            delta=DEFAULT_PARAMS["delta"],
            wavelet=DEFAULT_PARAMS["wavelet"],
            level=DEFAULT_PARAMS["dwt_level"],
            target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
        )
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert extracted bits to hex
        extracted_bytes = np.packbits(extracted_bits)
        extracted_hex = extracted_bytes.tobytes().hex()
        
        # Calculate BER
        expected_bits = np.unpackbits(np.frombuffer(expected_payload, dtype=np.uint8))
        ber = float(np.sum(extracted_bits != expected_bits)) / len(extracted_bits)
        
        # Save suspected image file
        suspected_filename = f"VER_{file.filename}"
        suspected_path = STORAGE_DIR / "suspected" / suspected_filename
        suspected_path.parent.mkdir(parents=True, exist_ok=True)
        suspected_path.write_bytes(image_data)
        
        # Create verification record
        verification = verification_service.create_verification(
            db,
            artwork_id=artwork_id,
            suspected_filename=file.filename,
            suspected_file_path=str(suspected_path),
            expected_payload=expected_payload_hex,
            extracted_payload=extracted_hex,
            ber=ber,
            processing_time_ms=processing_time_ms,
        )
        
        # Classify result
        result_status, message = verification_service.classify_result(ber)
        
        return JSONResponse({
            "status": "success",
            "verification_id": verification.verification_id,
            "artwork_id": artwork_id,
            "result_status": verification.result_status,
            "ber": ber,
            "processing_time_ms": processing_time_ms,
            "threshold_used": verification_service.BER_THRESHOLD,
            "message": message,
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/verifications")
async def get_verifications(db: Session = Depends(get_db), artwork_id: Optional[str] = None):
    """Get verifications, optionally filtered by artwork."""
    if artwork_id:
        verifications = verification_service.get_verifications_for_artwork(db, artwork_id)
    else:
        verifications = verification_service.get_all_verifications(db)
    
    return {
        "status": "success",
        "data": [
            {
                "verification_id": ver.verification_id,
                "artwork_id": ver.artwork_id,
                "suspected_filename": ver.suspected_filename,
                "verification_date": ver.verification_date.isoformat(),
                "result_status": ver.result_status,
                "ber": ver.ber,
                "processing_time_ms": ver.processing_time_ms,
            }
            for ver in verifications
        ],
    }


@app.get("/api/verifications/{verification_id}")
async def get_verification_detail(verification_id: str, db: Session = Depends(get_db)):
    """Get details for a specific verification."""
    verification = verification_service.get_verification(db, verification_id)
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")
    
    artwork = artwork_service.get_artwork(db, verification.artwork_id)
    
    return {
        "status": "success",
        "data": {
            "verification_id": verification.verification_id,
            "artwork_id": verification.artwork_id,
            "artwork_title": artwork.title if artwork else None,
            "artwork_creator": artwork.creator_name if artwork else None,
            "suspected_filename": verification.suspected_filename,
            "verification_date": verification.verification_date.isoformat(),
            "result_status": verification.result_status,
            "ber": verification.ber,
            "processing_time_ms": verification.processing_time_ms,
            "threshold_used": verification.threshold_used,
        },
    }


@app.get("/api/verifications/{verification_id}/report.csv")
async def get_verification_report(verification_id: str, db: Session = Depends(get_db)):
    """Download verification report as CSV."""
    csv_content = report_service.generate_verification_csv(db, verification_id)
    
    if csv_content is None:
        raise HTTPException(status_code=404, detail="Verification not found")
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=verification_{verification_id}.csv"},
    )


# ============================================================================
# LEGACY WATERMARK TESTING ENDPOINTS (kept for backward compatibility)
# ============================================================================

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


@app.post("/api/remove")
async def remove_watermark_endpoint(
    file: UploadFile = File(...),
    delta: float = Form(DEFAULT_PARAMS["delta"]),
):
    """
    Remove watermark from an image by extracting and inverting the embedded signal.
    
    Parameters:
    - file: Watermarked image file (JPEG, PNG)
    - delta: Quantization step used for embedding (default: 16)
    
    Returns:
    - Cleaned image (watermark removed) as base64-encoded PNG
    - Processing details
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
        
        # Convert RGB to YCbCr and extract Y channel
        ycbcr = rgb_to_ycbcr(image_array)
        y_channel = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y_channel, multiple=2**DEFAULT_PARAMS["dwt_level"])
        
        # Step 1: Extract the watermark bits
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
        
        # Step 2: Invert the bits (bitwise NOT) to get inverse watermark
        inverse_bits = 1 - extracted_bits
        
        # Step 3: Embed the inverse watermark to cancel out the original
        y_cleaned = embed_watermark(
            y_padded,
            inverse_bits,
            seed=42,  # Must match original embedding
            delta=float(delta),
            wavelet=DEFAULT_PARAMS["wavelet"],
            level=DEFAULT_PARAMS["dwt_level"],
            target_subbands=tuple(s.lower() for s in DEFAULT_PARAMS["target_subbands"]),
        )
        
        # Step 4: Remove padding
        y_cleaned = y_cleaned[:y_channel.shape[0], :y_channel.shape[1]]
        
        # Replace Y channel in YCbCr
        ycbcr[:, :, 0] = y_cleaned
        
        # Convert back to RGB
        cleaned_rgb = ycbcr_to_rgb(ycbcr)
        cleaned_rgb = np.clip(cleaned_rgb, 0, 255).astype(np.uint8)
        
        # Convert to base64
        result_base64 = image_to_base64(cleaned_rgb)
        
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
                "extracted_payload": np.packbits(extracted_bits).tobytes().hex(),
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
