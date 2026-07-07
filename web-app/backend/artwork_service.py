"""Artwork service for managing artwork registry."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session
from database import Artwork
from utils import generate_artwork_id, generate_payload_hex


class ArtworkService:
    """Service for managing artwork records."""
    
    def __init__(self, storage_base_path: str = "./storage"):
        self.storage_base_path = Path(storage_base_path)
        self.originals_dir = self.storage_base_path / "originals"
        self.watermarked_dir = self.storage_base_path / "watermarked"
        
        # Create directories if they don't exist
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.watermarked_dir.mkdir(parents=True, exist_ok=True)
    
    def create_artwork(
        self,
        db: Session,
        title: str,
        creator_name: str,
        original_filename: str,
        original_file_path: str,
        notes: Optional[str] = None,
    ) -> Artwork:
        """Create a new artwork record.
        
        Args:
            db: Database session
            title: Artwork title
            creator_name: Creator name
            original_filename: Original filename
            original_file_path: Path to original file
            notes: Optional notes
        
        Returns:
            Created Artwork record
        """
        artwork_id = generate_artwork_id(db)
        payload = generate_payload_hex(16)  # 128-bit payload
        
        artwork = Artwork(
            artwork_id=artwork_id,
            title=title,
            creator_name=creator_name,
            original_filename=original_filename,
            original_file_path=original_file_path,
            payload=payload,
            registration_date=datetime.utcnow(),
            watermark_status="registered",  # Will be updated to "embedded" after watermarking
            notes=notes,
        )
        
        db.add(artwork)
        db.commit()
        db.refresh(artwork)
        
        return artwork
    
    def get_artwork(self, db: Session, artwork_id: str) -> Optional[Artwork]:
        """Get artwork by ID."""
        return db.query(Artwork).filter(Artwork.artwork_id == artwork_id).first()
    
    def get_all_artworks(self, db: Session) -> List[Artwork]:
        """Get all artworks."""
        return db.query(Artwork).order_by(Artwork.registration_date.desc()).all()
    
    def update_watermark_status(
        self,
        db: Session,
        artwork_id: str,
        watermarked_filename: str,
        watermarked_file_path: str,
    ) -> Artwork:
        """Update artwork after watermarking."""
        artwork = self.get_artwork(db, artwork_id)
        if artwork:
            artwork.watermarked_filename = watermarked_filename
            artwork.watermarked_file_path = watermarked_file_path
            artwork.watermark_status = "embedded"
            db.commit()
            db.refresh(artwork)
        return artwork
    
    def get_payload(self, db: Session, artwork_id: str) -> Optional[str]:
        """Get payload for an artwork."""
        artwork = self.get_artwork(db, artwork_id)
        return artwork.payload if artwork else None

    def resolve_watermarked_path(self, artwork: Artwork) -> Optional[Path]:
        """Resolve a watermarked image path even if an old absolute path was stored."""
        if not artwork or not artwork.watermarked_file_path:
            return None

        stored_path = Path(artwork.watermarked_file_path)
        if stored_path.exists():
            return stored_path

        if artwork.watermarked_filename:
            fallback_path = self.watermarked_dir / artwork.watermarked_filename
            if fallback_path.exists():
                return fallback_path

        return None

    def resolve_original_path(self, artwork: Artwork) -> Optional[Path]:
        """Resolve an original image path even if an old absolute path was stored."""
        if not artwork or not artwork.original_file_path:
            return None

        stored_path = Path(artwork.original_file_path)
        if stored_path.exists():
            return stored_path

        if artwork.original_filename:
            fallback_path = self.originals_dir / artwork.original_filename
            if fallback_path.exists():
                return fallback_path

        return None
    
    def delete_artwork(self, db: Session, artwork_id: str) -> bool:
        """Delete an artwork and its files."""
        artwork = self.get_artwork(db, artwork_id)
        if artwork:
            # Delete files if they exist
            original_path = self.resolve_original_path(artwork)
            watermarked_path = self.resolve_watermarked_path(artwork)
            if original_path:
                os.remove(original_path)
            if watermarked_path:
                os.remove(watermarked_path)
            
            db.delete(artwork)
            db.commit()
            return True
        return False
