"""Verification service for managing watermark verification."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from sqlalchemy.orm import Session
from database import Verification, Artwork
from utils import generate_verification_id
from verification_policy import VERIFICATION_POLICY


class VerificationService:
    """Service for managing verification records."""
    
    POLICY = VERIFICATION_POLICY
    
    def create_verification(
        self,
        db: Session,
        artwork_id: str,
        suspected_filename: str,
        suspected_file_path: str,
        expected_payload: str,
        extracted_payload: Optional[str],
        ber: Optional[float],
        processing_time_ms: float,
        error_message: Optional[str] = None,
    ) -> Verification:
        """Create a verification record.
        
        Args:
            db: Database session
            artwork_id: Associated artwork ID
            suspected_filename: Name of suspected image file
            suspected_file_path: Path to suspected image file
            expected_payload: Expected payload from registry
            extracted_payload: Extracted payload from image
            ber: Bit Error Rate if available
            processing_time_ms: Processing time in milliseconds
            error_message: Error message if extraction failed
        
        Returns:
            Created Verification record
        """
        verification_id = generate_verification_id(db)
        
        result_status, _ = self.POLICY.classify(ber, error_message)
        
        verification = Verification(
            verification_id=verification_id,
            artwork_id=artwork_id,
            suspected_filename=suspected_filename,
            suspected_file_path=suspected_file_path,
            verification_date=datetime.utcnow(),
            result_status=result_status,
            expected_payload=expected_payload,
            extracted_payload=extracted_payload,
            ber=ber,
            processing_time_ms=processing_time_ms,
            error_message=error_message,
            threshold_used=self.POLICY.detection_ber_threshold,
            policy_version=self.POLICY.policy_version,
        )
        
        db.add(verification)
        db.commit()
        db.refresh(verification)
        
        return verification
    
    def get_verification(self, db: Session, verification_id: str) -> Optional[Verification]:
        """Get verification by ID."""
        return db.query(Verification).filter(Verification.verification_id == verification_id).first()
    
    def get_verifications_for_artwork(
        self,
        db: Session,
        artwork_id: str,
    ) -> List[Verification]:
        """Get all verifications for an artwork."""
        return db.query(Verification).filter(
            Verification.artwork_id == artwork_id
        ).order_by(Verification.verification_date.desc()).all()
    
    def get_all_verifications(self, db: Session) -> List[Verification]:
        """Get all verifications."""
        return db.query(Verification).order_by(Verification.verification_date.desc()).all()

    def delete_verification(self, db: Session, verification_id: str) -> bool:
        """Delete a verification record and any stored suspected image file."""
        verification = self.get_verification(db, verification_id)
        if verification is None:
            return False

        suspected_path = Path(verification.suspected_file_path) if verification.suspected_file_path else None
        db.delete(verification)
        db.commit()

        if suspected_path and suspected_path.exists():
            try:
                suspected_path.unlink()
            except OSError:
                # File cleanup failure should not fail the API after DB deletion.
                pass

        return True
    
    def classify_result(self, ber: Optional[float], error: Optional[str] = None) -> tuple:
        """Classify verification result and return status with message.
        
        Returns:
            (status, display_message)
        """
        return self.POLICY.classify(ber, error)
    
    def get_dashboard_stats(self, db: Session) -> dict:
        """Get statistics for dashboard."""
        total_artworks = db.query(Artwork).filter(
            Artwork.archived_at.is_(None),
            Artwork.watermark_status != "archived",
        ).count()
        total_verifications = db.query(Verification).count()
        
        matches = db.query(Verification).filter(Verification.result_status == "match").count()
        partials = db.query(Verification).filter(Verification.result_status == "partial").count()
        no_matches = db.query(Verification).filter(Verification.result_status == "no_match").count()
        errors = total_verifications - matches - partials - no_matches
        
        return {
            "total_artworks": total_artworks,
            "total_verifications": total_verifications,
            "matches": matches,
            "partials": partials,
            "no_matches": no_matches,
            "errors": errors,
        }
    
    def get_recent_activity(self, db: Session, limit: int = 10) -> List[dict]:
        """Get recent activity for dashboard."""
        artworks = db.query(Artwork).filter(
            Artwork.archived_at.is_(None),
            Artwork.watermark_status != "archived",
        ).order_by(Artwork.registration_date.desc()).limit(limit).all()
        verifications = db.query(Verification).order_by(Verification.verification_date.desc()).limit(limit).all()
        
        events = []
        
        for art in artworks:
            events.append({
                "type": "artwork_registered",
                "timestamp": art.registration_date,
                "text": f"{art.artwork_id} registered",
                "artwork_id": art.artwork_id,
            })
        
        for ver in verifications:
            if ver.result_status == "match":
                text = f"{ver.verification_id} matched {ver.artwork_id}"
            elif ver.result_status == "partial":
                text = f"{ver.verification_id} partial result for {ver.artwork_id}"
            else:
                text = f"{ver.verification_id} verified against {ver.artwork_id}"
            
            events.append({
                "type": "verification_completed",
                "timestamp": ver.verification_date,
                "text": text,
                "verification_id": ver.verification_id,
            })
        
        # Sort by timestamp descending and return top N
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return events[:limit]
