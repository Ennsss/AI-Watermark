"""Report service for generating verification reports."""

import csv
import io
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from database import Verification, Artwork


class ReportService:
    """Service for generating reports."""
    
    DISCLAIMER = "This report represents a technical watermark verification result produced by the system. It does not constitute legal proof of authorship, ownership, or copyright."
    
    @staticmethod
    def generate_verification_csv(db: Session, verification_id: str) -> Optional[str]:
        """Generate a CSV report for a verification.
        
        Args:
            db: Database session
            verification_id: Verification ID
        
        Returns:
            CSV content as string or None if verification not found
        """
        verification = db.query(Verification).filter(
            Verification.verification_id == verification_id
        ).first()
        
        if not verification:
            return None
        
        artwork = db.query(Artwork).filter(
            Artwork.artwork_id == verification.artwork_id
        ).first()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header with verification info
        writer.writerow(["Verification Report"])
        writer.writerow([])
        
        # Verification details
        writer.writerow(["Verification ID", verification.verification_id])
        writer.writerow(["Verification Date", verification.verification_date.isoformat()])
        writer.writerow(["Verification Result", verification.result_status.upper()])
        writer.writerow([])
        
        # Artwork details
        if artwork:
            writer.writerow(["Artwork ID", artwork.artwork_id])
            writer.writerow(["Artwork Title", artwork.title])
            writer.writerow(["Creator", artwork.creator_name])
            writer.writerow(["Registration Date", artwork.registration_date.isoformat()])
        writer.writerow([])
        
        # Suspected image details
        writer.writerow(["Suspected Filename", verification.suspected_filename])
        writer.writerow([])
        
        # Technical details
        writer.writerow(["Technical Analysis"])
        writer.writerow(["Expected Payload", verification.expected_payload or "N/A"])
        writer.writerow(["Extracted Payload", verification.extracted_payload or "N/A"])
        differing_bits = "N/A"
        if verification.expected_payload and verification.extracted_payload:
            try:
                expected = bin(int(verification.expected_payload, 16))[2:].zfill(len(verification.expected_payload) * 4)
                extracted = bin(int(verification.extracted_payload, 16))[2:].zfill(len(verification.extracted_payload) * 4)
                differing_bits = sum(a != b for a, b in zip(expected, extracted)) + abs(len(expected) - len(extracted))
            except ValueError:
                pass
        writer.writerow(["Differing Bits", differing_bits])
        writer.writerow(["Payload Length", len(verification.expected_payload) * 4 if verification.expected_payload else "N/A"])
        writer.writerow(["BER (Bit Error Rate)", verification.ber if verification.ber is not None else "N/A"])
        writer.writerow(["Threshold Used", verification.threshold_used if verification.threshold_used is not None else "N/A"])
        writer.writerow(["Policy Version", verification.policy_version or "Legacy / unavailable"])
        writer.writerow(["Threshold Status", "Provisional" if (verification.policy_version or "").startswith("provisional") else "Historical / see policy version"])
        writer.writerow(["Watermark Engine", "DWT-QIM"])
        writer.writerow(["Processing Time (ms)", verification.processing_time_ms if verification.processing_time_ms is not None else "N/A"])
        
        if verification.error_message:
            writer.writerow(["Error Message", verification.error_message])
        
        writer.writerow([])
        writer.writerow(["Disclaimer"])
        writer.writerow([ReportService.DISCLAIMER])
        
        return output.getvalue()
    
    @staticmethod
    def generate_batch_csv(db: Session, limit: int = 100) -> str:
        """Generate a batch report of all verifications.
        
        Args:
            db: Database session
            limit: Maximum number of records to include
        
        Returns:
            CSV content as string
        """
        verifications = db.query(Verification).order_by(
            Verification.verification_date.desc()
        ).limit(limit).all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Verification ID",
            "Artwork ID",
            "Suspected Filename",
            "Verification Date",
            "Result",
            "BER",
            "Processing Time (ms)",
        ])
        
        # Write data rows
        for ver in verifications:
            writer.writerow([
                ver.verification_id,
                ver.artwork_id,
                ver.suspected_filename,
                ver.verification_date.isoformat(),
                ver.result_status,
                ver.ber if ver.ber is not None else "N/A",
                ver.processing_time_ms,
            ])
        
        return output.getvalue()
