"""Utility functions for ID and payload generation."""

import secrets
from sqlalchemy.orm import Session
from database import Artwork, Verification


def generate_artwork_id(db: Session) -> str:
    """Generate a unique artwork ID in format ART-XXXX."""
    counter = 1
    while True:
        artwork_id = f"ART-{counter:04d}"
        # Check if ID already exists
        existing = db.query(Artwork).filter(Artwork.artwork_id == artwork_id).first()
        if not existing:
            return artwork_id
        counter += 1


def generate_verification_id(db: Session) -> str:
    """Generate a unique verification ID in format VER-XXXX."""
    counter = 1
    while True:
        verification_id = f"VER-{counter:04d}"
        # Check if ID already exists
        existing = db.query(Verification).filter(Verification.verification_id == verification_id).first()
        if not existing:
            return verification_id
        counter += 1


def generate_payload_hex(byte_length: int = 16) -> str:
    """Generate a unique random payload as hex string.
    
    Args:
        byte_length: Number of bytes (default 16 = 128 bits)
    
    Returns:
        Hex string representation of random bytes
    """
    return secrets.token_hex(byte_length)


def bytes_from_hex_payload(hex_payload: str) -> bytes:
    """Convert hex payload string to bytes."""
    return bytes.fromhex(hex_payload)


def hex_from_bytes_payload(payload_bytes: bytes) -> str:
    """Convert bytes payload to hex string."""
    return payload_bytes.hex()
