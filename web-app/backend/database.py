"""Database configuration and models for the capstone prototype."""

import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database configuration
DATABASE_PATH = Path(__file__).resolve().parent / "watermark_registry.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Models
class Artwork(Base):
    """Artwork registry model."""
    
    __tablename__ = "artworks"
    
    id = Column(Integer, primary_key=True, index=True)
    artwork_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    creator_name = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    original_file_path = Column(String)
    watermarked_filename = Column(String)
    watermarked_file_path = Column(String)
    payload = Column(Text, nullable=False)  # Store as hex string
    registration_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    watermark_status = Column(String, default="embedded", nullable=False)
    archived_at = Column(DateTime, nullable=True)
    notes = Column(Text)
    
    # Relationship
    verifications = relationship("Verification", back_populates="artwork", cascade="all, delete-orphan")


class Verification(Base):
    """Verification history model."""
    
    __tablename__ = "verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    verification_id = Column(String, unique=True, index=True, nullable=False)
    artwork_id = Column(String, ForeignKey("artworks.artwork_id"), nullable=False)
    suspected_filename = Column(String, nullable=False)
    suspected_file_path = Column(String)
    verification_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    result_status = Column(String, nullable=False)  # "match", "partial", "no_match", "error"
    expected_payload = Column(Text)
    extracted_payload = Column(Text)
    ber = Column(Float)  # Bit Error Rate
    processing_time_ms = Column(Float)
    error_message = Column(Text)
    threshold_used = Column(Float)
    policy_version = Column(String)
    archived_at = Column(DateTime, nullable=True)
    
    # Relationship
    artwork = relationship("Artwork", back_populates="verifications")


def init_db():
    """Initialize the database and create tables."""
    Base.metadata.create_all(bind=engine)
    # create_all does not add columns to an existing SQLite database. Keep this
    # small, idempotent migration here while the prototype has no migration tool.
    if "artworks" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("artworks")}
        if "archived_at" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE artworks ADD COLUMN archived_at DATETIME"))
    if "verifications" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("verifications")}
        if "policy_version" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE verifications ADD COLUMN policy_version VARCHAR"))
        if "archived_at" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE verifications ADD COLUMN archived_at DATETIME"))
    print(f"Database initialized successfully: {DATABASE_PATH}")


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
