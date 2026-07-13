"""Focused regression tests for registry archival and technical reporting."""

import csv
import io
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

BACKEND_DIR = Path(__file__).resolve().parents[1] / "web-app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from artwork_service import ArtworkService  # noqa: E402
from database import Artwork, Base, Verification  # noqa: E402
from report_service import ReportService  # noqa: E402
from verification_service import VerificationService  # noqa: E402
from verification_policy import VERIFICATION_POLICY  # noqa: E402


def make_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def add_artwork(db):
    artwork = Artwork(
        artwork_id="ART-0001",
        title="Long-lived provenance record",
        creator_name="Creator",
        original_filename="original.png",
        payload="0" * 32,
        watermark_status="embedded",
    )
    db.add(artwork)
    db.commit()
    return artwork


def test_archive_excludes_active_artwork_but_preserves_history_and_files(tmp_path):
    db = make_session()
    artwork = add_artwork(db)
    preserved_file = tmp_path / "watermarked.png"
    preserved_file.write_bytes(b"preserve me")
    artwork.watermarked_file_path = str(preserved_file)
    verification = Verification(
        verification_id="VER-0001",
        artwork_id=artwork.artwork_id,
        suspected_filename="repost.png",
        result_status="match",
        expected_payload="0" * 32,
        extracted_payload="0" * 32,
        ber=0.0,
        processing_time_ms=12.5,
        threshold_used=VERIFICATION_POLICY.detection_ber_threshold,
    )
    db.add(verification)
    db.commit()

    service = ArtworkService(str(tmp_path / "storage"))
    archived = service.archive_artwork(db, artwork.artwork_id)

    assert archived.archived_at is not None
    assert archived.watermark_status == "archived"
    assert service.get_all_artworks(db) == []
    assert service.get_active_artwork(db, artwork.artwork_id) is None
    assert service.get_artwork(db, artwork.artwork_id) is not None
    assert db.query(Verification).filter_by(verification_id="VER-0001").one()
    assert preserved_file.exists()
    assert service.archive_artwork(db, artwork.artwork_id) is None


def test_dashboard_counts_only_active_artworks():
    db = make_session()
    artwork = add_artwork(db)
    artwork.archived_at = artwork.registration_date
    artwork.watermark_status = "archived"
    db.commit()

    stats = VerificationService().get_dashboard_stats(db)

    assert stats["total_artworks"] == 0


def test_individual_report_contains_safe_payload_comparison_fields():
    db = make_session()
    artwork = add_artwork(db)
    db.add(Verification(
        verification_id="VER-0002",
        artwork_id=artwork.artwork_id,
        suspected_filename="changed.png",
        result_status="partial",
        expected_payload="0" * 32,
        extracted_payload="8" + "0" * 31,
        ber=1 / 128,
        processing_time_ms=8.25,
        threshold_used=VERIFICATION_POLICY.detection_ber_threshold,
    ))
    db.commit()

    report = ReportService.generate_verification_csv(db, "VER-0002")
    rows = dict(row for row in csv.reader(io.StringIO(report)) if len(row) == 2)

    assert rows["Expected Payload Fingerprint"].startswith("sha256:")
    assert rows["Extracted Payload Fingerprint"].startswith("sha256:")
    assert rows["Expected Payload Fingerprint"] != rows["Extracted Payload Fingerprint"]
    assert "0" * 32 not in report
    assert "8" + "0" * 31 not in report
    assert rows["Differing Bits"] == "1"
    assert rows["Payload Length"] == "128"
    assert rows["Watermark Engine"] == "DWT-QIM"
