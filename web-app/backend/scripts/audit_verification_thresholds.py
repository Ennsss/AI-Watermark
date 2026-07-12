"""Read-only audit of stored verification math and classification consistency."""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from verification_policy import VERIFICATION_POLICY  # noqa: E402


def differing_bits(expected_hex, extracted_hex):
    if not expected_hex or not extracted_hex:
        return None
    try:
        expected = bin(int(expected_hex, 16))[2:].zfill(len(expected_hex) * 4)
        extracted = bin(int(extracted_hex, 16))[2:].zfill(len(extracted_hex) * 4)
    except ValueError:
        return None
    return sum(a != b for a, b in zip(expected, extracted)) + abs(len(expected) - len(extracted))


def status_for(ber, threshold, error):
    if error:
        return "error"
    if ber is None:
        return "extraction_failed"
    if ber == 0:
        return "match"
    return "partial" if threshold is not None and ber <= threshold else "no_match"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=BACKEND_DIR / "watermark_registry.db")
    parser.add_argument("--output", type=Path, default=BACKEND_DIR / "calibration" / "verification_threshold_audit.csv")
    parser.add_argument("--labels", type=Path, default=BACKEND_DIR / "calibration" / "verification_ground_truth.csv")
    args = parser.parse_args()
    database = args.database.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT * FROM verifications ORDER BY verification_date").fetchall()
    labels = {}
    if args.labels.exists():
        with args.labels.open(newline="", encoding="utf-8") as handle:
            labels = {row["verification_id"]: row for row in csv.DictReader(handle)}

    fields = [
        "verification_id", "artwork_id", "suspected_filename", "verification_date",
        "stored_result_status", "ber", "threshold_used", "expected_payload",
        "extracted_payload", "differing_bits", "payload_length", "processing_time_ms",
        "error_message", "source_database_path", "recomputed_differing_bits",
        "recomputed_ber", "status_under_stored_rule", "status_under_current_rule",
        "status_consistent", "ber_consistent", "threshold_consistent", "payload_lengths_match",
        "representable_128_bit_ber", "issues", "ground_truth", "condition", "notes",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            diff = differing_bits(row["expected_payload"], row["extracted_payload"])
            expected_len = len(row["expected_payload"]) * 4 if row["expected_payload"] else None
            extracted_len = len(row["extracted_payload"]) * 4 if row["extracted_payload"] else None
            recomputed_ber = diff / expected_len if diff is not None and expected_len else None
            stored_rule = status_for(row["ber"], row["threshold_used"], row["error_message"])
            current_rule = VERIFICATION_POLICY.classify(row["ber"], row["error_message"])[0]
            ber_ok = (row["ber"] is None and recomputed_ber is None) or (
                row["ber"] is not None and recomputed_ber is not None and abs(row["ber"] - recomputed_ber) < 1e-12
            )
            representable = row["ber"] is None or abs(row["ber"] * 128 - round(row["ber"] * 128)) < 1e-9
            issues = []
            if not ber_ok: issues.append("stored_ber_mismatch")
            if row["result_status"] != stored_rule: issues.append("stored_status_mismatch")
            if row["threshold_used"] is None: issues.append("missing_threshold")
            if row["ber"] is not None and not row["extracted_payload"]: issues.append("ber_without_extracted_payload")
            if expected_len != extracted_len: issues.append("payload_length_mismatch")
            if not representable: issues.append("ber_not_multiple_of_1_over_128")
            label = labels.get(row["verification_id"], {})
            writer.writerow({
                "verification_id": row["verification_id"], "artwork_id": row["artwork_id"],
                "suspected_filename": row["suspected_filename"], "verification_date": row["verification_date"],
                "stored_result_status": row["result_status"], "ber": row["ber"],
                "threshold_used": row["threshold_used"], "expected_payload": row["expected_payload"],
                "extracted_payload": row["extracted_payload"], "differing_bits": diff,
                "payload_length": expected_len, "processing_time_ms": row["processing_time_ms"],
                "error_message": row["error_message"], "source_database_path": str(database),
                "recomputed_differing_bits": diff, "recomputed_ber": recomputed_ber,
                "status_under_stored_rule": stored_rule, "status_under_current_rule": current_rule,
                "status_consistent": row["result_status"] == stored_rule, "ber_consistent": ber_ok,
                "threshold_consistent": row["threshold_used"] == VERIFICATION_POLICY.detection_ber_threshold,
                "payload_lengths_match": expected_len == extracted_len, "representable_128_bit_ber": representable,
                "issues": ";".join(issues), "ground_truth": label.get("ground_truth", "unknown"),
                "condition": label.get("condition", ""), "notes": label.get("notes", ""),
            })
    print(f"Audited {len(rows)} rows from {database}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
