"""Generate labeled BER observations without mutating the registry database."""

import argparse
import csv
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))
from watermark.embedding import extract_watermark  # noqa: E402
from watermark.preprocessor import rgb_to_ycbcr, extract_y_channel, pad_to_multiple  # noqa: E402

PARAMS = {"delta": 16.0, "wavelet": "haar", "level": 2, "subbands": ("lh2", "hl2"), "seed": 42, "payload_bits": 128}


def resolve(stored, filename, folder):
    path = Path(stored) if stored else None
    if path and path.exists(): return path
    fallback = BACKEND_DIR / "storage" / folder / filename if filename else None
    return fallback if fallback and fallback.exists() else None


def decode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def condition_image(image, condition):
    if condition == "clean_watermarked": return image.copy()
    if condition.startswith("jpeg_"):
        quality = int(condition.split("_")[1])
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
    if condition.startswith("resize_"):
        scale = int(condition.split("_")[1]) / 100
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if condition == "png_resave":
        ok, encoded = cv2.imencode(".png", image)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
    return image.copy()


def extract_hex(image):
    normalized = cv2.resize(image, (512, 512)) if image.shape[:2] != (512, 512) else image
    rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    y = extract_y_channel(rgb_to_ycbcr(rgb))
    padded, _ = pad_to_multiple(y, multiple=2 ** PARAMS["level"])
    bits = extract_watermark(padded, num_bits=PARAMS["payload_bits"], seed=PARAMS["seed"], delta=PARAMS["delta"], wavelet=PARAMS["wavelet"], level=PARAMS["level"], target_subbands=PARAMS["subbands"])
    return np.packbits(bits).tobytes().hex(), bits, image.shape[1], image.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=BACKEND_DIR / "watermark_registry.db")
    parser.add_argument("--output", type=Path, default=BACKEND_DIR / "calibration" / "controlled_calibration_observations.csv")
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    artworks = connection.execute("SELECT * FROM artworks WHERE archived_at IS NULL AND watermark_status != 'archived' ORDER BY artwork_id").fetchall()
    usable = []
    for row in artworks:
        watermarked = resolve(row["watermarked_file_path"], row["watermarked_filename"], "watermarked")
        original = resolve(row["original_file_path"], row["original_filename"], "originals")
        if watermarked: usable.append((row, watermarked, original))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip() or "unknown"
    fields = ["sample_id", "artwork_id", "condition", "ground_truth", "expected_payload", "extracted_payload", "differing_bits", "ber", "delta", "wavelet", "dwt_level", "target_subbands", "payload_length", "coefficient_seed", "source_image", "source_width", "source_height", "normalized_width", "normalized_height", "processing_timestamp", "code_commit", "endpoint_semantics"]
    observations = []

    def observe(row, source, image, condition, truth):
        extracted, bits, width, height = extract_hex(image)
        expected = np.unpackbits(np.frombuffer(bytes.fromhex(row["payload"]), dtype=np.uint8))
        diff = int(np.sum(bits != expected))
        observations.append({
            "sample_id": f"CAL-{len(observations)+1:04d}", "artwork_id": row["artwork_id"], "condition": condition,
            "ground_truth": truth, "expected_payload": row["payload"], "extracted_payload": extracted,
            "differing_bits": diff, "ber": diff / PARAMS["payload_bits"], "delta": PARAMS["delta"],
            "wavelet": PARAMS["wavelet"], "dwt_level": PARAMS["level"], "target_subbands": "+".join(PARAMS["subbands"]),
            "payload_length": PARAMS["payload_bits"], "coefficient_seed": PARAMS["seed"], "source_image": str(source),
            "source_width": width, "source_height": height, "normalized_width": 512, "normalized_height": 512,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(), "code_commit": commit,
            "endpoint_semantics": "selected_record_equivalent",
        })

    conditions = ["clean_watermarked", "jpeg_90", "jpeg_70", "jpeg_50", "resize_75", "resize_50", "png_resave"]
    for row, watermarked, original in usable:
        image = decode(watermarked)
        if image is not None:
            for condition in conditions:
                transformed = condition_image(image, condition)
                if transformed is not None: observe(row, watermarked, transformed, condition, "positive")
        if original:
            original_image = decode(original)
            if original_image is not None: observe(row, original, original_image, "non_watermarked", "negative")
    if len(usable) > 1:
        for index, (row, _, _) in enumerate(usable):
            wrong_source = usable[(index + 1) % len(usable)][1]
            wrong_image = decode(wrong_source)
            if wrong_image is not None: observe(row, wrong_source, wrong_image, "wrong_artwork_record", "negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(observations)
    print(f"Wrote {len(observations)} controlled observations from {len(usable)} artworks to {args.output.resolve()}")


if __name__ == "__main__":
    main()
