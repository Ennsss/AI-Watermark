"""Reproduce the exact curated dataset from the committed manifest.

A teammate with deepghs gate access (see README) runs this to regenerate the
identical 10,000/1,000/500 dataset locally, without re-running selection/dedup/
split. It reads data/manifests/dataset_manifest.csv (id -> split) and downloads +
preprocesses each image into data/curated/{split}/{id}.png.

Because preprocessing is deterministic (alpha-flatten -> center-crop -> Lanczos
512), the regenerated PNGs match the originals. Re-runnable / resumable: images
already present are skipped.

Prereqs:
    pip install cheesechaser
    huggingface-cli login            # account must have deepghs/safebooru_full access
Usage:
    python scripts/download_from_ids.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import acquire

DATA = ROOT / "data"
MANIFEST = DATA / "manifests" / "dataset_manifest.csv"
CURATED = DATA / "curated"


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}\n"
                 "Commit data/manifests/ to the repo first.")

    by_split: dict[str, list[dict]] = defaultdict(list)
    with MANIFEST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_split[row["split"]].append({"id": int(row["id"])})

    total = sum(len(v) for v in by_split.values())
    print(f"Manifest: {total} images across {dict((k, len(v)) for k, v in by_split.items())}")

    for split, cands in by_split.items():
        out = CURATED / split
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split}] downloading {len(cands)} images into {out} ...")
        _, stats = acquire.download_and_preprocess(cands, out, chunk_size=250)
        print(f"[{split}] {stats}")

    # report
    print("\n=== result ===")
    ok = True
    for split, cands in by_split.items():
        have = len(list((CURATED / split).glob("*.png")))
        flag = "OK" if have == len(cands) else "MISMATCH"
        ok = ok and have == len(cands)
        print(f"  {split}: {have}/{len(cands)} {flag}")
    print("DONE" if ok else "INCOMPLETE - re-run to resume missing images")


if __name__ == "__main__":
    main()
