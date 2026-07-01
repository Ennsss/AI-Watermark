"""Deduplication: exact (MD5) then near-duplicate (perceptual hash).

Exact duplicates are detected with the canonical Danbooru per-file MD5 (taken
from metadata, so originals never need to be retained). Near-duplicates are
detected with a 64-bit perceptual hash; two images are considered near-duplicate
when their Hamming distance is <= threshold (default 6 / 64, per the dataset spec).

Records are plain dicts. Order is preserved; the first occurrence of a group is
kept and later members are dropped. Near-duplicate removal is greedy and seeded
by input order, so the result is deterministic for a fixed input order.
"""

from __future__ import annotations

from typing import Any

import imagehash
import numpy as np
from PIL import Image

NEAR_DUP_THRESHOLD = 6


def compute_phash(image: Image.Image) -> int:
    """Return the 8x8 DCT perceptual hash of an image as a 64-bit integer."""
    h = imagehash.phash(image)  # 8x8 -> 64 bits
    return int(str(h), 16)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integer hashes."""
    return int(bin(a ^ b).count("1"))


def remove_exact_duplicates(
    records: list[dict[str, Any]],
    key: str = "md5",
) -> tuple[list[dict[str, Any]], int]:
    """Keep the first record per unique ``key`` value.

    Returns (kept_records, num_removed). Records missing the key are kept as-is
    (treated as unique).
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for rec in records:
        val = rec.get(key)
        if val is None:
            kept.append(rec)
            continue
        if val in seen:
            removed += 1
            continue
        seen.add(val)
        kept.append(rec)
    return kept, removed


def remove_near_duplicates(
    records: list[dict[str, Any]],
    phash_key: str = "phash",
    threshold: int = NEAR_DUP_THRESHOLD,
) -> tuple[list[dict[str, Any]], int]:
    """Greedily drop records whose phash is within ``threshold`` of a kept one.

    Uses vectorized popcount (numpy.bitwise_count) against the running set of kept
    hashes, so the cost is O(n * kept) with cheap inner operations.

    Returns (kept_records, num_removed).
    """
    kept: list[dict[str, Any]] = []
    kept_hashes = np.empty(0, dtype=np.uint64)
    removed = 0
    for rec in records:
        ph = rec.get(phash_key)
        if ph is None:
            kept.append(rec)
            continue
        cand = np.uint64(int(ph))
        if kept_hashes.size:
            dists = np.bitwise_count(kept_hashes ^ cand)
            if dists.min() <= threshold:
                removed += 1
                continue
        kept.append(rec)
        kept_hashes = np.append(kept_hashes, cand)
    return kept, removed
