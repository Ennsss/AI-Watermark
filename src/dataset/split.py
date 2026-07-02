"""Artist-disjoint, deterministic train/val/test split.

No artist appears in more than one partition, preventing the CNN decoder from
seeing an artist's style at train time and being tested on the same artist.

Determinism: artist groups are ordered by sha256(seed | artist) (NOT Python's
salted ``hash``), so the split is byte-reproducible for a fixed seed and input.

Assignment is greedy whole-group: each artist's images go entirely to one split.
A group is placed into the current split only if it does not overshoot that
split's target; otherwise it is held for a later split. Images with no artist tag
become singleton groups, which also act as exact-count fillers.
"""

from __future__ import annotations

import hashlib
from typing import Any

DEFAULT_QUOTAS = {"train": 10000, "val": 1000, "test": 500}


def _group_key(record: dict[str, Any], artist_key: str, id_key: str) -> str:
    artist = record.get(artist_key)
    if not artist:
        return f"__noartist__{record[id_key]}"
    return str(artist)


def _order_hash(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}|{group_key}".encode("utf-8")).hexdigest()


def split_artist_disjoint(
    records: list[dict[str, Any]],
    quotas: dict[str, int] | None = None,
    seed: int = 0,
    artist_key: str = "artist",
    id_key: str = "id",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], int]:
    """Partition records into artist-disjoint train/val/test sets.

    Returns (splits, counts, num_unused) where splits maps split name -> records,
    counts maps split name -> image count, and num_unused is the number of images
    left over after all quotas were met (or could not be filled without overshoot).

    Splits are filled in the order test, val, train so the small held-out sets are
    satisfied first.
    """
    quotas = quotas or DEFAULT_QUOTAS

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(_group_key(rec, artist_key, id_key), []).append(rec)

    ordered = sorted(groups.items(), key=lambda kv: _order_hash(seed, kv[0]))

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    counts: dict[str, int] = {name: 0 for name in quotas}

    remaining = ordered
    for name in ("test", "val", "train"):
        if name not in quotas:
            continue
        target = quotas[name]
        held: list[tuple[str, list[dict[str, Any]]]] = []
        for gk, recs in remaining:
            if counts[name] >= target or counts[name] + len(recs) > target:
                held.append((gk, recs))
                continue
            splits[name].extend(recs)
            counts[name] += len(recs)
        remaining = held

    num_unused = sum(len(recs) for _, recs in remaining)
    return splits, counts, num_unused


def assert_disjoint(splits: dict[str, list[dict[str, Any]]], artist_key: str = "artist") -> None:
    """Raise AssertionError if any real artist appears in more than one split."""
    artists_per_split = {}
    for name, recs in splits.items():
        artists_per_split[name] = {
            r[artist_key] for r in recs if r.get(artist_key)
        }
    names = list(artists_per_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = artists_per_split[names[i]] & artists_per_split[names[j]]
            assert not overlap, (
                f"Artist overlap between {names[i]} and {names[j]}: "
                f"{sorted(overlap)[:5]}"
            )
