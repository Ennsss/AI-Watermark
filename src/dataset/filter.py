"""Metadata-stage inclusion/exclusion filtering for Safebooru records.

Pure logic over plain record dicts (one per image, as read from the deepghs
metadata parquet), so it is unit-testable without network or disk. Field names
match the ``deepghs/safebooru_full`` table schema:

    id, rating, width, height, mimetype, hash (=MD5), tags (flat space-separated)

Artist identity is derived from the global tag-type table (tags.parquet), where
tag ``type == 1`` denotes an artist tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

ARTIST_TAG_TYPE = 1

DEFAULT_WHITELIST = frozenset({
    # single / mixed-character
    "1girl", "1boy", "1other",
    # multi-character (same- and mixed-gender)
    "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "multiple_others",
    # non-character illustration content
    "scenery", "no_humans", "original", "illustration",
    "line_art", "monochrome", "cel_shading", "flat_color",
})
DEFAULT_BLACKLIST = frozenset({
    "photo", "3d", "screenshot", "comic", "multiple_views",
    "text_focus", "realistic",
})
DEFAULT_RATINGS = frozenset({"general", "safe"})
DEFAULT_MIMETYPES = frozenset({"image/jpeg", "image/png"})


@dataclass(frozen=True)
class FilterConfig:
    """Inclusion/exclusion parameters (defaults match the dataset spec)."""

    min_width: int = 512
    min_height: int = 512
    aspect_lo: float = 0.5
    aspect_hi: float = 2.0
    ratings: frozenset[str] = DEFAULT_RATINGS
    mimetypes: frozenset[str] = DEFAULT_MIMETYPES
    require_artist: bool = True
    whitelist: frozenset[str] = DEFAULT_WHITELIST
    blacklist: frozenset[str] = DEFAULT_BLACKLIST


# Stable list of reasons, used for per-filter drop accounting during a run.
DROP_REASONS = ("rating", "mimetype", "dimensions", "aspect", "blacklist",
                "whitelist", "artist")


def tag_tokens(tag_string: Any) -> set[str]:
    """Split a flat space-separated tag string into a token set."""
    if not tag_string:
        return set()
    return set(str(tag_string).split())


def extract_artists(tokens: set[str], artist_tags: set[str]) -> list[str]:
    """Return the sorted artist tags present in a token set."""
    return sorted(tokens & artist_tags)


def primary_artist(artists: list[str]) -> str | None:
    """The split key: the alphabetically-first artist, or None if none."""
    return artists[0] if artists else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_reason(record: dict[str, Any], artist_tags: set[str],
                  cfg: FilterConfig) -> str | None:
    """Return the first failing filter's name, or None if the record passes."""
    if record.get("rating") not in cfg.ratings:
        return "rating"
    if record.get("mimetype") not in cfg.mimetypes:
        return "mimetype"

    w = _as_float(record.get("width"))
    h = _as_float(record.get("height"))
    if w is None or h is None or w < cfg.min_width or h < cfg.min_height:
        return "dimensions"
    ratio = w / h
    if not (cfg.aspect_lo <= ratio <= cfg.aspect_hi):
        return "aspect"

    tokens = tag_tokens(record.get("tags"))
    if tokens & cfg.blacklist:
        return "blacklist"
    if cfg.whitelist and not (tokens & cfg.whitelist):
        return "whitelist"
    if cfg.require_artist and not (tokens & artist_tags):
        return "artist"
    return None


def passes(record: dict[str, Any], artist_tags: set[str], cfg: FilterConfig) -> bool:
    """True if a record passes all inclusion/exclusion filters."""
    return filter_reason(record, artist_tags, cfg) is None


def to_candidate(record: dict[str, Any], artist_tags: set[str]) -> dict[str, Any]:
    """Normalize a passing record into a candidate dict for downstream stages."""
    tokens = tag_tokens(record.get("tags"))
    artists = extract_artists(tokens, artist_tags)
    return {
        "id": int(record["id"]),
        "artist": primary_artist(artists),
        "artists": artists,
        "md5": record.get("hash"),
        "width": _as_float(record.get("width")),
        "height": _as_float(record.get("height")),
        "mimetype": record.get("mimetype"),
        "rating": record.get("rating"),
    }


def select(records: Iterable[dict[str, Any]], artist_tags: set[str],
           cfg: FilterConfig, limit: int | None = None
           ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Filter an iterable of records into candidates.

    Returns (candidates, drop_counts, n_scanned). Stops early once ``limit``
    candidates are collected (if limit is given).
    """
    candidates: list[dict[str, Any]] = []
    drops = {r: 0 for r in DROP_REASONS}
    scanned = 0
    for rec in records:
        scanned += 1
        reason = filter_reason(rec, artist_tags, cfg)
        if reason is None:
            candidates.append(to_candidate(rec, artist_tags))
            if limit is not None and len(candidates) >= limit:
                break
        else:
            drops[reason] += 1
    return candidates, drops, scanned
