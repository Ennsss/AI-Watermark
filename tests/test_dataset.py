"""Tests for the schema-independent dataset modules: preprocess, dedup, split."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.dataset import dedup, filter as dfilter, preprocess, split


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #

def test_preprocess_outputs_512_rgb():
    im = Image.new("RGB", (800, 400), (10, 20, 30))
    arr = preprocess.preprocess_image(im)
    assert arr.shape == (512, 512, 3)
    assert arr.dtype == np.uint8


def test_center_crop_square_takes_shorter_side():
    arr = np.zeros((400, 800, 3), dtype=np.uint8)
    cropped = preprocess.center_crop_square(arr)
    assert cropped.shape == (400, 400, 3)


def test_flatten_alpha_composites_onto_white():
    # Fully transparent image -> should become pure white after flatten.
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    out = preprocess.flatten_alpha(rgba, bg=(255, 255, 255))
    assert out.mode == "RGB"
    assert np.asarray(out).min() == 255


def test_flatten_alpha_preserves_opaque_pixels():
    rgba = Image.new("RGBA", (64, 64), (12, 34, 56, 255))
    out = np.asarray(preprocess.flatten_alpha(rgba))
    assert tuple(out[0, 0]) == (12, 34, 56)


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #

def test_remove_exact_duplicates_keeps_first():
    records = [
        {"id": 1, "md5": "aaa"},
        {"id": 2, "md5": "bbb"},
        {"id": 3, "md5": "aaa"},  # dup of id 1
    ]
    kept, removed = dedup.remove_exact_duplicates(records)
    assert removed == 1
    assert [r["id"] for r in kept] == [1, 2]


def test_hamming_distance():
    assert dedup.hamming_distance(0b1010, 0b1000) == 1
    assert dedup.hamming_distance(0xFFFFFFFFFFFFFFFF, 0) == 64


def test_remove_near_duplicates_drops_identical_phash():
    records = [
        {"id": 1, "phash": 0x0F0F0F0F0F0F0F0F},
        {"id": 2, "phash": 0x0F0F0F0F0F0F0F0F},  # distance 0 -> near dup
        {"id": 3, "phash": 0xF0F0F0F0F0F0F0F0},  # distance 64 -> keep
    ]
    kept, removed = dedup.remove_near_duplicates(records, threshold=6)
    assert removed == 1
    assert {r["id"] for r in kept} == {1, 3}


def test_remove_near_duplicates_respects_threshold():
    # distance of exactly 6 -> removed; distance 7 -> kept
    base = 0
    near = (1 << 6) - 1  # six set bits -> distance 6
    far = (1 << 7) - 1   # seven set bits -> distance 7
    kept6, removed6 = dedup.remove_near_duplicates(
        [{"id": 1, "phash": base}, {"id": 2, "phash": near}], threshold=6)
    assert removed6 == 1
    kept7, removed7 = dedup.remove_near_duplicates(
        [{"id": 1, "phash": base}, {"id": 2, "phash": far}], threshold=6)
    assert removed7 == 0


def test_phash_identical_images_match():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
    assert dedup.compute_phash(img) == dedup.compute_phash(img.copy())


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #

def _make_records():
    records = []
    rid = 0
    # 10 artists, 5 images each
    for a in range(10):
        for _ in range(5):
            records.append({"id": rid, "artist": f"artist_{a}"})
            rid += 1
    # 20 artist-less singletons (act as fillers)
    for _ in range(20):
        records.append({"id": rid, "artist": ""})
        rid += 1
    return records


def test_split_is_artist_disjoint():
    records = _make_records()
    splits, counts, _ = split.split_artist_disjoint(
        records, quotas={"train": 30, "val": 10, "test": 10}, seed=42)
    split.assert_disjoint(splits)  # raises if any artist crosses splits


def test_split_hits_quotas_with_fillers():
    records = _make_records()
    quotas = {"train": 30, "val": 10, "test": 10}
    splits, counts, unused = split.split_artist_disjoint(records, quotas=quotas, seed=42)
    # singleton fillers make exact counts achievable
    assert counts["test"] == 10
    assert counts["val"] == 10
    assert counts["train"] == 30


def test_split_is_deterministic():
    records = _make_records()
    q = {"train": 30, "val": 10, "test": 10}
    a, _, _ = split.split_artist_disjoint(records, quotas=q, seed=7)
    b, _, _ = split.split_artist_disjoint(records, quotas=q, seed=7)
    assert [r["id"] for r in a["test"]] == [r["id"] for r in b["test"]]


def test_split_seed_changes_partition():
    records = _make_records()
    q = {"train": 30, "val": 10, "test": 10}
    a, _, _ = split.split_artist_disjoint(records, quotas=q, seed=1)
    b, _, _ = split.split_artist_disjoint(records, quotas=q, seed=2)
    assert [r["id"] for r in a["test"]] != [r["id"] for r in b["test"]]


# --------------------------------------------------------------------------- #
# filter
# --------------------------------------------------------------------------- #

ARTIST_TAGS = {"artist_a", "artist_b"}


def _rec(**kw):
    base = {
        "id": 1, "rating": "general", "mimetype": "image/jpeg",
        "width": 1000, "height": 1000, "hash": "abc",
        "tags": " 1girl artist_a",
    }
    base.update(kw)
    return base


def test_filter_passes_good_record():
    assert dfilter.passes(_rec(), ARTIST_TAGS, dfilter.FilterConfig())


def test_filter_rejects_bad_rating():
    assert dfilter.filter_reason(_rec(rating="explicit"), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "rating"


def test_filter_rejects_gif():
    assert dfilter.filter_reason(_rec(mimetype="image/gif"), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "mimetype"


def test_filter_rejects_small():
    assert dfilter.filter_reason(_rec(width=400, height=400), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "dimensions"


def test_filter_rejects_extreme_aspect():
    assert dfilter.filter_reason(_rec(width=3000, height=1000), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "aspect"


def test_filter_rejects_blacklist():
    assert dfilter.filter_reason(_rec(tags=" 1girl artist_a photo"), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "blacklist"


def test_filter_rejects_no_whitelist_tag():
    assert dfilter.filter_reason(_rec(tags=" artist_a flower"), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "whitelist"


def test_filter_accepts_multi_character_after_broadening():
    # same-gender multi-character art should now pass the whitelist
    assert dfilter.passes(_rec(tags=" 2girls artist_a"), ARTIST_TAGS,
                          dfilter.FilterConfig())


def test_filter_rejects_no_artist_when_required():
    assert dfilter.filter_reason(_rec(tags=" 1girl flower"), ARTIST_TAGS,
                                 dfilter.FilterConfig()) == "artist"


def test_filter_allows_no_artist_when_not_required():
    cfg = dfilter.FilterConfig(require_artist=False)
    assert dfilter.passes(_rec(tags=" 1girl flower"), ARTIST_TAGS, cfg)


def test_extract_and_primary_artist():
    toks = dfilter.tag_tokens(" 1girl artist_b artist_a")
    artists = dfilter.extract_artists(toks, ARTIST_TAGS)
    assert artists == ["artist_a", "artist_b"]
    assert dfilter.primary_artist(artists) == "artist_a"


def test_select_collects_and_counts_drops():
    records = [
        _rec(id=1),                                  # pass
        _rec(id=2, rating="explicit"),               # drop: rating
        _rec(id=3, tags=" 1girl flower"),            # drop: artist
        _rec(id=4),                                  # pass
    ]
    cands, drops, scanned = dfilter.select(records, ARTIST_TAGS, dfilter.FilterConfig())
    assert scanned == 4
    assert [c["id"] for c in cands] == [1, 4]
    assert drops["rating"] == 1 and drops["artist"] == 1


def test_to_candidate_shape():
    c = dfilter.to_candidate(_rec(tags=" 1girl artist_a"), ARTIST_TAGS)
    assert c["id"] == 1 and c["artist"] == "artist_a" and c["md5"] == "abc"
