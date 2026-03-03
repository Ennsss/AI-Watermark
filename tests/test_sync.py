"""Tests for watermark.sync — PN sync signal embedding and crop offset detection."""

import numpy as np
import pytest

from watermark.sync import (
    _detect_offset_mod,
    detect_crop_offset,
    embed_sync_pattern,
    generate_sync_sequences,
)


class TestSyncSequences:
    def test_deterministic(self):
        """Same seed produces same sequences."""
        s1h, s1v = generate_sync_sequences(42)
        s2h, s2v = generate_sync_sequences(42)
        assert np.array_equal(s1h, s2h)
        assert np.array_equal(s1v, s2v)

    def test_different_seeds(self):
        """Different seeds produce different sequences."""
        s1h, s1v = generate_sync_sequences(42)
        s2h, s2v = generate_sync_sequences(99)
        assert not np.array_equal(s1h, s2h)

    def test_shape_and_values(self):
        """Sequences should be +/-1 with correct length."""
        seq_h, seq_v = generate_sync_sequences(42, sync_period=64)
        assert seq_h.shape == (64,)
        assert seq_v.shape == (64,)
        assert set(seq_h.tolist()).issubset({-1.0, 1.0})
        assert set(seq_v.tolist()).issubset({-1.0, 1.0})


class TestEmbedSync:
    def test_output_shape(self):
        """Embedding preserves shape."""
        y = np.random.default_rng(42).uniform(50, 200, (256, 256))
        result = embed_sync_pattern(y, seed=42)
        assert result.shape == y.shape

    def test_low_distortion(self):
        """Sync pattern should cause minimal pixel distortion."""
        y = np.random.default_rng(42).uniform(50, 200, (256, 256))
        result = embed_sync_pattern(y, seed=42, amplitude=3.0)
        max_diff = np.max(np.abs(result - y))
        # Max diff is 2 * amplitude (horizontal + vertical)
        assert max_diff <= 6.1  # 3.0 + 3.0 + small float tolerance


class TestCropOffsetDetection:
    @pytest.mark.parametrize("crop_h,crop_w", [
        (0, 0),
        (10, 0),
        (0, 15),
        (37, 23),
        (100, 50),
    ])
    def test_detects_crop_offset(self, crop_h, crop_w):
        """Sync signal correctly detects known crop offsets."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (1024, 1024))
        seed = 12345
        sync_period = 32
        tile_size = 256

        # Embed sync
        y_sync = embed_sync_pattern(y, seed=seed, sync_period=sync_period)

        # Crop
        cropped = y_sync[crop_h:, crop_w:]

        # Detect
        candidates = detect_crop_offset(
            cropped, seed=seed,
            sync_period=sync_period,
            tile_size=tile_size,
        )

        # Expected tile alignment offset
        expected_r = (tile_size - crop_h % tile_size) % tile_size
        expected_c = (tile_size - crop_w % tile_size) % tile_size

        # The correct offset should be among the candidates
        assert (expected_r, expected_c) in candidates, (
            f"Expected ({expected_r}, {expected_c}) not in candidates: {candidates[:5]}..."
        )

    def test_survives_noise(self):
        """Sync detection works with moderate noise added."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (1024, 1024))
        seed = 12345

        y_sync = embed_sync_pattern(y, seed=seed, sync_period=32, amplitude=3.0)

        # Add noise
        noisy = y_sync + rng.normal(0, 5, y_sync.shape)

        # Crop
        crop_h, crop_w = 45, 67
        cropped = noisy[crop_h:, crop_w:]

        candidates = detect_crop_offset(
            cropped, seed=seed, sync_period=32, tile_size=256,
        )

        expected_r = (256 - crop_h % 256) % 256
        expected_c = (256 - crop_w % 256) % 256
        assert (expected_r, expected_c) in candidates

    def test_empty_candidates_for_large_crop(self):
        """Very large crop that leaves no room for tiles returns empty list."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (300, 300))
        seed = 12345

        y_sync = embed_sync_pattern(y, seed=seed)

        # Crop to less than tile_size
        cropped = y_sync[:200, :200]

        candidates = detect_crop_offset(
            cropped, seed=seed, tile_size=256,
        )
        # No 256x256 tile fits in 200x200
        assert len(candidates) == 0
