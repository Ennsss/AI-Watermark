"""Tests for watermark.extraction — full extraction pipeline, BER, confidence."""

import numpy as np
import pytest

from watermark.extraction import compute_ber, extract_from_image
from watermark.embedding import embed_watermark
from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    rgb_to_ycbcr,
)
from watermark.reconstruction import reconstruct_image


class TestExtractFromImage:
    def test_perfect_roundtrip(self, sample_rgb_image):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)

        ycbcr = rgb_to_ycbcr(sample_rgb_image)
        y = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y, 4)
        ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

        wm_y = embed_watermark(y_padded, bits, seed=42, delta=40.0)
        wm_rgb = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

        extracted, confidence = extract_from_image(
            wm_rgb, num_bits=256, seed=42, delta=40.0,
        )
        assert np.array_equal(bits, extracted)
        assert confidence > 0.9

    def test_small_image(self, small_rgb_image):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=64, dtype=np.uint8)

        ycbcr = rgb_to_ycbcr(small_rgb_image)
        y = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y, 4)
        ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

        wm_y = embed_watermark(y_padded, bits, seed=42, delta=40.0)
        wm_rgb = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

        extracted, confidence = extract_from_image(
            wm_rgb, num_bits=64, seed=42, delta=40.0,
        )
        ber = compute_ber(bits, extracted)
        assert ber < 0.1


class TestBER:
    def test_identical_bits(self):
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        assert compute_ber(bits, bits) == 0.0

    def test_all_different(self):
        a = np.array([0, 0, 0, 0], dtype=np.uint8)
        b = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(a, b) == 1.0

    def test_half_errors(self):
        a = np.array([0, 0, 1, 1], dtype=np.uint8)
        b = np.array([0, 1, 1, 0], dtype=np.uint8)
        assert compute_ber(a, b) == 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_ber(np.array([0, 1]), np.array([0]))
