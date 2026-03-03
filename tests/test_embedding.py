"""Tests for watermark.embedding — DWT, QIM, spread-spectrum embedding."""

import numpy as np
import pytest

from watermark.embedding import (
    _get_embedding_locations,
    dwt2_decompose,
    dwt2_reconstruct,
    embed_watermark,
    extract_watermark,
    qim_embed_bit,
    qim_extract_bit,
)


class TestQIM:
    @pytest.mark.parametrize("bit", [0, 1])
    def test_embed_extract_single(self, bit):
        coeff = 42.7
        delta = 20.0
        modified = qim_embed_bit(coeff, bit, delta)
        extracted = qim_extract_bit(modified, delta)
        assert extracted == bit

    @pytest.mark.parametrize("delta", [10.0, 20.0, 40.0, 60.0, 80.0])
    def test_robustness_within_quarter_delta(self, delta):
        for bit in [0, 1]:
            coeff = 100.0
            modified = qim_embed_bit(coeff, bit, delta)
            # Add noise less than delta/4
            noise = delta / 4 - 0.01
            assert qim_extract_bit(modified + noise, delta) == bit
            assert qim_extract_bit(modified - noise, delta) == bit

    def test_modification_is_bounded(self):
        coeff = 50.0
        delta = 20.0
        for bit in [0, 1]:
            modified = qim_embed_bit(coeff, bit, delta)
            assert abs(modified - coeff) <= delta


class TestDWT:
    def test_decompose_reconstruct(self):
        y = np.random.rand(512, 512) * 255
        coeffs = dwt2_decompose(y, wavelet="haar", level=2)
        reconstructed = dwt2_reconstruct(coeffs, wavelet="haar")
        assert np.allclose(y, reconstructed, atol=1e-10)

    def test_coeffs_structure(self):
        y = np.random.rand(64, 64) * 255
        coeffs = dwt2_decompose(y, wavelet="haar", level=2)
        # [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]
        assert len(coeffs) == 3
        assert len(coeffs[1]) == 3  # LH2, HL2, HH2
        assert len(coeffs[2]) == 3  # LH1, HL1, HH1

    def test_db4_wavelet(self):
        y = np.random.rand(128, 128) * 255
        coeffs = dwt2_decompose(y, wavelet="db4", level=2)
        reconstructed = dwt2_reconstruct(coeffs, wavelet="db4")
        assert np.allclose(y, reconstructed, atol=1e-10)


class TestEmbeddingLocations:
    def test_deterministic(self):
        locs1 = _get_embedding_locations((64, 64), 100, seed=42)
        locs2 = _get_embedding_locations((64, 64), 100, seed=42)
        assert np.array_equal(locs1, locs2)

    def test_different_seeds(self):
        locs1 = _get_embedding_locations((64, 64), 100, seed=42)
        locs2 = _get_embedding_locations((64, 64), 100, seed=99)
        assert not np.array_equal(locs1, locs2)

    def test_no_duplicates(self):
        locs = _get_embedding_locations((64, 64), 200, seed=42)
        flat = locs[:, 0] * 64 + locs[:, 1]
        assert len(set(flat)) == 200

    def test_too_many_bits_raises(self):
        with pytest.raises(ValueError):
            _get_embedding_locations((4, 4), 100, seed=42)


class TestEmbedExtract:
    def test_perfect_roundtrip(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (512, 512))
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)

        wm_y = embed_watermark(y, bits, seed=12345, delta=60.0)
        extracted = extract_watermark(wm_y, num_bits=256, seed=12345, delta=60.0)
        assert np.array_equal(bits, extracted)

    def test_different_payloads(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (256, 256))
        bits0 = np.zeros(128, dtype=np.uint8)
        bits1 = np.ones(128, dtype=np.uint8)

        wm0 = embed_watermark(y.copy(), bits0, seed=99, delta=60.0)
        wm1 = embed_watermark(y.copy(), bits1, seed=99, delta=60.0)

        ext0 = extract_watermark(wm0, 128, seed=99, delta=60.0)
        ext1 = extract_watermark(wm1, 128, seed=99, delta=60.0)

        assert np.array_equal(ext0, bits0)
        assert np.array_equal(ext1, bits1)


class TestLL2Fallback:
    def test_ll2_embed_extract_roundtrip(self):
        """LL2 subband embedding should produce a perfect roundtrip."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (256, 256))
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)

        wm_y = embed_watermark(
            y, bits, seed=12345, delta=60.0,
            target_subbands=("ll2",),
        )
        extracted = extract_watermark(
            wm_y, num_bits=128, seed=12345, delta=60.0,
            target_subbands=("ll2",),
        )
        assert np.array_equal(bits, extracted)

    def test_ll2_with_line_art_image(self, line_art_image):
        """LL2 works on sparse line art where LH2/HL2 would fail."""
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        y = line_art_image[:, :, 0].astype(np.float64)

        wm_y = embed_watermark(
            y, bits, seed=12345, delta=80.0,
            target_subbands=("ll2",),
        )
        extracted = extract_watermark(
            wm_y, num_bits=128, seed=12345, delta=80.0,
            target_subbands=("ll2",),
        )
        assert np.array_equal(bits, extracted)
