"""Tests for watermark.preprocessor — color conversion, padding, I/O."""

import numpy as np
import pytest

from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    replace_y_channel,
    rgb_to_ycbcr,
    unpad,
    ycbcr_to_rgb,
)


class TestColorConversion:
    def test_roundtrip_preserves_shape(self, sample_rgb_image):
        ycbcr = rgb_to_ycbcr(sample_rgb_image)
        assert ycbcr.shape == sample_rgb_image.shape
        assert ycbcr.dtype == np.float64

    def test_roundtrip_close_to_original(self, sample_rgb_image):
        ycbcr = rgb_to_ycbcr(sample_rgb_image)
        recovered = ycbcr_to_rgb(ycbcr)
        # Allow ±1 for rounding
        assert np.allclose(sample_rgb_image, recovered, atol=1)

    def test_y_channel_range(self, sample_rgb_image):
        ycbcr = rgb_to_ycbcr(sample_rgb_image)
        y = ycbcr[:, :, 0]
        assert y.min() >= 0
        assert y.max() <= 255

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            rgb_to_ycbcr(np.zeros((10, 10), dtype=np.uint8))


class TestYChannel:
    def test_extract_and_replace(self, sample_rgb_image):
        ycbcr = rgb_to_ycbcr(sample_rgb_image)
        y = extract_y_channel(ycbcr)
        assert y.shape == sample_rgb_image.shape[:2]

        modified_y = y + 5.0
        result = replace_y_channel(ycbcr, modified_y)
        assert np.allclose(result[:, :, 0], modified_y)
        # Cb, Cr unchanged
        assert np.array_equal(result[:, :, 1], ycbcr[:, :, 1])
        assert np.array_equal(result[:, :, 2], ycbcr[:, :, 2])


class TestPadding:
    def test_already_aligned(self):
        img = np.zeros((512, 512), dtype=np.float64)
        padded, pad_sizes = pad_to_multiple(img, 4)
        assert pad_sizes == (0, 0)
        assert padded.shape == (512, 512)

    def test_needs_padding(self):
        img = np.zeros((510, 511), dtype=np.float64)
        padded, pad_sizes = pad_to_multiple(img, 4)
        assert padded.shape[0] % 4 == 0
        assert padded.shape[1] % 4 == 0
        assert pad_sizes == (2, 1)

    def test_pad_unpad_roundtrip(self):
        img = np.random.rand(100, 103)
        padded, pad_sizes = pad_to_multiple(img, 4)
        recovered = unpad(padded, pad_sizes)
        assert recovered.shape == img.shape
        assert np.allclose(recovered, img)

    def test_3d_padding(self, sample_rgb_image):
        img = sample_rgb_image[:510, :511, :]
        padded, pad_sizes = pad_to_multiple(img, 4)
        assert padded.shape[0] % 4 == 0
        assert padded.shape[1] % 4 == 0
        recovered = unpad(padded, pad_sizes)
        assert recovered.shape == img.shape
