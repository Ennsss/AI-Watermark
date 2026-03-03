"""Tests for attacks.suite — all 7 attack types."""

import numpy as np
import pytest

from attacks.suite import (
    combined_chain,
    format_conversion,
    gaussian_noise,
    get_default_attacks,
    jpeg_compression,
    random_crop,
    resize_attack,
    run_attack,
    screenshot_simulation,
)


@pytest.fixture
def test_image():
    rng = np.random.default_rng(42)
    return rng.integers(30, 225, (256, 256, 3), dtype=np.uint8)


class TestJPEG:
    @pytest.mark.parametrize("quality", [50, 70, 90])
    def test_output_shape_preserved(self, test_image, quality):
        result = jpeg_compression(test_image, quality)
        assert result.image.shape == test_image.shape
        assert result.image.dtype == np.uint8

    def test_higher_quality_closer(self, test_image):
        r50 = jpeg_compression(test_image, 50)
        r90 = jpeg_compression(test_image, 90)
        diff50 = np.mean(np.abs(test_image.astype(float) - r50.image.astype(float)))
        diff90 = np.mean(np.abs(test_image.astype(float) - r90.image.astype(float)))
        assert diff90 < diff50


class TestResize:
    def test_downscale(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = resize_attack(img, max_dim=1080)
        assert max(result.image.shape[:2]) == 1080

    def test_noop_if_smaller(self, test_image):
        result = resize_attack(test_image, max_dim=1080)
        assert result.image.shape == test_image.shape


class TestCrop:
    def test_reduces_size(self, test_image):
        result = random_crop(test_image, crop_ratio=0.2, seed=42)
        assert result.image.shape[0] < test_image.shape[0] or \
               result.image.shape[1] < test_image.shape[1]

    def test_deterministic(self, test_image):
        r1 = random_crop(test_image, crop_ratio=0.2, seed=42)
        r2 = random_crop(test_image, crop_ratio=0.2, seed=42)
        assert np.array_equal(r1.image, r2.image)


class TestScreenshot:
    def test_output_shape(self, test_image):
        result = screenshot_simulation(test_image)
        assert result.image.shape == test_image.shape


class TestFormatConversion:
    def test_output_shape(self, test_image):
        result = format_conversion(test_image)
        assert result.image.shape == test_image.shape


class TestGaussianNoise:
    def test_adds_noise(self, test_image):
        result = gaussian_noise(test_image, sigma=10, seed=42)
        assert not np.array_equal(result.image, test_image)
        assert result.image.shape == test_image.shape

    def test_deterministic(self, test_image):
        r1 = gaussian_noise(test_image, sigma=5, seed=42)
        r2 = gaussian_noise(test_image, sigma=5, seed=42)
        assert np.array_equal(r1.image, r2.image)


class TestCombinedChain:
    def test_runs_without_error(self, test_image):
        result = combined_chain(test_image, max_dim=200, seed=42)
        assert result.image.ndim == 3


class TestDefaultAttacks:
    def test_all_attacks_run(self, test_image):
        attacks = get_default_attacks()
        assert len(attacks) >= 15
        for name, fn, kwargs in attacks:
            result = fn(test_image, **kwargs)
            assert result.image.ndim == 3
            assert result.image.dtype == np.uint8


class TestRunAttack:
    def test_dispatch(self, test_image):
        result = run_attack(test_image, "jpeg", quality=80)
        assert result.image.shape == test_image.shape

    def test_unknown_attack_raises(self, test_image):
        with pytest.raises(ValueError):
            run_attack(test_image, "nonexistent")
