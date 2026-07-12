"""Tests for watermark.masking — local variance, adaptive delta."""

import numpy as np

from watermark.masking import build_delta_map, compute_adaptive_delta, compute_local_variance


class TestLocalVariance:
    def test_uniform_image_zero_variance(self):
        flat = np.ones((64, 64)) * 100.0
        var = compute_local_variance(flat, window_size=5)
        assert np.allclose(var, 0, atol=1e-10)

    def test_noisy_image_positive_variance(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(100, 20, (64, 64))
        var = compute_local_variance(noisy, window_size=5)
        assert var.mean() > 100  # Expect variance around 400 (sigma^2)

    def test_output_shape(self):
        img = np.random.rand(100, 80)
        var = compute_local_variance(img, window_size=7)
        assert var.shape == (100, 80)


class TestAdaptiveDelta:
    def test_flat_region_gets_low_delta(self):
        # Image with flat left and noisy right
        img = np.zeros((64, 128))
        img[:, 64:] = np.random.default_rng(42).normal(0, 50, (64, 64))
        deltas = compute_adaptive_delta(img, delta_min=10, delta_max=100)
        # Flat region should have lower deltas
        assert deltas[:, :32].mean() < deltas[:, 96:].mean()

    def test_delta_range(self):
        rng = np.random.default_rng(42)
        img = rng.normal(0, 30, (128, 128))
        deltas = compute_adaptive_delta(img, delta_min=15, delta_max=75)
        assert deltas.min() >= 15 - 1e-10
        assert deltas.max() <= 75 + 1e-10

    def test_uniform_image_midpoint(self):
        flat = np.ones((64, 64)) * 50.0
        deltas = compute_adaptive_delta(flat, delta_min=20, delta_max=80)
        expected = (20 + 80) / 2
        assert np.allclose(deltas, expected)


class TestBuildDeltaMap:
    def test_returns_both_subbands(self):
        y = np.random.default_rng(42).uniform(50, 200, (256, 256))
        dmap = build_delta_map(y, wavelet="haar", delta_min=20, delta_max=80)
        assert "lh2" in dmap
        assert "hl2" in dmap
        assert dmap["lh2"].shape == (64, 64)  # 256 / 2^2 = 64
        assert dmap["hl2"].shape == (64, 64)
