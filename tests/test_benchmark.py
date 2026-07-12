"""Tests for benchmark.runner — batch BER/SSIM measurement."""

import numpy as np

from benchmark.runner import BenchmarkConfig, BenchmarkSummary, embed_image, run_benchmark


class TestEmbedImage:
    def test_returns_watermarked_and_bits(self):
        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
        config = BenchmarkConfig()
        wm, bits, elapsed = embed_image(img, config)
        assert wm.shape == img.shape
        assert wm.dtype == np.uint8
        assert len(bits) > 0
        assert elapsed > 0


class TestBenchmarkSummary:
    def test_csv_export(self, tmp_path):
        summary = BenchmarkSummary()
        from benchmark.runner import BenchmarkResult
        summary.results.append(BenchmarkResult(
            image_name="test", attack_name="jpeg_q70",
            wavelet="haar", delta=40.0, adaptive=False,
            ber_pre_ecc=0.05, recovery_success=True,
            ssim_score=0.98, embed_time_s=0.5, extract_time_s=0.3,
            num_bits=256, confidence=0.95,
        ))
        csv_path = tmp_path / "results.csv"
        summary.to_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "jpeg_q70" in content
        assert "0.050000" in content


class TestRunBenchmark:
    def test_runs_on_single_image(self):
        rng = np.random.default_rng(42)
        images = {"test_256": rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)}
        config = BenchmarkConfig(delta=40.0)
        # Use only JPEG attacks for speed
        from attacks.suite import jpeg_compression
        attacks = [("jpeg_q80", jpeg_compression, {"quality": 80})]
        summary = run_benchmark(images, config, attacks=attacks)
        assert len(summary.results) == 2  # 1 no-attack + 1 attack
        assert summary.mean_ssim > 0
