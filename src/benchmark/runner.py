"""F7: Robustness benchmark runner — batch BER/SSIM measurement, CSV export."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from attacks.suite import AttackResult, get_default_attacks
from watermark.embedding import embed_watermark, extract_watermark
from watermark.extraction import compute_ber, extract_from_image
from watermark.masking import build_delta_map, detect_sparse_subbands
from watermark.payload import (
    decode_payload_bits,
    derive_seed,
    encode_payload,
)
from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    rgb_to_ycbcr,
)
from watermark.reconstruction import reconstruct_image


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    wavelet: str = "haar"
    delta: float = 60.0
    adaptive: bool = False
    delta_min: float = 20.0
    delta_max: float = 80.0
    rs_nsym: int = 128
    artist_id: str = "benchmark_artist"
    key: bytes = b"benchmark_secret_key_32bytes!!"
    timestamp: int = 1709337600
    repetitions: int = 1


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test (one image + one attack)."""

    image_name: str
    attack_name: str
    wavelet: str
    delta: float
    adaptive: bool
    ber_pre_ecc: float
    recovery_success: bool
    ssim_score: float
    embed_time_s: float
    extract_time_s: float
    num_bits: int
    confidence: float
    notes: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregate statistics from a benchmark run."""

    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def mean_ber(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.ber_pre_ecc for r in self.results]))

    @property
    def mean_ssim(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.ssim_score for r in self.results]))

    @property
    def recovery_rate(self) -> float:
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.recovery_success)
        return successes / len(self.results)

    def to_csv(self, path: str | Path) -> None:
        """Export results to CSV.

        Args:
            path: Output CSV file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "image_name", "attack_name", "wavelet", "delta", "adaptive",
            "ber_pre_ecc", "recovery_success", "ssim_score",
            "embed_time_s", "extract_time_s", "num_bits", "confidence", "notes",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "image_name": r.image_name,
                    "attack_name": r.attack_name,
                    "wavelet": r.wavelet,
                    "delta": f"{r.delta:.1f}",
                    "adaptive": r.adaptive,
                    "ber_pre_ecc": f"{r.ber_pre_ecc:.6f}",
                    "recovery_success": r.recovery_success,
                    "ssim_score": f"{r.ssim_score:.6f}",
                    "embed_time_s": f"{r.embed_time_s:.4f}",
                    "extract_time_s": f"{r.extract_time_s:.4f}",
                    "num_bits": r.num_bits,
                    "confidence": f"{r.confidence:.4f}",
                    "notes": r.notes,
                })

    def print_summary(self) -> None:
        """Print aggregate statistics to stdout."""
        print(f"\n{'='*60}")
        print(f"Benchmark Summary: {len(self.results)} tests")
        print(f"{'='*60}")
        print(f"  Mean BER (pre-ECC):   {self.mean_ber:.4f}")
        print(f"  Mean SSIM:            {self.mean_ssim:.4f}")
        print(f"  Recovery Rate:        {self.recovery_rate:.1%}")
        print(f"{'='*60}\n")


def embed_image(
    image: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed a watermark into an image using the benchmark config.

    Args:
        image: (H, W, 3) uint8 RGB.
        config: Benchmark configuration.

    Returns:
        Tuple of (watermarked_rgb, payload_bits, embed_time_seconds).
    """
    t0 = time.perf_counter()

    # Encode payload
    bits = encode_payload(
        config.artist_id, image, config.key,
        timestamp=config.timestamp, rs_nsym=config.rs_nsym,
        repetitions=config.repetitions,
    )

    # Pre-process
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, pad_sizes = pad_to_multiple(y, 4)
    ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

    # Build delta map if adaptive
    delta_map = None
    if config.adaptive:
        delta_map = build_delta_map(
            y_padded, wavelet=config.wavelet,
            delta_min=config.delta_min, delta_max=config.delta_max,
        )

    seed = derive_seed(config.key)

    # Detect sparse subbands (line art) and use LL2 fallback
    sparse = detect_sparse_subbands(y_padded, wavelet=config.wavelet)
    target_subbands = ("ll2",) if sparse else ("lh2", "hl2")

    # Embed
    wm_y = embed_watermark(
        y_padded, bits, seed=seed, delta=config.delta,
        wavelet=config.wavelet, delta_map=delta_map,
        target_subbands=target_subbands,
    )

    # Reconstruct
    watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

    elapsed = time.perf_counter() - t0
    return watermarked, bits, elapsed


def extract_and_measure(
    attacked_image: np.ndarray,
    original_bits: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[float, bool, float, float]:
    """Extract watermark from attacked image and measure accuracy.

    Args:
        attacked_image: (H, W, 3) uint8 RGB (possibly degraded).
        original_bits: Ground truth payload bits.
        config: Benchmark configuration (must match embedding).

    Returns:
        Tuple of (ber, recovery_success, confidence, extract_time).
    """
    t0 = time.perf_counter()
    seed = derive_seed(config.key)
    num_bits = len(original_bits)

    # Build delta map if adaptive (from the attacked image)
    delta_map = None
    ycbcr_att = rgb_to_ycbcr(attacked_image)
    y_att = extract_y_channel(ycbcr_att)
    y_att_padded, _ = pad_to_multiple(y_att, 4)

    if config.adaptive:
        delta_map = build_delta_map(
            y_att_padded, wavelet=config.wavelet,
            delta_min=config.delta_min, delta_max=config.delta_max,
        )

    # Detect sparse subbands (line art) and use LL2 fallback
    sparse = detect_sparse_subbands(y_att_padded, wavelet=config.wavelet)
    target_subbands = ("ll2",) if sparse else ("lh2", "hl2")

    extracted_bits, confidence = extract_from_image(
        attacked_image, num_bits=num_bits, seed=seed,
        delta=config.delta, wavelet=config.wavelet, delta_map=delta_map,
        target_subbands=target_subbands,
    )

    ber = compute_ber(original_bits, extracted_bits)

    # Try full payload decode
    recovery_success = False
    try:
        _prov = decode_payload_bits(
            extracted_bits, config.key, rs_nsym=config.rs_nsym,
            repetitions=config.repetitions,
        )
        recovery_success = True
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    return ber, recovery_success, confidence, elapsed


def run_benchmark(
    images: dict[str, np.ndarray],
    config: BenchmarkConfig | None = None,
    attacks: list | None = None,
) -> BenchmarkSummary:
    """Run the full benchmark: embed each image, apply each attack, measure.

    Args:
        images: Dict of {name: rgb_uint8_array}.
        config: Benchmark config. Uses defaults if None.
        attacks: List of (name, fn, kwargs). Uses default suite if None.

    Returns:
        BenchmarkSummary with all results.
    """
    if config is None:
        config = BenchmarkConfig()
    if attacks is None:
        attacks = get_default_attacks()

    summary = BenchmarkSummary()

    for img_name, image in images.items():
        print(f"\nProcessing: {img_name} ({image.shape[1]}x{image.shape[0]})")

        # Embed
        watermarked, bits, embed_time = embed_image(image, config)

        # SSIM of watermarked vs original (no attack)
        ssim_base = ssim(image, watermarked, channel_axis=2)
        summary.results.append(BenchmarkResult(
            image_name=img_name, attack_name="none",
            wavelet=config.wavelet, delta=config.delta,
            adaptive=config.adaptive, ber_pre_ecc=0.0,
            recovery_success=True, ssim_score=ssim_base,
            embed_time_s=embed_time, extract_time_s=0.0,
            num_bits=len(bits), confidence=1.0,
        ))

        # Apply each attack
        for atk_name, atk_fn, atk_kwargs in attacks:
            try:
                result: AttackResult = atk_fn(watermarked, **atk_kwargs)
                attacked = result.image

                # If attack changed dimensions, resize back to original
                # for extraction attempt (coefficient alignment required).
                if attacked.shape[:2] != watermarked.shape[:2]:
                    attacked = cv2.resize(
                        cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR),
                        (watermarked.shape[1], watermarked.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    attacked = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)

                # Measure SSIM between original and attacked
                ssim_score = ssim(image, attacked, channel_axis=2)

                # Extract and measure
                ber, success, conf, ext_time = extract_and_measure(
                    attacked, bits, config,
                )

                summary.results.append(BenchmarkResult(
                    image_name=img_name, attack_name=atk_name,
                    wavelet=config.wavelet, delta=config.delta,
                    adaptive=config.adaptive, ber_pre_ecc=ber,
                    recovery_success=success, ssim_score=ssim_score,
                    embed_time_s=embed_time, extract_time_s=ext_time,
                    num_bits=len(bits), confidence=conf,
                ))
            except Exception as e:
                summary.results.append(BenchmarkResult(
                    image_name=img_name, attack_name=atk_name,
                    wavelet=config.wavelet, delta=config.delta,
                    adaptive=config.adaptive, ber_pre_ecc=1.0,
                    recovery_success=False, ssim_score=0.0,
                    embed_time_s=embed_time, extract_time_s=0.0,
                    num_bits=len(bits), confidence=0.0,
                    notes=str(e),
                ))

    return summary
