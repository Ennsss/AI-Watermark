"""Comprehensive metrics for watermark evaluation.

Extends the base BER/SSIM metrics with PSNR, Normalized Correlation,
capacity measures, and false positive rate analysis.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between original and modified images.

    Args:
        original: Reference image, uint8.
        modified: Distorted image, uint8 (same shape as original).

    Returns:
        PSNR in dB. Higher is better; inf if images are identical.
    """
    if original.shape != modified.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs modified {modified.shape}"
        )
    return float(peak_signal_noise_ratio(original, modified, data_range=255))


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Structural Similarity Index between original and modified images.

    Args:
        original: Reference image, uint8.
        modified: Distorted image, uint8 (same shape as original).

    Returns:
        SSIM in [0, 1]. Higher is better.
    """
    if original.shape != modified.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs modified {modified.shape}"
        )
    channel_axis = 2 if original.ndim == 3 else None
    return float(structural_similarity(
        original, modified, data_range=255, channel_axis=channel_axis,
    ))


def compute_nc(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Normalized Correlation between original and extracted bit sequences.

    Maps bits from {0,1} to {-1,+1} and computes the normalized dot product.
    Standard metric in watermarking literature (NC=1.0 means perfect match).

    Args:
        original_bits: Ground truth bits (0/1).
        extracted_bits: Extracted bits (0/1), same length.

    Returns:
        NC in [-1, 1]. 1.0 = perfect, 0.0 = uncorrelated, -1.0 = inverted.
    """
    if len(original_bits) != len(extracted_bits):
        raise ValueError(
            f"Length mismatch: {len(original_bits)} vs {len(extracted_bits)}"
        )
    if len(original_bits) == 0:
        return 1.0

    # Map {0, 1} → {-1, +1}
    a = 2.0 * original_bits.astype(np.float64) - 1.0
    b = 2.0 * extracted_bits.astype(np.float64) - 1.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_capacity_bpp(num_bits: int, image_shape: tuple[int, ...]) -> float:
    """Embedding capacity in bits per pixel.

    Args:
        num_bits: Total number of embedded bits (including ECC/repetition).
        image_shape: Image shape (H, W) or (H, W, C).

    Returns:
        Bits per pixel.
    """
    h, w = image_shape[:2]
    return num_bits / (h * w)


def compute_subband_utilization(
    num_bits: int,
    image_shape: tuple[int, ...],
    level: int = 2,
) -> float:
    """Fraction of available DWT subband coefficients used for embedding.

    At decomposition level L, each detail subband (LH, HL) has
    H/(2^L) x W/(2^L) coefficients. We embed into LH and HL subbands.

    Args:
        num_bits: Number of bits embedded.
        image_shape: Image shape (H, W) or (H, W, C).
        level: DWT decomposition level.

    Returns:
        Utilization fraction in [0, 1].
    """
    h, w = image_shape[:2]
    sub_h = h // (2 ** level)
    sub_w = w // (2 ** level)
    # Two subbands: LH and HL at the target level
    total_coefficients = 2 * sub_h * sub_w
    if total_coefficients == 0:
        return 0.0
    return min(num_bits / total_coefficients, 1.0)


def compute_false_positive_rate(
    num_trials: int,
    image_shape: tuple[int, int],
    num_bits: int,
    delta: float,
    seed: int,
    key: bytes,
    rs_nsym: int = 128,
    repetitions: int = 1,
    wavelet: str = "haar",
) -> tuple[float, int]:
    """Estimate false positive rate by attempting extraction on unwatermarked images.

    Generates random images, attempts extraction + RS decoding, and counts
    how many succeed (false positives).

    Args:
        num_trials: Number of random images to test.
        image_shape: (H, W) for generated images.
        num_bits: Expected payload length in bits.
        delta: QIM quantization step.
        seed: PRNG seed for extraction.
        key: AES key for RS decoding attempts.
        rs_nsym: Reed-Solomon redundancy symbols.
        repetitions: Repetition coding factor.
        wavelet: Wavelet basis.

    Returns:
        Tuple of (false_positive_rate, num_false_positives).
    """
    from watermark.extraction import extract_from_image
    from watermark.payload import decode_payload_bits

    rng = np.random.default_rng(42)
    false_positives = 0

    for i in range(num_trials):
        # Generate random image
        random_img = rng.integers(0, 256, size=(*image_shape, 3), dtype=np.uint8)

        try:
            extracted_bits, _conf = extract_from_image(
                random_img, num_bits=num_bits, seed=seed,
                delta=delta, wavelet=wavelet,
            )
            # Try RS decoding
            decode_payload_bits(
                extracted_bits, key, rs_nsym=rs_nsym, repetitions=repetitions,
            )
            false_positives += 1
        except Exception:
            pass

    rate = false_positives / num_trials if num_trials > 0 else 0.0
    return rate, false_positives
