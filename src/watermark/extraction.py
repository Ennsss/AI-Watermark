"""F5: Watermark extractor — DWT decomposition, QIM reading, payload recovery."""

from __future__ import annotations

import numpy as np

from watermark.embedding import extract_watermark
from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    rgb_to_ycbcr,
)


def extract_from_image(
    image: np.ndarray,
    num_bits: int,
    seed: int,
    delta: float = 60.0,
    wavelet: str = "haar",
    level: int = 2,
    delta_map: np.ndarray | None = None,
    target_subbands: tuple[str, ...] = ("lh2", "hl2"),
) -> tuple[np.ndarray, float]:
    """Extract watermark bits from a (possibly degraded) RGB image.

    Full extraction pipeline: RGB → YCbCr → Y → pad → DWT → QIM read.

    Args:
        image: (H, W, 3) uint8 RGB image.
        num_bits: Expected number of embedded bits.
        seed: PRNG seed (must match embedding).
        delta: Base QIM step size (must match embedding).
        wavelet: Wavelet basis (must match embedding).
        level: DWT decomposition levels.
        delta_map: Optional adaptive delta map (must match embedding).
        target_subbands: Must match embedding. ("ll2",) for LL2 fallback.

    Returns:
        Tuple of (extracted_bits, confidence):
        - extracted_bits: 1D uint8 array of 0s and 1s.
        - confidence: float in [0, 1] — ratio of bits where QIM
          decision margin exceeds delta/4 (higher = more reliable).
    """
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, _pad_sizes = pad_to_multiple(y, multiple=2**level)

    extracted_bits = extract_watermark(
        y_padded,
        num_bits=num_bits,
        seed=seed,
        delta=delta,
        wavelet=wavelet,
        level=level,
        delta_map=delta_map,
        target_subbands=target_subbands,
    )

    # Compute confidence based on QIM decision margins
    confidence = _compute_confidence(
        y_padded, num_bits, seed, delta, wavelet, level, delta_map,
        target_subbands,
    )

    return extracted_bits, confidence


def _compute_confidence(
    y_channel: np.ndarray,
    num_bits: int,
    seed: int,
    delta: float,
    wavelet: str,
    level: int,
    delta_map: np.ndarray | None,
    target_subbands: tuple[str, ...] = ("lh2", "hl2"),
) -> float:
    """Compute extraction confidence based on QIM decision margins.

    A bit is considered "confident" if the winning grid distance is less
    than delta/4 away from the coefficient — meaning the coefficient hasn't
    been shifted beyond the robustness bound.

    Returns:
        Float in [0, 1]. 1.0 = all bits have strong margins.
    """
    from watermark.embedding import (
        _get_embedding_locations,
        _get_ll2_safe_locations,
        dwt2_decompose,
    )

    coeffs = dwt2_decompose(y_channel, wavelet=wavelet, level=level)

    if target_subbands == ("ll2",):
        ll2 = coeffs[0]
        locs = _get_ll2_safe_locations(ll2, num_bits, seed, delta, level)
        confident_count = 0
        for _i, (r, c) in enumerate(locs):
            coeff = ll2[r, c]
            d0 = abs(coeff - delta * np.round(coeff / delta))
            d1 = abs(coeff - (delta * np.round((coeff - delta / 2) / delta) + delta / 2))
            margin = abs(d0 - d1)
            if margin > delta / 4:
                confident_count += 1
        return confident_count / num_bits if num_bits > 0 else 0.0

    lh2, hl2, _hh2 = coeffs[1]

    half = num_bits // 2
    locs_lh2 = _get_embedding_locations(lh2.shape, half, seed)
    locs_hl2 = _get_embedding_locations(hl2.shape, num_bits - half, seed + 1)

    confident_count = 0

    for i, (r, c) in enumerate(locs_lh2):
        d = delta_map["lh2"][r, c] if delta_map is not None else delta
        coeff = lh2[r, c]
        d0 = abs(coeff - d * np.round(coeff / d))
        d1 = abs(coeff - (d * np.round((coeff - d / 2) / d) + d / 2))
        margin = abs(d0 - d1)
        if margin > d / 4:
            confident_count += 1

    for i, (r, c) in enumerate(locs_hl2):
        d = delta_map["hl2"][r, c] if delta_map is not None else delta
        coeff = hl2[r, c]
        d0 = abs(coeff - d * np.round(coeff / d))
        d1 = abs(coeff - (d * np.round((coeff - d / 2) / d) + d / 2))
        margin = abs(d0 - d1)
        if margin > d / 4:
            confident_count += 1

    return confident_count / num_bits if num_bits > 0 else 0.0


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Compute Bit Error Rate between original and extracted bit arrays.

    Args:
        original_bits: Ground truth bit payload.
        extracted_bits: Extracted bit payload (same length).

    Returns:
        BER as float in [0, 1]. 0 = perfect match.
    """
    if len(original_bits) != len(extracted_bits):
        raise ValueError(
            f"Bit length mismatch: {len(original_bits)} vs {len(extracted_bits)}"
        )
    errors = np.sum(original_bits != extracted_bits)
    return float(errors / len(original_bits))
