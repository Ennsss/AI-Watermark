"""F3: DWT embedding engine — 2-level DWT, QIM into LH2/HL2, spread-spectrum."""

from __future__ import annotations

import numpy as np
import pywt


def dwt2_decompose(y_channel: np.ndarray, wavelet: str = "haar", level: int = 2) -> list:
    """Perform multi-level 2D DWT decomposition on the Y channel.

    Args:
        y_channel: (H, W) float64 luminance array. Dimensions must be
                   divisible by 2^level.
        wavelet: Wavelet name ('haar' or 'db4').
        level: Decomposition levels (default 2).

    Returns:
        PyWavelets coefficient list: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        where cH=LH (horizontal detail), cV=HL (vertical detail), cD=HH (diagonal).
    """
    coeffs = pywt.wavedec2(y_channel, wavelet=wavelet, level=level)
    return coeffs


def dwt2_reconstruct(coeffs: list, wavelet: str = "haar") -> np.ndarray:
    """Reconstruct Y channel from DWT coefficients via inverse DWT.

    Args:
        coeffs: PyWavelets coefficient list from dwt2_decompose.
        wavelet: Must match the wavelet used for decomposition.

    Returns:
        (H, W) float64 reconstructed luminance array.
    """
    return pywt.waverec2(coeffs, wavelet=wavelet)


def qim_embed_bit(coefficient: float, bit: int, delta: float) -> float:
    """Embed a single bit into a coefficient using QIM.

    Bit 0: quantize to nearest multiple of delta.
    Bit 1: quantize to nearest multiple of delta, offset by delta/2.

    Args:
        coefficient: Original wavelet coefficient value.
        bit: 0 or 1.
        delta: Quantization step size.

    Returns:
        Modified coefficient with embedded bit.
    """
    if bit == 0:
        return delta * np.round(coefficient / delta)
    else:
        return delta * np.round((coefficient - delta / 2) / delta) + delta / 2


def qim_extract_bit(coefficient: float, delta: float) -> int:
    """Extract a single bit from a coefficient using QIM decision rule.

    Checks which quantization grid the coefficient is closer to.

    Args:
        coefficient: Watermarked (possibly degraded) wavelet coefficient.
        delta: Quantization step size (must match embedding).

    Returns:
        Extracted bit (0 or 1).
    """
    # Distance to grid 0 (multiples of delta)
    d0 = abs(coefficient - delta * np.round(coefficient / delta))
    # Distance to grid 1 (multiples of delta, offset by delta/2)
    d1 = abs(coefficient - (delta * np.round((coefficient - delta / 2) / delta) + delta / 2))
    return 0 if d0 <= d1 else 1


def _get_embedding_locations(
    subband_shape: tuple[int, int],
    num_bits: int,
    seed: int,
) -> np.ndarray:
    """Generate pseudo-random embedding locations via PRNG.

    Args:
        subband_shape: (H, W) of the target subband.
        num_bits: Number of bits to embed.
        seed: PRNG seed (derived from secret key).

    Returns:
        (num_bits, 2) array of (row, col) indices into the subband.
    """
    total_coeffs = subband_shape[0] * subband_shape[1]
    if num_bits > total_coeffs:
        raise ValueError(
            f"Payload ({num_bits} bits) exceeds available coefficients "
            f"({total_coeffs}) in subband {subband_shape}"
        )
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(total_coeffs, size=num_bits, replace=False)
    flat_indices.sort()  # Sorted for deterministic access order
    rows = flat_indices // subband_shape[1]
    cols = flat_indices % subband_shape[1]
    return np.stack([rows, cols], axis=1)


def embed_watermark(
    y_channel: np.ndarray,
    bits: np.ndarray,
    seed: int,
    delta: float = 60.0,
    wavelet: str = "haar",
    level: int = 2,
    delta_map: np.ndarray | None = None,
    target_subbands: tuple[str, ...] = ("lh2", "hl2"),
) -> np.ndarray:
    """Embed a bit payload into the Y channel using DWT + QIM.

    By default, bits are spread across LH2 and HL2 subbands. For sparse
    images (line art), target_subbands=("ll2",) embeds all bits into the
    LL2 approximation subband instead.

    Args:
        y_channel: (H, W) float64 luminance array (padded to multiple of 4).
        bits: 1D uint8 array of 0s and 1s.
        seed: PRNG seed for spread-spectrum location mapping.
        delta: Base QIM quantization step. Ignored if delta_map provided.
        wavelet: Wavelet basis ('haar' or 'db4').
        level: DWT decomposition levels.
        delta_map: Optional per-coefficient delta arrays as a dict-like
                   with keys 'lh2' and 'hl2', each shaped to match
                   the respective subband. If None, uniform delta is used.
        target_subbands: Which subbands to embed into. Default ("lh2", "hl2").
                         Use ("ll2",) for line-art LL2 fallback.

    Returns:
        (H, W) float64 watermarked luminance array.
    """
    coeffs = dwt2_decompose(y_channel, wavelet=wavelet, level=level)
    num_bits = len(bits)

    if target_subbands == ("ll2",):
        # LL2 fallback: embed all bits into approximation subband
        ll2 = coeffs[0]
        locs = _get_embedding_locations(ll2.shape, num_bits, seed)
        for i, (r, c) in enumerate(locs):
            ll2[r, c] = qim_embed_bit(ll2[r, c], int(bits[i]), delta)
        coeffs[0] = ll2
    else:
        # Default: split across LH2 and HL2
        lh2, hl2, hh2 = coeffs[1]
        half = num_bits // 2
        bits_lh2 = bits[:half]
        bits_hl2 = bits[half:]

        locs_lh2 = _get_embedding_locations(lh2.shape, len(bits_lh2), seed)
        locs_hl2 = _get_embedding_locations(hl2.shape, len(bits_hl2), seed + 1)

        for i, (r, c) in enumerate(locs_lh2):
            d = delta_map["lh2"][r, c] if delta_map is not None else delta
            lh2[r, c] = qim_embed_bit(lh2[r, c], int(bits_lh2[i]), d)

        for i, (r, c) in enumerate(locs_hl2):
            d = delta_map["hl2"][r, c] if delta_map is not None else delta
            hl2[r, c] = qim_embed_bit(hl2[r, c], int(bits_hl2[i]), d)

        coeffs[1] = (lh2, hl2, hh2)

    return dwt2_reconstruct(coeffs, wavelet=wavelet)


def extract_watermark(
    y_channel: np.ndarray,
    num_bits: int,
    seed: int,
    delta: float = 60.0,
    wavelet: str = "haar",
    level: int = 2,
    delta_map: np.ndarray | None = None,
    target_subbands: tuple[str, ...] = ("lh2", "hl2"),
) -> np.ndarray:
    """Extract a bit payload from a watermarked Y channel.

    Args:
        y_channel: (H, W) float64 luminance array.
        num_bits: Expected number of embedded bits.
        seed: PRNG seed (must match embedding).
        delta: Base QIM quantization step. Ignored if delta_map provided.
        wavelet: Wavelet basis (must match embedding).
        level: DWT decomposition levels.
        delta_map: Optional per-coefficient delta map (must match embedding).
        target_subbands: Must match embedding. ("ll2",) for LL2 fallback.

    Returns:
        1D uint8 array of extracted bits (0s and 1s).
    """
    coeffs = dwt2_decompose(y_channel, wavelet=wavelet, level=level)

    if target_subbands == ("ll2",):
        ll2 = coeffs[0]
        locs = _get_embedding_locations(ll2.shape, num_bits, seed)
        extracted = np.zeros(num_bits, dtype=np.uint8)
        for i, (r, c) in enumerate(locs):
            extracted[i] = qim_extract_bit(ll2[r, c], delta)
        return extracted

    lh2, hl2, _hh2 = coeffs[1]
    half = num_bits // 2
    locs_lh2 = _get_embedding_locations(lh2.shape, half, seed)
    locs_hl2 = _get_embedding_locations(hl2.shape, num_bits - half, seed + 1)

    extracted = np.zeros(num_bits, dtype=np.uint8)

    for i, (r, c) in enumerate(locs_lh2):
        d = delta_map["lh2"][r, c] if delta_map is not None else delta
        extracted[i] = qim_extract_bit(lh2[r, c], d)

    for i, (r, c) in enumerate(locs_hl2):
        d = delta_map["hl2"][r, c] if delta_map is not None else delta
        extracted[half + i] = qim_extract_bit(hl2[r, c], d)

    return extracted
