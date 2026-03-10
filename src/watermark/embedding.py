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
    subband_data: np.ndarray | None = None,
    min_magnitude: float = 20.0,
    sparsity_threshold: float = 0.5,
) -> np.ndarray:
    """Generate pseudo-random embedding locations via PRNG.

    When *subband_data* is provided **and** the subband is sparse (more than
    *sparsity_threshold* fraction of coefficients have magnitude < 1.0),
    magnitude thresholding is applied: only locations where
    ``abs(coefficient) >= min_magnitude`` are eligible.  This avoids
    embedding into near-zero DWT coefficients (common in line art) that
    are fragile under JPEG compression.

    For non-sparse subbands (natural images) the filter is skipped entirely,
    preserving the original location-selection behaviour and guaranteeing
    that embedder and extractor agree on locations even after QIM modifies
    coefficient values.

    If filtering is active but not enough locations pass, the threshold is
    halved repeatedly (floor 0) until enough are found.

    Args:
        subband_shape: (H, W) of the target subband.
        num_bits: Number of bits to embed.
        seed: PRNG seed (derived from secret key).
        subband_data: Optional (H, W) array of coefficient values used for
                      magnitude thresholding.  When *None*, no filtering is
                      performed regardless of other parameters.
        min_magnitude: Minimum ``abs(coefficient)`` required for a location
                       to be eligible when the subband is sparse.
        sparsity_threshold: Fraction of near-zero coefficients (magnitude
                            < 1.0) above which the subband is considered
                            sparse and magnitude filtering is activated.

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
    # Generate a full random permutation so the order is deterministic
    all_indices = rng.permutation(total_coeffs)

    # Decide whether magnitude filtering should be applied.
    # Only activate for sparse subbands (e.g. line art) where a large
    # fraction of coefficients are near zero.
    use_filtering = False
    if subband_data is not None:
        sparsity = float(np.mean(np.abs(subband_data) < 1.0))
        if sparsity > sparsity_threshold:
            use_filtering = True

    if not use_filtering:
        # Original behaviour: pick first num_bits from permutation
        chosen = np.sort(all_indices[:num_bits])
        rows = chosen // subband_shape[1]
        cols = chosen % subband_shape[1]
        return np.stack([rows, cols], axis=1)

    # Magnitude thresholding for sparse subbands
    perm_rows = all_indices // subband_shape[1]
    perm_cols = all_indices % subband_shape[1]
    magnitudes = np.abs(subband_data[perm_rows, perm_cols])

    threshold = min_magnitude
    while True:
        if threshold <= 0:
            # Floor reached — take first num_bits without filtering
            chosen = np.sort(all_indices[:num_bits])
            rows = chosen // subband_shape[1]
            cols = chosen % subband_shape[1]
            return np.stack([rows, cols], axis=1)

        mask = magnitudes >= threshold
        passing = all_indices[mask]

        if len(passing) >= num_bits:
            chosen = np.sort(passing[:num_bits])
            rows = chosen // subband_shape[1]
            cols = chosen % subband_shape[1]
            return np.stack([rows, cols], axis=1)

        # Not enough — halve threshold and retry
        threshold /= 2.0
        if threshold < 0.5:
            threshold = 0.0


def _get_ll2_safe_locations(
    ll2: np.ndarray,
    num_bits: int,
    seed: int,
    delta: float,
    level: int = 2,
) -> np.ndarray:
    """Generate embedding locations for LL2, skipping saturated coefficients.

    LL2 coefficients near 0 or the maximum (255 * 2^level) can cause QIM
    to produce values that clip after IDWT -> uint8 -> DWT roundtrip, flipping
    bits. This selects only coefficients with enough headroom.

    Args:
        ll2: 2D float64 LL2 subband array.
        num_bits: Number of bits to embed.
        seed: PRNG seed.
        delta: QIM step size.
        level: DWT decomposition level.

    Returns:
        (num_bits, 2) array of (row, col) indices.
    """
    max_coeff = 255.0 * (2 ** level)
    margin = delta

    safe_mask = (ll2 >= margin) & (ll2 <= max_coeff - margin)
    safe_indices = np.flatnonzero(safe_mask.ravel())

    if len(safe_indices) < num_bits:
        safe_indices = np.arange(ll2.size)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(safe_indices), size=num_bits, replace=False)
    chosen.sort()
    flat_indices = safe_indices[chosen]

    rows = flat_indices // ll2.shape[1]
    cols = flat_indices % ll2.shape[1]
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
        # Use safe locations to avoid saturation clipping
        ll2 = coeffs[0]
        locs = _get_ll2_safe_locations(ll2, num_bits, seed, delta, level)
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
        locs = _get_ll2_safe_locations(ll2, num_bits, seed, delta, level)
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
