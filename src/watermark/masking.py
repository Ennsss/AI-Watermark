"""F10: Perceptual masking — local variance heuristic for adaptive QIM delta."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def compute_local_variance(
    subband: np.ndarray, window_size: int = 7
) -> np.ndarray:
    """Compute local variance in a sliding window over a subband.

    Args:
        subband: 2D float64 wavelet subband array.
        window_size: Size of the square averaging window.

    Returns:
        2D float64 array of local variance values, same shape as subband.
    """
    mean = uniform_filter(subband, size=window_size, mode="reflect")
    mean_sq = uniform_filter(subband**2, size=window_size, mode="reflect")
    variance = mean_sq - mean**2
    # Clamp numerical noise
    return np.maximum(variance, 0.0)


def compute_adaptive_delta(
    subband: np.ndarray,
    delta_min: float = 20.0,
    delta_max: float = 80.0,
    window_size: int = 7,
) -> np.ndarray:
    """Compute spatially-varying QIM delta based on local variance.

    High variance (textured) regions get larger delta (more robust).
    Low variance (flat) regions get smaller delta (less visible).

    Args:
        subband: 2D float64 wavelet subband.
        delta_min: Minimum delta for flat regions.
        delta_max: Maximum delta for textured regions.
        window_size: Local variance window size.

    Returns:
        2D float64 array of per-coefficient delta values.
    """
    variance = compute_local_variance(subband, window_size)

    # Normalize variance to [0, 1] using robust scaling
    v_low = np.percentile(variance, 5)
    v_high = np.percentile(variance, 95)

    if v_high - v_low < 1e-10:
        # Near-uniform image — use midpoint delta
        return np.full_like(subband, (delta_min + delta_max) / 2)

    normalized = np.clip((variance - v_low) / (v_high - v_low), 0.0, 1.0)

    # Linear mapping to [delta_min, delta_max]
    return delta_min + normalized * (delta_max - delta_min)


def detect_sparse_subbands(
    y_channel: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    threshold: float = 1.0,
    sparsity_limit: float = 0.80,
) -> bool:
    """Detect whether LH2/HL2 subbands are too sparse for effective QIM.

    Line art and flat images have most detail-subband coefficients near zero,
    making QIM ineffective. When detected, callers should fall back to LL2.

    Args:
        y_channel: (H, W) float64 luminance array (padded).
        wavelet: Wavelet basis.
        level: DWT decomposition level.
        threshold: Absolute value below which a coefficient is "near-zero".
        sparsity_limit: Fraction of near-zero coefficients that triggers sparse detection.

    Returns:
        True if LH2/HL2 are too sparse (should use LL2 fallback).
    """
    import pywt

    coeffs = pywt.wavedec2(y_channel, wavelet=wavelet, level=level)
    lh2, hl2, _hh2 = coeffs[1]

    total = lh2.size + hl2.size
    near_zero = np.sum(np.abs(lh2) < threshold) + np.sum(np.abs(hl2) < threshold)
    return (near_zero / total) > sparsity_limit


def build_delta_map(
    y_channel: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    delta_min: float = 20.0,
    delta_max: float = 80.0,
    window_size: int = 7,
) -> dict[str, np.ndarray]:
    """Build per-coefficient adaptive delta maps for LH2 and HL2 subbands.

    Uses the LL2 (approximation) subband to determine texture regions.
    LL2 is stable under JPEG compression, so both embed and extract sides
    compute nearly identical delta maps even after image degradation.

    Args:
        y_channel: (H, W) float64 luminance array (padded).
        wavelet: Wavelet basis.
        level: DWT decomposition level.
        delta_min: Minimum delta for flat regions.
        delta_max: Maximum delta for textured regions.
        window_size: Local variance window size.

    Returns:
        Dict with keys 'lh2' and 'hl2', each containing a 2D delta array.
    """
    import pywt
    from scipy.ndimage import zoom

    coeffs = pywt.wavedec2(y_channel, wavelet=wavelet, level=level)
    ll2 = coeffs[0]
    lh2, hl2, _hh2 = coeffs[1]

    # Compute variance from LL2 (most stable under compression)
    ll2_variance = compute_local_variance(ll2, window_size)

    # LL2 and LH2/HL2 have the same shape at level 2, so use directly.
    # Normalize and map to delta range.
    v_low = np.percentile(ll2_variance, 5)
    v_high = np.percentile(ll2_variance, 95)

    if v_high - v_low < 1e-10:
        delta_lh2 = np.full_like(lh2, (delta_min + delta_max) / 2)
        delta_hl2 = np.full_like(hl2, (delta_min + delta_max) / 2)
    else:
        normalized = np.clip((ll2_variance - v_low) / (v_high - v_low), 0.0, 1.0)
        delta_base = delta_min + normalized * (delta_max - delta_min)

        # Resize if shapes differ (shouldn't at level 2, but safe guard)
        if delta_base.shape != lh2.shape:
            zoom_factors = (lh2.shape[0] / delta_base.shape[0],
                            lh2.shape[1] / delta_base.shape[1])
            delta_lh2 = zoom(delta_base, zoom_factors, order=1)
            delta_hl2 = zoom(delta_base, zoom_factors, order=1)
        else:
            delta_lh2 = delta_base.copy()
            delta_hl2 = delta_base.copy()

    return {
        "lh2": delta_lh2,
        "hl2": delta_hl2,
    }
