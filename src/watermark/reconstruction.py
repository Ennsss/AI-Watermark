"""F4: Inverse DWT & output — IDWT reconstruction, channel recombination, RGB export."""

from __future__ import annotations

import numpy as np

from watermark.preprocessor import (
    replace_y_channel,
    unpad,
    ycbcr_to_rgb,
)


def reconstruct_image(
    ycbcr: np.ndarray,
    watermarked_y: np.ndarray,
    pad_sizes: tuple[int, int],
) -> np.ndarray:
    """Reconstruct a full RGB image from a watermarked Y channel.

    Replaces the Y channel in the original YCbCr image, removes padding,
    and converts back to RGB uint8.

    Args:
        ycbcr: (H_padded, W_padded, 3) float64 YCbCr image (with padding).
        watermarked_y: (H_padded, W_padded) float64 watermarked luminance.
        pad_sizes: (pad_h, pad_w) from pad_to_multiple, to remove padding.

    Returns:
        (H_orig, W_orig, 3) uint8 RGB image.
    """
    # Clip watermarked Y to valid luminance range
    watermarked_y = np.clip(watermarked_y, 0, 255)

    # Replace Y channel
    result_ycbcr = replace_y_channel(ycbcr, watermarked_y)

    # Remove padding
    result_ycbcr = unpad(result_ycbcr, pad_sizes)

    # Convert back to RGB
    return ycbcr_to_rgb(result_ycbcr)
