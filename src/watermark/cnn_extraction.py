"""CNN-assisted extraction utilities.

The CNN branch receives degraded transform-domain coefficients, not raw RGB.
It predicts the same 128-bit payload embedded by the classical DWT-QIM embedder.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from watermark.embedding import dwt2_decompose
from watermark.preprocessor import extract_y_channel, pad_to_multiple, rgb_to_ycbcr


class BitPredictor(Protocol):
    """Minimal interface expected from a CNN decoder."""

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Return bit probabilities for a batch of CNN inputs."""


def prepare_cnn_input_from_y(
    y_channel: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    mode: str = "symmetric",
) -> np.ndarray:
    """Stack degraded LH2 and HL2 maps as a CNN input tensor.

    Args:
        y_channel: Luminance channel. In the main experiment this is 512 x 512.
        wavelet: DWT wavelet. The main experiment uses Haar.
        level: DWT level. The main experiment uses level 2.
        mode: DWT signal extension mode.

    Returns:
        Float32 array shaped (128, 128, 2) for 512 x 512 inputs.
    """
    y_padded, _ = pad_to_multiple(y_channel, multiple=2**level)
    coeffs = dwt2_decompose(y_padded, wavelet=wavelet, level=level, mode=mode)
    lh2, hl2, _hh2 = coeffs[1]
    return np.stack([lh2, hl2], axis=-1).astype(np.float32)


def prepare_cnn_input_from_image(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    mode: str = "symmetric",
) -> np.ndarray:
    """Convert an RGB image into the LH2/HL2 CNN input tensor."""
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    return prepare_cnn_input_from_y(y, wavelet=wavelet, level=level, mode=mode)


def predict_payload_bits(
    model: BitPredictor,
    cnn_input: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run a CNN decoder and threshold its 128 sigmoid outputs."""
    batch = cnn_input[np.newaxis, ...]
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    return (probabilities >= threshold).astype(np.uint8)

