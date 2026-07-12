"""F1: Image pre-processing — RGB/YCbCr conversion, Y channel extraction, padding."""

from __future__ import annotations

import numpy as np
import cv2


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to YCbCr float64.

    Args:
        image: (H, W, 3) uint8 array in RGB order.

    Returns:
        (H, W, 3) float64 array in YCbCr order.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image.shape}")
    # OpenCV expects BGR input
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    # OpenCV outputs Y, Cr, Cb — reorder to Y, Cb, Cr
    y = ycrcb[:, :, 0]
    cb = ycrcb[:, :, 2]
    cr = ycrcb[:, :, 1]
    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert a YCbCr float64 image back to RGB uint8.

    Args:
        ycbcr: (H, W, 3) float64 array in Y, Cb, Cr order.

    Returns:
        (H, W, 3) uint8 array in RGB order.
    """
    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) YCbCr image, got shape {ycbcr.shape}")
    # Reorder to Y, Cr, Cb for OpenCV
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    ycrcb = np.stack([y, cr, cb], axis=2)
    ycrcb_u8 = np.clip(np.round(ycrcb), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def extract_y_channel(ycbcr: np.ndarray) -> np.ndarray:
    """Extract the Y (luminance) channel from a YCbCr image.

    Returns:
        (H, W) float64 array.
    """
    return ycbcr[:, :, 0].copy()


def replace_y_channel(ycbcr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace the Y channel in a YCbCr image.

    Args:
        ycbcr: (H, W, 3) float64 YCbCr image.
        y: (H, W) float64 replacement luminance channel.

    Returns:
        New (H, W, 3) float64 YCbCr image with replaced Y.
    """
    result = ycbcr.copy()
    result[:, :, 0] = y
    return result


def pad_to_multiple(image: np.ndarray, multiple: int = 4) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a 2D array so both dimensions are multiples of `multiple`.

    Uses symmetric (reflect) padding to avoid edge artifacts.

    Args:
        image: (H, W) or (H, W, C) array.
        multiple: Target multiple (default 4 for 2-level DWT).

    Returns:
        Tuple of (padded_image, (pad_h, pad_w)) where pad values
        indicate how many rows/cols were added.
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image.copy(), (0, 0)

    if image.ndim == 2:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="symmetric")
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
    return padded, (pad_h, pad_w)


def unpad(image: np.ndarray, pad_sizes: tuple[int, int]) -> np.ndarray:
    """Remove padding added by pad_to_multiple.

    Args:
        image: Padded array.
        pad_sizes: (pad_h, pad_w) from pad_to_multiple.

    Returns:
        Cropped array with padding removed.
    """
    pad_h, pad_w = pad_sizes
    h, w = image.shape[:2]
    end_h = h - pad_h if pad_h > 0 else h
    end_w = w - pad_w if pad_w > 0 else w
    if image.ndim == 2:
        return image[:end_h, :end_w].copy()
    return image[:end_h, :end_w, :].copy()


def load_image(path: str) -> np.ndarray:
    """Load an image file as RGB uint8 array.

    Args:
        path: Path to image file (PNG, JPEG, TIFF, etc.).

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path: str, image: np.ndarray) -> None:
    """Save an RGB uint8 image to file.

    Args:
        path: Output path. Use .png for lossless output.
        image: (H, W, 3) uint8 RGB array.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, bgr)
    if not success:
        raise IOError(f"Failed to write image: {path}")
