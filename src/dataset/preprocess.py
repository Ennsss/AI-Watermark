"""Preprocessing: standardize a raw illustration into a 512x512 RGB PNG.

Steps, in order:
    1. Flatten any alpha channel onto a solid background (default white), because
       QIM cannot operate on transparent regions whose coefficients are zero by
       construction.
    2. Center-crop to a square on the shorter side (no aspect distortion; valid
       because acquisition pre-filters to aspect ratio 0.5..2.0).
    3. Lanczos-resize the square to TARGET x TARGET.
    4. Return a contiguous uint8 RGB array; the caller writes lossless PNG.

All functions operate on numpy arrays / PIL Images so they are unit-testable
without touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 512
DEFAULT_BG = (255, 255, 255)  # white; pass bg=(128, 128, 128) for neutral gray


def flatten_alpha(image: Image.Image, bg: tuple[int, int, int] = DEFAULT_BG) -> Image.Image:
    """Composite an image with transparency onto a solid background.

    Returns an RGB image. Images already without alpha are returned as RGB
    unchanged (mode-converted if necessary).
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*bg, 255))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")
    return image.convert("RGB")


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Crop the central square (side = min(H, W)) from an (H, W, C) array."""
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return image[top:top + side, left:left + side]


def resize_square(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Lanczos-resize a square (H == W) array to size x size."""
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image)
    # cv2 works in BGR; round-trip to keep channel order correct for RGB input.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def preprocess_image(
    image: Image.Image,
    size: int = TARGET_SIZE,
    bg: tuple[int, int, int] = DEFAULT_BG,
) -> np.ndarray:
    """Full preprocess: flatten alpha -> center-crop square -> resize.

    Returns an (size, size, 3) uint8 RGB array.
    """
    rgb = flatten_alpha(image, bg=bg)
    arr = np.asarray(rgb)
    arr = center_crop_square(arr)
    arr = resize_square(arr, size=size)
    return arr


def preprocess_file(
    src: str | Path,
    dst: str | Path,
    size: int = TARGET_SIZE,
    bg: tuple[int, int, int] = DEFAULT_BG,
) -> None:
    """Read an image file, preprocess it, and write a lossless PNG to dst."""
    with Image.open(src) as im:
        im.load()
        arr = preprocess_image(im, size=size, bg=bg)
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    # PIL writes PNG in RGB order directly (no BGR confusion).
    Image.fromarray(arr, mode="RGB").save(out, format="PNG")
