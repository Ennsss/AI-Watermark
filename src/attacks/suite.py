"""F6: Social media attack suite — 7 attack types, parameterized pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class AttackResult:
    """Result of applying an attack to an image."""

    image: np.ndarray  # Attacked RGB uint8 image
    name: str  # Human-readable attack description


def jpeg_compression(image: np.ndarray, quality: int = 70) -> AttackResult:
    """Simulate JPEG compression at a given quality level.

    Args:
        image: (H, W, 3) uint8 RGB.
        quality: JPEG quality 1-100.

    Returns:
        AttackResult with JPEG-compressed image.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", bgr, encode_params)
    bgr_decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr_decoded, cv2.COLOR_BGR2RGB)
    return AttackResult(image=rgb, name=f"jpeg_q{quality}")


def resize_attack(image: np.ndarray, max_dim: int = 1080) -> AttackResult:
    """Downscale image so largest dimension equals max_dim.

    Preserves aspect ratio. No-op if already smaller.

    Args:
        image: (H, W, 3) uint8 RGB.
        max_dim: Target maximum dimension in pixels.

    Returns:
        AttackResult with resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return AttackResult(image=image.copy(), name=f"resize_{max_dim}px_noop")

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return AttackResult(image=rgb, name=f"resize_{max_dim}px")


def random_crop(image: np.ndarray, crop_ratio: float = 0.2, seed: int = 0) -> AttackResult:
    """Randomly crop a percentage of the image from edges.

    Args:
        image: (H, W, 3) uint8 RGB.
        crop_ratio: Fraction of area to remove (0-1). Applied symmetrically.
        seed: RNG seed for reproducibility.

    Returns:
        AttackResult with cropped image.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]

    # Compute per-side crop as fraction of the linear factor
    # crop_ratio of area ≈ 1 - (1 - linear_ratio)^2
    linear = 1 - np.sqrt(1 - crop_ratio)

    top = int(rng.uniform(0, linear) * h)
    bottom = int(rng.uniform(0, linear) * h)
    left = int(rng.uniform(0, linear) * w)
    right = int(rng.uniform(0, linear) * w)

    # Ensure at least 50% of image remains
    bottom = min(bottom, h - top - h // 2)
    right = min(right, w - left - w // 2)
    bottom = max(bottom, 0)
    right = max(right, 0)

    cropped = image[top : h - bottom, left : w - right].copy()
    pct = int(crop_ratio * 100)
    return AttackResult(image=cropped, name=f"crop_{pct}pct")


def screenshot_simulation(image: np.ndarray, quality: int = 85) -> AttackResult:
    """Simulate screenshot: color space round-trip + JPEG recompression.

    Approximates sRGB → Display P3 → sRGB by applying a slight gamma shift
    and saturation change, then JPEG compress.

    Args:
        image: (H, W, 3) uint8 RGB.
        quality: JPEG quality for the screenshot save.

    Returns:
        AttackResult with simulated screenshot.
    """
    # Simulate color space conversion artifacts with a gamma shift
    float_img = image.astype(np.float64) / 255.0
    # sRGB → linear (approximate gamma 2.2)
    linear = np.power(float_img, 2.2)
    # Slight saturation boost (simulating wider gamut mapping)
    gray = np.mean(linear, axis=2, keepdims=True)
    boosted = gray + 1.02 * (linear - gray)
    # Back to sRGB
    srgb = np.power(np.clip(boosted, 0, 1), 1 / 2.2)
    result = np.clip(srgb * 255, 0, 255).astype(np.uint8)

    # JPEG recompression
    return AttackResult(
        image=jpeg_compression(result, quality).image,
        name=f"screenshot_q{quality}",
    )


def format_conversion(image: np.ndarray) -> AttackResult:
    """Simulate format conversion chain: PNG → JPEG → WebP → JPEG.

    Args:
        image: (H, W, 3) uint8 RGB.

    Returns:
        AttackResult after format round-trip.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # PNG → JPEG Q85
    _, jpg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    bgr1 = cv2.imdecode(jpg_buf, cv2.IMREAD_COLOR)

    # JPEG → WebP Q80
    _, webp_buf = cv2.imencode(".webp", bgr1, [cv2.IMWRITE_WEBP_QUALITY, 80])
    bgr2 = cv2.imdecode(webp_buf, cv2.IMREAD_COLOR)

    # WebP → JPEG Q80
    _, jpg_buf2 = cv2.imencode(".jpg", bgr2, [cv2.IMWRITE_JPEG_QUALITY, 80])
    bgr3 = cv2.imdecode(jpg_buf2, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(bgr3, cv2.COLOR_BGR2RGB)
    return AttackResult(image=rgb, name="format_chain_png_jpg_webp_jpg")


def gaussian_noise(image: np.ndarray, sigma: float = 5.0, seed: int = 0) -> AttackResult:
    """Add Gaussian noise to simulate sensor/capture noise.

    Args:
        image: (H, W, 3) uint8 RGB.
        sigma: Standard deviation of noise.
        seed: RNG seed for reproducibility.

    Returns:
        AttackResult with noisy image.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return AttackResult(image=noisy, name=f"noise_sigma{sigma}")


def combined_chain(
    image: np.ndarray, max_dim: int = 1440, jpeg_q1: int = 70,
    crop_ratio: float = 0.2, jpeg_q2: int = 80, seed: int = 0,
) -> AttackResult:
    """Realistic worst-case: resize → JPEG → crop → JPEG.

    Args:
        image: (H, W, 3) uint8 RGB.
        max_dim: Resize target.
        jpeg_q1: First JPEG quality.
        crop_ratio: Crop percentage.
        jpeg_q2: Second JPEG quality.
        seed: RNG seed for crop.

    Returns:
        AttackResult after the full chain.
    """
    result = resize_attack(image, max_dim).image
    result = jpeg_compression(result, jpeg_q1).image
    result = random_crop(result, crop_ratio, seed).image
    result = jpeg_compression(result, jpeg_q2).image
    return AttackResult(
        image=result,
        name=f"chain_resize{max_dim}_q{jpeg_q1}_crop{int(crop_ratio*100)}_q{jpeg_q2}",
    )


# ---------------------------------------------------------------------------
# Default attack suite configuration
# ---------------------------------------------------------------------------

def get_default_attacks() -> list[tuple[str, Callable[..., AttackResult], dict]]:
    """Return the default parameterized attack suite.

    Returns:
        List of (name, attack_fn, kwargs) tuples covering all 7 attack types
        with standard social media parameters.
    """
    attacks = []

    # JPEG compression
    for q in [50, 60, 70, 80, 85, 90]:
        attacks.append((f"jpeg_q{q}", jpeg_compression, {"quality": q}))

    # Resize
    for dim in [1080, 1440, 2048]:
        attacks.append((f"resize_{dim}", resize_attack, {"max_dim": dim}))

    # Random crop
    for ratio in [0.1, 0.2, 0.3, 0.4]:
        attacks.append((f"crop_{int(ratio*100)}pct", random_crop, {"crop_ratio": ratio}))

    # Screenshot simulation
    attacks.append(("screenshot", screenshot_simulation, {}))

    # Format conversion chain
    attacks.append(("format_chain", format_conversion, {}))

    # Gaussian noise
    for sigma in [2, 5, 10]:
        attacks.append((f"noise_sigma{sigma}", gaussian_noise, {"sigma": sigma}))

    # Combined chain
    attacks.append(("combined_chain", combined_chain, {}))

    return attacks


def run_attack(image: np.ndarray, attack_name: str, **kwargs) -> AttackResult:
    """Run a single named attack on an image.

    Args:
        image: (H, W, 3) uint8 RGB.
        attack_name: One of 'jpeg', 'resize', 'crop', 'screenshot',
                     'format', 'noise', 'chain'.
        **kwargs: Attack-specific parameters.

    Returns:
        AttackResult.
    """
    dispatch = {
        "jpeg": jpeg_compression,
        "resize": resize_attack,
        "crop": random_crop,
        "screenshot": screenshot_simulation,
        "format": format_conversion,
        "noise": gaussian_noise,
        "chain": combined_chain,
    }
    fn = dispatch.get(attack_name)
    if fn is None:
        raise ValueError(f"Unknown attack: {attack_name}. Options: {list(dispatch.keys())}")
    return fn(image, **kwargs)
