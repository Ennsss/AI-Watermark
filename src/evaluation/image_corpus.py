"""Test image corpus builder for evaluation.

Builds a 25+ image corpus from three sources:
- scikit-image sample data (natural photos, textures)
- Synthetic images (NumPy-generated with fixed seeds)
- Existing test fixtures (artistic styles)

All images are resized to target dimensions and converted to 3-channel RGB uint8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Category definitions for stratified analysis
IMAGE_CATEGORIES: dict[str, list[str]] = {
    "natural_photo": ["astronaut", "camera", "coffee", "chelsea", "rocket", "moon", "peppers"],
    "natural_complex": ["mandrill", "hubble_deep_field", "mixed_media"],
    "texture": ["brick", "grass", "gravel", "noise_field"],
    "synthetic_smooth": ["flat_gray", "radial_gradient", "gradient"],
    "art_style": ["cel_shaded", "painterly"],
    "line_art": ["line_art", "high_contrast_edges", "checkerboard"],
}

# Reverse mapping: image name → category
_NAME_TO_CATEGORY: dict[str, str] = {}
for _cat, _names in IMAGE_CATEGORIES.items():
    for _name in _names:
        _NAME_TO_CATEGORY[_name] = _cat


@dataclass
class CorpusImage:
    """A single image in the evaluation corpus."""

    name: str
    category: str
    image: np.ndarray  # (H, W, 3) uint8 RGB
    source: str  # "skimage", "synthetic", "fixture"


@dataclass
class ImageCorpus:
    """Collection of test images for evaluation."""

    images: dict[str, CorpusImage] = field(default_factory=dict)
    target_size: int = 512

    def __len__(self) -> int:
        return len(self.images)

    def filter_by_category(self, category: str) -> dict[str, CorpusImage]:
        """Return images matching a category."""
        return {
            name: img for name, img in self.images.items()
            if img.category == category
        }

    def get_images_dict(self) -> dict[str, np.ndarray]:
        """Return {name: ndarray} dict compatible with benchmark runner."""
        return {name: img.image for name, img in self.images.items()}

    def get_categories(self) -> list[str]:
        """Return sorted list of categories present in corpus."""
        return sorted({img.category for img in self.images.values()})

    @classmethod
    def build(
        cls,
        fixtures_dir: str | Path | None = None,
        target_size: int = 512,
    ) -> ImageCorpus:
        """Build the full evaluation corpus.

        Args:
            fixtures_dir: Path to tests/fixtures/ directory. If None, skips fixtures.
            target_size: Target dimension (images resized to target_size x target_size).

        Returns:
            ImageCorpus with 25+ images.
        """
        corpus = cls(target_size=target_size)

        # Source 1: scikit-image data
        corpus._load_skimage_images(target_size)

        # Source 2: Synthetic images
        corpus._load_synthetic_images(target_size)

        # Source 3: Test fixtures
        if fixtures_dir is not None:
            corpus._load_fixture_images(Path(fixtures_dir), target_size)

        return corpus

    def _load_skimage_images(self, target_size: int) -> None:
        """Load images from skimage.data."""
        import skimage.data

        skimage_images = {
            "astronaut": skimage.data.astronaut,
            "camera": skimage.data.camera,
            "coffee": skimage.data.coffee,
            "chelsea": skimage.data.chelsea,
            "rocket": skimage.data.rocket,
            "hubble_deep_field": skimage.data.hubble_deep_field,
            "brick": skimage.data.brick,
            "grass": skimage.data.grass,
            "gravel": skimage.data.gravel,
            "moon": skimage.data.moon,
            "checkerboard": skimage.data.checkerboard,
        }

        for name, loader in skimage_images.items():
            img = loader()
            img = _ensure_rgb(img)
            img = _resize_to_square(img, target_size)
            category = _NAME_TO_CATEGORY.get(name, "unknown")
            self.images[name] = CorpusImage(
                name=name, category=category, image=img, source="skimage",
            )

    def _load_synthetic_images(self, target_size: int) -> None:
        """Generate synthetic test images with fixed seeds."""
        synthetics = {
            "flat_gray": _generate_flat_gray,
            "radial_gradient": _generate_radial_gradient,
            "noise_field": _generate_noise_field,
            "high_contrast_edges": _generate_high_contrast_edges,
        }

        for name, generator in synthetics.items():
            img = generator(target_size)
            category = _NAME_TO_CATEGORY.get(name, "unknown")
            self.images[name] = CorpusImage(
                name=name, category=category, image=img, source="synthetic",
            )

    def _load_fixture_images(self, fixtures_dir: Path, target_size: int) -> None:
        """Load images from test fixtures directory."""
        fixture_names = [
            "cel_shaded", "gradient", "line_art", "mandrill",
            "mixed_media", "painterly", "peppers",
        ]

        for name in fixture_names:
            path = fixtures_dir / f"{name}.png"
            if not path.exists():
                continue

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = _resize_to_square(img, target_size)
            category = _NAME_TO_CATEGORY.get(name, "unknown")
            self.images[name] = CorpusImage(
                name=name, category=category, image=img, source="fixture",
            )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale to 3-channel RGB if needed."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB
        return image[:, :, :3]
    return image


def _resize_to_square(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target_size x target_size using Lanczos interpolation."""
    if image.shape[0] == target_size and image.shape[1] == target_size:
        return image
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(
        bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _generate_flat_gray(size: int) -> np.ndarray:
    """Uniform gray image (worst case for visibility — any artifact shows)."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    return img


def _generate_radial_gradient(size: int) -> np.ndarray:
    """Smooth radial gradient from center (white) to edges (black)."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    gray = (1.0 - dist / max_dist) * 255
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def _generate_noise_field(size: int) -> np.ndarray:
    """Random noise (high texture energy, should embed well)."""
    rng = np.random.default_rng(seed=12345)
    noise = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return noise


def _generate_high_contrast_edges(size: int) -> np.ndarray:
    """High-contrast edge pattern (alternating black/white stripes)."""
    img = np.zeros((size, size), dtype=np.uint8)
    stripe_width = max(size // 32, 4)
    for i in range(0, size, stripe_width * 2):
        img[i : i + stripe_width, :] = 255
        img[:, i : i + stripe_width] ^= 255  # Cross-hatch
    return np.stack([img, img, img], axis=2)
