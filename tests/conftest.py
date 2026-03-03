import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_rgb_image(rng: np.random.Generator) -> np.ndarray:
    """512x512 synthetic RGB image (uint8) with varied texture."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    img[:, :, 0] = np.linspace(40, 220, w, dtype=np.uint8)
    img[:, :, 1] = np.linspace(60, 200, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = 128
    # Add some texture blocks
    img[100:200, 100:200] = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    img[300:400, 300:400] = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    return img


@pytest.fixture
def small_rgb_image(rng: np.random.Generator) -> np.ndarray:
    """64x64 small RGB image for fast tests."""
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_bits(rng: np.random.Generator) -> np.ndarray:
    """256-bit random payload for testing."""
    return rng.integers(0, 2, size=256, dtype=np.uint8)


@pytest.fixture
def line_art_image(rng: np.random.Generator) -> np.ndarray:
    """512x512 line-art style image: white background with sparse black lines."""
    h, w = 512, 512
    img = np.full((h, w, 3), 240, dtype=np.uint8)  # Near-white background
    # Draw a few thin horizontal and vertical lines
    for y in range(50, h, 80):
        img[y : y + 2, :] = 20  # Dark lines
    for x in range(30, w, 100):
        img[:, x : x + 2] = 20
    # Add a diagonal
    for i in range(min(h, w)):
        img[i, i] = 20
    return img


@pytest.fixture
def secret_key() -> bytes:
    """Fixed 32-byte AES key for tests."""
    return b"test_secret_key_for_watermark_32"
