"""Crop synchronization via periodic PN spread-spectrum signals.

Embeds low-amplitude periodic pseudo-noise sequences into the Y channel.
After cropping, the crop offset is recovered via FFT cross-correlation,
enabling correct tile alignment for watermark extraction.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft


def generate_sync_sequences(
    seed: int,
    sync_period: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate horizontal and vertical PN sync sequences.

    Args:
        seed: PRNG seed (derived from secret key).
        sync_period: Period of the sync signal. Shorter periods give more
                     robust detection (more periods to average over).

    Returns:
        Tuple of (seq_h, seq_v), each a 1D float64 array of +/-1 values
        with length sync_period.
    """
    rng_h = np.random.default_rng(seed + 1000)
    rng_v = np.random.default_rng(seed + 2000)
    seq_h = rng_h.choice([-1.0, 1.0], size=sync_period)
    seq_v = rng_v.choice([-1.0, 1.0], size=sync_period)
    return seq_h, seq_v


def embed_sync_pattern(
    y_channel: np.ndarray,
    seed: int,
    sync_period: int = 32,
    amplitude: float = 3.0,
) -> np.ndarray:
    """Add periodic sync patterns to Y channel for crop offset detection.

    Embeds two independent periodic PN sequences:
    - Horizontal: varies along columns with given period
    - Vertical: varies along rows with given period

    The amplitude is low enough to be visually imperceptible but strong
    enough to survive JPEG compression and be detected via cross-correlation.

    Args:
        y_channel: (H, W) float64 luminance array.
        seed: PRNG seed for sync sequence generation.
        sync_period: Period of the sync signals.
        amplitude: Signal amplitude in pixel value units.

    Returns:
        (H, W) float64 Y channel with embedded sync pattern.
    """
    seq_h, seq_v = generate_sync_sequences(seed, sync_period)
    h, w = y_channel.shape
    result = y_channel.copy()
    sync_row = np.tile(seq_h, (w // sync_period) + 1)[:w]
    sync_col = np.tile(seq_v, (h // sync_period) + 1)[:h]
    result += amplitude * sync_row[np.newaxis, :]
    result += amplitude * sync_col[:, np.newaxis]
    return result


def detect_crop_offset(
    y_channel: np.ndarray,
    seed: int,
    sync_period: int = 32,
    tile_size: int = 256,
) -> list[tuple[int, int]]:
    """Detect candidate tile-aligned offsets after cropping.

    Uses FFT cross-correlation to detect the crop offset modulo sync_period,
    then generates candidate offsets modulo tile_size.

    Args:
        y_channel: (H, W) float64 luminance array (cropped/attacked).
        seed: PRNG seed (must match embedding).
        sync_period: Must match embedding.
        tile_size: Tile size used during embedding.

    Returns:
        List of (row_offset, col_offset) candidates for tile alignment
        in the cropped image, sorted by most likely first. Each offset
        indicates where a complete tile from the original grid starts.
    """
    seq_h, seq_v = generate_sync_sequences(seed, sync_period)

    c_mod = _detect_offset_mod(y_channel, seq_h, sync_period, axis=1)
    r_mod = _detect_offset_mod(y_channel, seq_v, sync_period, axis=0)

    h, w = y_channel.shape
    candidates = []
    step = sync_period

    for r_mult in range(tile_size // step):
        crop_r = (r_mod + r_mult * step) % tile_size
        r_off = (tile_size - crop_r) % tile_size
        for c_mult in range(tile_size // step):
            crop_c = (c_mod + c_mult * step) % tile_size
            c_off = (tile_size - crop_c) % tile_size
            # Only include if at least one complete tile fits
            if r_off + tile_size <= h and c_off + tile_size <= w:
                candidates.append((int(r_off), int(c_off)))

    return candidates


def _detect_offset_mod(
    y_channel: np.ndarray,
    seq: np.ndarray,
    sync_period: int,
    axis: int,
) -> int:
    """Detect crop offset modulo sync_period along one axis.

    Averages across the perpendicular axis, folds into one period,
    and uses FFT cross-correlation to find the phase offset.
    """
    # Average across perpendicular axis
    if axis == 1:  # horizontal offset
        avg = np.mean(y_channel, axis=0)
    else:  # vertical offset
        avg = np.mean(y_channel, axis=1)

    avg = avg - np.mean(avg)  # remove DC
    w = len(avg)

    # Fold into one period for better SNR
    n_periods = w // sync_period
    folded = np.zeros(sync_period)
    for i in range(max(n_periods, 1)):
        end = min((i + 1) * sync_period, w)
        length = end - i * sync_period
        folded[:length] += avg[i * sync_period : end]
    if n_periods > 0:
        folded /= n_periods

    # FFT cross-correlation
    corr = np.real(ifft(fft(folded) * np.conj(fft(seq))))
    return int((sync_period - np.argmax(corr)) % sync_period)
