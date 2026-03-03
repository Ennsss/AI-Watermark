"""Tiled redundant embedding for crop resistance.

Embeds the same payload into independent non-overlapping tiles, plus a
periodic PN sync signal for crop offset detection. After cropping,
the sync signal reveals candidate tile alignments via FFT cross-correlation,
and RS decoding is attempted at each candidate to find the correct one.
"""

from __future__ import annotations

import numpy as np

from watermark.embedding import (
    embed_watermark,
    extract_watermark,
)
from watermark.sync import detect_crop_offset, embed_sync_pattern


def compute_tile_grid(
    shape: tuple[int, int],
    tile_size: int = 256,
) -> list[tuple[int, int, int, int]]:
    """Compute non-overlapping tile positions for an image.

    Args:
        shape: (H, W) of the image or Y channel.
        tile_size: Side length of each square tile. Must be >= 16 and
                   divisible by 4 (for 2-level DWT).

    Returns:
        List of (row_start, col_start, row_end, col_end) tuples.
        Only tiles that fully fit within the image are returned.
    """
    h, w = shape[:2]
    tiles = []
    for r in range(0, h - tile_size + 1, tile_size):
        for c in range(0, w - tile_size + 1, tile_size):
            tiles.append((r, c, r + tile_size, c + tile_size))
    return tiles


def _choose_tile_size(shape: tuple[int, int], requested: int) -> int:
    """Choose an appropriate tile size based on image dimensions.

    For smaller images where the requested tile_size would leave too few
    tiles, automatically reduces to a smaller tile size.

    Args:
        shape: (H, W) of the image.
        requested: Requested tile size.

    Returns:
        Tile size to use (may be smaller than requested).
    """
    h, w = shape[:2]
    tile_size = requested
    # Ensure at least 2x2 tile grid for meaningful redundancy
    while tile_size > 64 and (h // tile_size < 2 or w // tile_size < 2):
        tile_size //= 2
    return tile_size


def embed_watermark_tiled(
    y_channel: np.ndarray,
    bits: np.ndarray,
    seed: int,
    delta: float = 60.0,
    wavelet: str = "haar",
    level: int = 2,
    tile_size: int = 256,
    sync_period: int = 32,
    sync_amplitude: float = 3.0,
) -> np.ndarray:
    """Embed the same payload into each tile, plus a sync signal for crop detection.

    The sync signal is a low-amplitude periodic PN sequence that enables
    crop offset detection via FFT cross-correlation after the image is cropped.

    Args:
        y_channel: (H, W) float64 luminance array (padded to multiple of 4).
        bits: 1D uint8 array of 0s and 1s.
        seed: PRNG seed for spread-spectrum location mapping.
        delta: QIM quantization step.
        wavelet: Wavelet basis.
        level: DWT decomposition levels.
        tile_size: Side length of each tile (must be divisible by 4).
        sync_period: Period of the sync signal for crop detection.
        sync_amplitude: Amplitude of sync signal in pixel value units.

    Returns:
        (H, W) float64 watermarked luminance array with sync signal.
    """
    effective_tile_size = _choose_tile_size(y_channel.shape, tile_size)
    result = y_channel.copy()
    tiles = compute_tile_grid(y_channel.shape, effective_tile_size)

    for r0, c0, r1, c1 in tiles:
        tile = result[r0:r1, c0:c1].copy()
        wm_tile = embed_watermark(
            tile, bits, seed=seed, delta=delta,
            wavelet=wavelet, level=level,
        )
        result[r0:r1, c0:c1] = wm_tile

    # Embed sync pattern on top for crop offset detection
    result = embed_sync_pattern(
        result, seed=seed,
        sync_period=sync_period,
        amplitude=sync_amplitude,
    )

    return result


def _extract_at_offset(
    y_channel: np.ndarray,
    r_off: int,
    c_off: int,
    num_bits: int,
    seed: int,
    delta: float,
    wavelet: str,
    level: int,
    tile_size: int,
) -> list[np.ndarray]:
    """Extract from all complete tiles at a given alignment offset."""
    h, w = y_channel.shape[:2]
    extractions = []
    for r in range(r_off, h - tile_size + 1, tile_size):
        for c in range(c_off, w - tile_size + 1, tile_size):
            tile = y_channel[r : r + tile_size, c : c + tile_size]
            extracted = extract_watermark(
                tile, num_bits=num_bits, seed=seed,
                delta=delta, wavelet=wavelet, level=level,
            )
            extractions.append(extracted)
    return extractions


def _majority_vote(extractions: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Majority vote across tile extractions. Returns (voted_bits, consensus).

    Consensus is the average fraction of tiles that agree with the majority
    for each bit position -- higher means better alignment.
    """
    stacked = np.stack(extractions, axis=0)
    n = len(extractions)
    votes = stacked.sum(axis=0)
    result = (votes > n / 2).astype(np.uint8)
    # Consensus: for each bit, what fraction of tiles agreed with majority
    agreement = np.where(result == 1, votes, n - votes) / n
    consensus = float(agreement.mean())
    return result, consensus


def _try_rs_decode(bits: np.ndarray, key: bytes, rs_nsym: int, repetitions: int) -> bool:
    """Try RS decoding at a candidate offset. Returns True if decode succeeds."""
    try:
        from watermark.payload import decode_payload_bits
        decode_payload_bits(bits, key, rs_nsym=rs_nsym, repetitions=repetitions)
        return True
    except Exception:
        return False


def extract_watermark_tiled(
    y_channel: np.ndarray,
    num_bits: int,
    seed: int,
    delta: float = 60.0,
    wavelet: str = "haar",
    level: int = 2,
    tile_size: int = 256,
    sync_period: int = 32,
    key: bytes | None = None,
    rs_nsym: int = 128,
    repetitions: int = 1,
) -> np.ndarray:
    """Extract watermark using sync-based crop offset detection and majority voting.

    Pipeline:
    1. Use FFT cross-correlation on the sync signal to detect crop offset
       modulo sync_period, generating candidate tile alignments.
    2. For each candidate, extract from all complete tiles and majority vote.
    3. If a key is provided, try RS decoding at each candidate — first success wins.
    4. Otherwise, select the candidate with highest inter-tile consensus.

    Args:
        y_channel: (H, W) float64 luminance array (possibly cropped).
        num_bits: Expected number of embedded bits.
        seed: PRNG seed (must match embedding).
        delta: QIM quantization step (must match embedding).
        wavelet: Wavelet basis (must match embedding).
        level: DWT decomposition levels.
        tile_size: Tile size (must match embedding).
        sync_period: Sync period (must match embedding).
        key: Optional AES key for RS-based candidate selection.
        rs_nsym: RS redundancy symbols for RS-based selection.
        repetitions: Repetition coding factor for RS-based selection.

    Returns:
        1D uint8 array of extracted bits via majority vote.
    """
    effective_tile_size = _choose_tile_size(y_channel.shape, tile_size)

    # Step 1: Get candidate offsets from sync detection
    candidates = detect_crop_offset(
        y_channel, seed=seed,
        sync_period=sync_period,
        tile_size=effective_tile_size,
    )

    # Also include (0, 0) as fallback if not already present
    if (0, 0) not in candidates:
        h, w = y_channel.shape
        if effective_tile_size <= h and effective_tile_size <= w:
            candidates.append((0, 0))

    best_result = None
    best_consensus = -1.0

    for r_off, c_off in candidates:
        extractions = _extract_at_offset(
            y_channel, r_off, c_off, num_bits, seed,
            delta, wavelet, level, effective_tile_size,
        )
        if len(extractions) < 1:
            continue

        if len(extractions) == 1:
            result = extractions[0]
            consensus = 0.5
        else:
            result, consensus = _majority_vote(extractions)

        # RS-based selection: try decoding, first success is correct
        if key is not None:
            if _try_rs_decode(result, key, rs_nsym, repetitions):
                return result

        if consensus > best_consensus:
            best_consensus = consensus
            best_result = result

    if best_result is None:
        # Fallback: extract from full image (no tiling)
        return extract_watermark(
            y_channel, num_bits=num_bits, seed=seed,
            delta=delta, wavelet=wavelet, level=level,
        )

    return best_result
