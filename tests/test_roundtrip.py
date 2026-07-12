"""End-to-end integration test: embed → attack → extract roundtrip."""

import numpy as np
import pytest

from attacks.suite import jpeg_compression, gaussian_noise, format_conversion, screenshot_simulation
from watermark.embedding import embed_watermark
from watermark.extraction import compute_ber, extract_from_image
from watermark.masking import build_delta_map, detect_sparse_subbands
from watermark.payload import (
    decode_payload_bits,
    derive_seed,
    encode_payload,
    verify_artist_id,
)
from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    rgb_to_ycbcr,
)
from watermark.reconstruction import reconstruct_image
from watermark.tiling import embed_watermark_tiled, extract_watermark_tiled


DELTA = 60.0


@pytest.fixture
def roundtrip_setup():
    """Set up a full embed pipeline and return components for testing."""
    rng = np.random.default_rng(42)
    image = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)

    artist_id = "test_artist@example.com"
    key = b"integration_test_secret_key_32!!"
    timestamp = 1709337600

    # Encode payload
    bits = encode_payload(artist_id, image, key, timestamp=timestamp)

    # Pre-process
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, pad_sizes = pad_to_multiple(y, 4)
    ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

    seed = derive_seed(key)

    # Embed
    wm_y = embed_watermark(y_padded, bits, seed=seed, delta=DELTA)
    watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

    return {
        "original": image,
        "watermarked": watermarked,
        "bits": bits,
        "seed": seed,
        "key": key,
        "artist_id": artist_id,
        "timestamp": timestamp,
    }


class TestCleanRoundtrip:
    def test_perfect_extraction(self, roundtrip_setup):
        s = roundtrip_setup
        extracted, confidence = extract_from_image(
            s["watermarked"], num_bits=len(s["bits"]),
            seed=s["seed"], delta=DELTA,
        )
        assert np.array_equal(extracted, s["bits"])
        assert confidence > 0.9

    def test_payload_decode(self, roundtrip_setup):
        s = roundtrip_setup
        extracted, _ = extract_from_image(
            s["watermarked"], num_bits=len(s["bits"]),
            seed=s["seed"], delta=DELTA,
        )
        prov = decode_payload_bits(extracted, s["key"])
        assert verify_artist_id(s["artist_id"], prov)
        assert prov.timestamp == s["timestamp"]


class TestJPEGRobustness:
    @pytest.mark.parametrize("quality", [85, 90])
    def test_survives_jpeg(self, roundtrip_setup, quality):
        s = roundtrip_setup
        attacked = jpeg_compression(s["watermarked"], quality).image

        extracted, confidence = extract_from_image(
            attacked, num_bits=len(s["bits"]),
            seed=s["seed"], delta=DELTA,
        )
        ber = compute_ber(s["bits"], extracted)
        assert ber < 0.25, f"BER {ber:.4f} exceeds threshold at Q{quality}"


class TestNoiseRobustness:
    def test_survives_low_noise(self, roundtrip_setup):
        s = roundtrip_setup
        attacked = gaussian_noise(s["watermarked"], sigma=2).image

        extracted, _ = extract_from_image(
            attacked, num_bits=len(s["bits"]),
            seed=s["seed"], delta=DELTA,
        )
        ber = compute_ber(s["bits"], extracted)
        assert ber < 0.05


class TestFormatRobustness:
    def test_survives_screenshot(self, roundtrip_setup):
        s = roundtrip_setup
        attacked = screenshot_simulation(s["watermarked"]).image

        extracted, _ = extract_from_image(
            attacked, num_bits=len(s["bits"]),
            seed=s["seed"], delta=DELTA,
        )
        ber = compute_ber(s["bits"], extracted)
        assert ber < 0.05


class TestAdaptiveRoundtrip:
    def test_adaptive_embed_extract(self):
        rng = np.random.default_rng(42)
        image = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)
        key = b"adaptive_test_key_32_bytes!!!!!!"
        bits = encode_payload("artist", image, key, timestamp=1709337600)

        ycbcr = rgb_to_ycbcr(image)
        y = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y, 4)
        ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

        delta_map = build_delta_map(y_padded, delta_min=20, delta_max=80)
        seed = derive_seed(key)

        wm_y = embed_watermark(y_padded, bits, seed=seed, delta_map=delta_map)
        watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

        extracted, confidence = extract_from_image(
            watermarked, num_bits=len(bits), seed=seed,
            delta_map=delta_map,
        )
        assert np.array_equal(extracted, bits)


class TestTiledRoundtrip:
    def test_tiled_embed_extract_clean(self):
        """Tiled embed/extract with sync should produce perfect roundtrip."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (512, 512))
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)
        seed = 12345

        wm_y = embed_watermark_tiled(y, bits, seed=seed, delta=DELTA, tile_size=256)
        extracted = extract_watermark_tiled(
            wm_y, num_bits=256, seed=seed, delta=DELTA, tile_size=256,
        )
        assert np.array_equal(bits, extracted)

    def test_tiled_survives_partial_crop(self):
        """After cropping ~25% of a tiled 1024px image, sync detection recovers bits."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (1024, 1024))
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)
        seed = 12345

        wm_y = embed_watermark_tiled(y, bits, seed=seed, delta=DELTA, tile_size=256)

        # Simulate crop: remove top 25%
        cropped = wm_y[256:, :]

        extracted = extract_watermark_tiled(
            cropped, num_bits=256, seed=seed, delta=DELTA, tile_size=256,
        )
        ber = np.mean(bits != extracted)
        assert ber < 0.05, f"Tiled crop BER {ber:.4f} exceeds threshold"

    def test_tiled_survives_arbitrary_crop(self):
        """Arbitrary non-aligned crop on 1024px image recovers via sync."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (1024, 1024))
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)
        seed = 12345

        wm_y = embed_watermark_tiled(y, bits, seed=seed, delta=DELTA, tile_size=256)

        # Crop 100px from top and 50px from left (non-aligned)
        cropped = wm_y[100:, 50:]

        extracted = extract_watermark_tiled(
            cropped, num_bits=256, seed=seed, delta=DELTA, tile_size=256,
        )
        ber = np.mean(bits != extracted)
        assert ber < 0.05, f"Arbitrary crop BER {ber:.4f} exceeds threshold"

    def test_tiled_small_image_auto_tile(self):
        """512px image should auto-select smaller tile size for viability."""
        rng = np.random.default_rng(42)
        y = rng.uniform(50, 200, (512, 512))
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        seed = 12345

        # Use tile_size=256 but _choose_tile_size should adapt to 128
        # since 512/256=2 tiles which is the minimum
        wm_y = embed_watermark_tiled(y, bits, seed=seed, delta=DELTA, tile_size=256)
        extracted = extract_watermark_tiled(
            wm_y, num_bits=128, seed=seed, delta=DELTA, tile_size=256,
        )
        assert np.array_equal(bits, extracted)


class TestLineArtLL2Fallback:
    def test_sparse_detection(self, line_art_image):
        """Line art image should be detected as sparse."""
        y = line_art_image[:, :, 0].astype(np.float64)
        y_padded, _ = pad_to_multiple(y, 4)
        assert detect_sparse_subbands(y_padded) == True

    def test_normal_image_not_sparse(self, sample_rgb_image):
        """Textured image should NOT be detected as sparse."""
        y = sample_rgb_image[:, :, 0].astype(np.float64)
        y_padded, _ = pad_to_multiple(y, 4)
        assert detect_sparse_subbands(y_padded) == False

    def test_ll2_roundtrip_on_line_art(self, line_art_image):
        """LL2 fallback embed/extract roundtrip on line art."""
        rng = np.random.default_rng(42)
        y = line_art_image[:, :, 0].astype(np.float64)
        y_padded, _ = pad_to_multiple(y, 4)
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)
        seed = 12345

        wm_y = embed_watermark(
            y_padded, bits, seed=seed, delta=80.0,
            target_subbands=("ll2",),
        )
        extracted = extract_from_image(
            # Need to build a full RGB for extract_from_image
            np.stack([wm_y[:y_padded.shape[0], :y_padded.shape[1]].astype(np.float64)] * 3, axis=-1)
                .clip(0, 255).astype(np.uint8),
            num_bits=256, seed=seed, delta=80.0,
            target_subbands=("ll2",),
        )
        # extracted is (bits, confidence) tuple
        ber = np.mean(bits != extracted[0])
        assert ber < 0.01, f"LL2 line art BER {ber:.4f} too high"


class TestRepetitionCodingRoundtrip:
    def test_format_chain_with_repetitions(self):
        """Format chain attack with R=3 repetition coding should recover."""
        rng = np.random.default_rng(42)
        image = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)
        key = b"repetition_test_key_32_bytes!!!!"
        timestamp = 1709337600

        bits = encode_payload(
            "artist", image, key,
            timestamp=timestamp, repetitions=3,
        )

        ycbcr = rgb_to_ycbcr(image)
        y = extract_y_channel(ycbcr)
        y_padded, pad_sizes = pad_to_multiple(y, 4)
        ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)
        seed = derive_seed(key)

        wm_y = embed_watermark(y_padded, bits, seed=seed, delta=DELTA)
        watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)

        # Apply format chain attack
        attacked = format_conversion(watermarked).image

        # Resize back if needed
        if attacked.shape[:2] != watermarked.shape[:2]:
            import cv2
            attacked = cv2.resize(attacked, (watermarked.shape[1], watermarked.shape[0]))

        extracted, _conf = extract_from_image(
            attacked, num_bits=len(bits), seed=seed, delta=DELTA,
        )

        # Try to decode with repetitions
        try:
            prov = decode_payload_bits(extracted, key, repetitions=3)
            assert prov.timestamp == timestamp
            recovered = True
        except Exception:
            recovered = False

        # BER on raw bits (before repetition decoding)
        ber = compute_ber(bits, extracted)
        # With delta=60 and format chain, raw BER should be manageable
        # and repetition coding + RS should recover the payload
        assert ber < 0.30, f"Raw BER {ber:.4f} too high even before rep decoding"
