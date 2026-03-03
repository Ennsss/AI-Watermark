"""F8: CLI interface — embed, extract, benchmark subcommands."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed a watermark into an image."""
    from watermark.embedding import embed_watermark
    from watermark.masking import build_delta_map
    from watermark.payload import derive_seed, encode_payload
    from watermark.preprocessor import (
        extract_y_channel,
        load_image,
        pad_to_multiple,
        rgb_to_ycbcr,
        save_image,
    )
    from watermark.reconstruction import reconstruct_image

    print(f"Loading image: {args.input}")
    image = load_image(args.input)
    print(f"  Size: {image.shape[1]}x{image.shape[0]}")

    key = args.key.encode("utf-8")
    timestamp = args.timestamp

    # Encode payload
    print(f"Building payload for artist: {args.artist_id}")
    bits = encode_payload(
        args.artist_id, image, key,
        timestamp=timestamp, rs_nsym=args.rs_nsym,
        repetitions=args.repetitions,
    )
    print(f"  Payload: {len(bits)} bits ({len(bits) // 8} bytes)")

    # Pre-process
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, pad_sizes = pad_to_multiple(y, 4)
    ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

    # Adaptive masking
    delta_map = None
    if args.adaptive:
        print(f"  Adaptive masking: delta [{args.delta_min}, {args.delta_max}]")
        delta_map = build_delta_map(
            y_padded, wavelet=args.wavelet,
            delta_min=args.delta_min, delta_max=args.delta_max,
        )

    seed = derive_seed(key)

    # Embed
    t0 = time.perf_counter()
    wm_y = embed_watermark(
        y_padded, bits, seed=seed, delta=args.delta,
        wavelet=args.wavelet, delta_map=delta_map,
    )
    watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)
    elapsed = time.perf_counter() - t0

    # Save
    output = args.output or str(Path(args.input).stem) + "_watermarked.png"
    save_image(output, watermarked)

    # SSIM
    from skimage.metrics import structural_similarity as ssim
    score = ssim(image, watermarked, channel_axis=2)

    print(f"  Wavelet: {args.wavelet}, Delta: {args.delta}")
    print(f"  Embed time: {elapsed:.3f}s")
    print(f"  SSIM: {score:.4f}")
    print(f"  Saved: {output}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract a watermark from an image."""
    from watermark.extraction import extract_from_image
    from watermark.masking import build_delta_map
    from watermark.payload import (
        decode_payload_bits,
        derive_seed,
        verify_artist_id,
    )
    from watermark.preprocessor import (
        extract_y_channel,
        load_image,
        pad_to_multiple,
        rgb_to_ycbcr,
    )

    print(f"Loading image: {args.input}")
    image = load_image(args.input)
    print(f"  Size: {image.shape[1]}x{image.shape[0]}")

    key = args.key.encode("utf-8")

    # Build delta map if adaptive
    delta_map = None
    if args.adaptive:
        ycbcr = rgb_to_ycbcr(image)
        y = extract_y_channel(ycbcr)
        y_padded, _ = pad_to_multiple(y, 4)
        delta_map = build_delta_map(
            y_padded, wavelet=args.wavelet,
            delta_min=args.delta_min, delta_max=args.delta_max,
        )

    seed = derive_seed(key)

    t0 = time.perf_counter()
    extracted_bits, confidence = extract_from_image(
        image, num_bits=args.num_bits, seed=seed,
        delta=args.delta, wavelet=args.wavelet, delta_map=delta_map,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Extracted {len(extracted_bits)} bits in {elapsed:.3f}s")
    print(f"  Confidence: {confidence:.4f}")

    # Try to decode payload
    try:
        prov = decode_payload_bits(
            extracted_bits, key, rs_nsym=args.rs_nsym,
            repetitions=args.repetitions,
        )
        print(f"\n  Provenance recovered successfully:")
        print(f"    Artist ID hash: {prov.artist_id}")
        print(f"    Timestamp:      {prov.timestamp}")
        print(f"    pHash:          {prov.phash:#018x}")

        if args.verify_artist:
            match = verify_artist_id(args.verify_artist, prov)
            print(f"    Artist match:   {'YES' if match else 'NO'} ({args.verify_artist})")
    except Exception as e:
        print(f"\n  Payload recovery FAILED: {e}")
        return 1

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the robustness benchmark."""
    from benchmark.runner import BenchmarkConfig, run_benchmark
    from watermark.preprocessor import load_image

    # Load images
    images = {}
    for path in args.images:
        p = Path(path)
        if p.is_dir():
            for img_path in sorted(p.glob("*")):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                    images[img_path.name] = load_image(str(img_path))
        else:
            images[p.name] = load_image(str(p))

    if not images:
        print("Error: No images found.")
        return 1

    print(f"Loaded {len(images)} image(s)")

    config = BenchmarkConfig(
        wavelet=args.wavelet,
        delta=args.delta,
        adaptive=args.adaptive,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        rs_nsym=args.rs_nsym,
        key=args.key.encode("utf-8"),
        artist_id=args.artist_id,
        repetitions=args.repetitions,
    )

    summary = run_benchmark(images, config)
    summary.print_summary()

    if args.output:
        summary.to_csv(args.output)
        print(f"Results exported to: {args.output}")

    # Print per-attack breakdown
    for r in summary.results:
        status = "OK" if r.recovery_success else "FAIL"
        print(f"  {r.image_name:20s} {r.attack_name:30s} "
              f"BER={r.ber_pre_ecc:.4f} SSIM={r.ssim_score:.4f} [{status}]")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="watermark",
        description="Frequency-domain steganographic watermarking for digital art provenance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- embed ---
    p_embed = subparsers.add_parser("embed", help="Embed a watermark into an image.")
    p_embed.add_argument("input", help="Input image path")
    p_embed.add_argument("-o", "--output", help="Output PNG path (default: <input>_watermarked.png)")
    p_embed.add_argument("--artist-id", required=True, help="Artist identifier (name, email, UUID)")
    p_embed.add_argument("--key", required=True, help="Secret key for encryption and PRNG")
    p_embed.add_argument("--timestamp", type=int, default=None, help="UTC epoch timestamp (default: now)")
    p_embed.add_argument("--wavelet", default="haar", choices=["haar", "db4"], help="Wavelet basis")
    p_embed.add_argument("--delta", type=float, default=60.0, help="Base QIM delta (default: 60.0)")
    p_embed.add_argument("--adaptive", action="store_true", help="Use adaptive perceptual masking")
    p_embed.add_argument("--delta-min", type=float, default=20.0, help="Min delta for adaptive mode")
    p_embed.add_argument("--delta-max", type=float, default=80.0, help="Max delta for adaptive mode")
    p_embed.add_argument("--rs-nsym", type=int, default=128, help="Reed-Solomon redundancy symbols")
    p_embed.add_argument("--repetitions", type=int, default=1, help="Repetition coding factor (1=none, 3=3x)")
    p_embed.add_argument("--tiled", action="store_true", help="Use tiled embedding for crop resistance")
    p_embed.add_argument("--tile-size", type=int, default=256, help="Tile size for tiled embedding")
    p_embed.set_defaults(func=cmd_embed)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract a watermark from an image.")
    p_extract.add_argument("input", help="Input watermarked image path")
    p_extract.add_argument("--key", required=True, help="Secret key (must match embedding)")
    p_extract.add_argument("--num-bits", type=int, required=True, help="Expected number of payload bits")
    p_extract.add_argument("--verify-artist", help="Artist ID to verify against extracted payload")
    p_extract.add_argument("--wavelet", default="haar", choices=["haar", "db4"], help="Wavelet basis")
    p_extract.add_argument("--delta", type=float, default=60.0, help="Base QIM delta")
    p_extract.add_argument("--adaptive", action="store_true", help="Use adaptive perceptual masking")
    p_extract.add_argument("--delta-min", type=float, default=20.0, help="Min delta for adaptive mode")
    p_extract.add_argument("--delta-max", type=float, default=80.0, help="Max delta for adaptive mode")
    p_extract.add_argument("--rs-nsym", type=int, default=128, help="Reed-Solomon redundancy symbols")
    p_extract.add_argument("--repetitions", type=int, default=1, help="Repetition coding factor (must match embed)")
    p_extract.add_argument("--tiled", action="store_true", help="Use tiled extraction")
    p_extract.add_argument("--tile-size", type=int, default=256, help="Tile size (must match embed)")
    p_extract.set_defaults(func=cmd_extract)

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Run robustness benchmark.")
    p_bench.add_argument("images", nargs="+", help="Image paths or directories")
    p_bench.add_argument("-o", "--output", help="Output CSV path")
    p_bench.add_argument("--artist-id", default="benchmark_artist", help="Artist ID for test payloads")
    p_bench.add_argument("--key", default="benchmark_key_32bytes!!!!!!!!!", help="Secret key")
    p_bench.add_argument("--wavelet", default="haar", choices=["haar", "db4"], help="Wavelet basis")
    p_bench.add_argument("--delta", type=float, default=60.0, help="Base QIM delta")
    p_bench.add_argument("--adaptive", action="store_true", help="Use adaptive masking")
    p_bench.add_argument("--delta-min", type=float, default=20.0, help="Min adaptive delta")
    p_bench.add_argument("--delta-max", type=float, default=80.0, help="Max adaptive delta")
    p_bench.add_argument("--rs-nsym", type=int, default=128, help="RS redundancy symbols")
    p_bench.add_argument("--repetitions", type=int, default=1, help="Repetition coding factor")
    p_bench.add_argument("--tiled", action="store_true", help="Use tiled embedding for crop resistance")
    p_bench.add_argument("--tile-size", type=int, default=256, help="Tile size for tiled embedding")
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
