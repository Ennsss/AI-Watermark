"""Train and run the CNN-assisted extractor on the benchmark attack suite.

This is a practical runner for the optional CNN branch. It generates supervised
training examples by embedding random fixed payloads into source images,
degrading those watermarked images with the same attack suite used by the
classical benchmark, and training the CNN to predict the embedded 128-bit
payload from LH2/HL2 coefficient maps.

Usage from the project root:
    python scripts/run_cnn_benchmark.py tests/fixtures -o cnn_results.csv

For a research-valid run, train on data/curated/train and evaluate on
data/curated/test. Training and evaluating on tests/fixtures is only a smoke
test because it reuses the same tiny image set.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.suite import get_default_attacks
from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image, predict_payload_bits
from watermark.embedding import embed_watermark
from watermark.extraction import compute_ber, extract_from_image
from watermark.models.cnn_decoder import build_cnn_decoder
from watermark.payload import generate_fixed_payload
from watermark.preprocessor import extract_y_channel, load_image, pad_to_multiple, rgb_to_ycbcr
from watermark.reconstruction import reconstruct_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def iter_image_paths(paths: list[str], limit: int | None = None) -> list[Path]:
    """Collect image files from one or more files/directories."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
    if limit is not None:
        found = found[:limit]
    return found


def standardize_image(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Match the curated dataset preprocessing geometry for CNN inputs."""
    return resize_square(center_crop_square(image), size=size)


def embed_image(
    image: np.ndarray,
    bits: np.ndarray,
    delta: float,
    wavelet: str,
    coefficient_seed: int,
) -> np.ndarray:
    """Embed raw payload bits into one RGB image."""
    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, pad_sizes = pad_to_multiple(y, 4)
    ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)
    wm_y = embed_watermark(
        y_padded,
        bits,
        seed=coefficient_seed,
        delta=delta,
        wavelet=wavelet,
        target_subbands=("lh2", "hl2"),
    )
    return reconstruct_image(ycbcr_padded, wm_y, pad_sizes)


def restore_size(attacked: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize attack output back to the reference dimensions if needed."""
    if attacked.shape[:2] == reference.shape[:2]:
        return attacked
    bgr = cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(
        bgr,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def normalize_inputs(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize coefficient tensors channel-wise."""
    mean = inputs.mean(axis=(0, 1, 2), keepdims=True)
    std = inputs.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((inputs - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(inputs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply training-set channel normalization."""
    return ((inputs - mean) / std).astype(np.float32)


def build_training_set(args: argparse.Namespace, attacks: list) -> tuple[np.ndarray, np.ndarray]:
    """Generate CNN training tensors and payload targets."""
    paths = iter_image_paths(args.train_images, limit=args.max_train_images)
    if not paths:
        raise SystemExit("No training images found.")

    rng = random.Random(args.seed)
    tensors: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for img_idx, path in enumerate(paths, start=1):
        image = standardize_image(load_image(path))
        print(f"[train data] {img_idx}/{len(paths)} {path.name}")
        for sample_idx in range(args.samples_per_image):
            payload_seed = args.seed + img_idx * 1000 + sample_idx
            coefficient_seed = args.coefficient_seed + sample_idx
            bits = generate_fixed_payload(args.payload_bits, payload_seed)
            watermarked = embed_image(image, bits, args.delta, args.wavelet, coefficient_seed)

            tensors.append(prepare_cnn_input_from_image(watermarked, wavelet=args.wavelet))
            targets.append(bits.astype(np.float32))

            attack_name, attack_fn, attack_kwargs = rng.choice(attacks)
            attacked = attack_fn(watermarked, **attack_kwargs).image
            attacked = restore_size(attacked, watermarked)
            tensors.append(prepare_cnn_input_from_image(attacked, wavelet=args.wavelet))
            targets.append(bits.astype(np.float32))

    return np.stack(tensors).astype(np.float32), np.stack(targets).astype(np.float32)


def evaluate(
    model,
    args: argparse.Namespace,
    attacks: list,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict[str, object]]:
    """Run classical and CNN extraction over the benchmark suite."""
    paths = iter_image_paths(args.eval_images, limit=args.max_eval_images)
    if not paths:
        raise SystemExit("No evaluation images found.")

    rows: list[dict[str, object]] = []
    for img_idx, path in enumerate(paths, start=1):
        image = standardize_image(load_image(path))
        bits = generate_fixed_payload(args.payload_bits, args.payload_seed)
        watermarked = embed_image(image, bits, args.delta, args.wavelet, args.coefficient_seed)
        print(f"[eval] {img_idx}/{len(paths)} {path.name}")

        eval_cases = [("none", lambda img: img)] + [
            (name, lambda img, fn=fn, kwargs=kwargs: fn(img, **kwargs).image)
            for name, fn, kwargs in attacks
        ]

        for attack_name, attack_runner in eval_cases:
            attacked = attack_runner(watermarked)
            attacked = restore_size(attacked, watermarked)

            t0 = time.perf_counter()
            classical_bits, classical_conf = extract_from_image(
                attacked,
                num_bits=len(bits),
                seed=args.coefficient_seed,
                delta=args.delta,
                wavelet=args.wavelet,
                target_subbands=("lh2", "hl2"),
            )
            classical_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            cnn_tensor = prepare_cnn_input_from_image(attacked, wavelet=args.wavelet)
            cnn_tensor = apply_normalization(cnn_tensor[np.newaxis, ...], mean, std)[0]
            cnn_bits = predict_payload_bits(model, cnn_tensor, threshold=args.threshold)
            cnn_time = time.perf_counter() - t0

            classical_ber = compute_ber(bits, classical_bits)
            cnn_ber = compute_ber(bits, cnn_bits)
            rows.append({
                "image_name": path.name,
                "attack_name": attack_name,
                "wavelet": args.wavelet,
                "delta": args.delta,
                "classical_ber": classical_ber,
                "classical_success": classical_ber == 0.0,
                "classical_confidence": classical_conf,
                "classical_time_s": classical_time,
                "cnn_ber": cnn_ber,
                "cnn_success": cnn_ber == 0.0,
                "cnn_time_s": cnn_time,
                "num_bits": len(bits),
            })
    return rows


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write evaluation rows to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "attack_name",
        "wavelet",
        "delta",
        "classical_ber",
        "classical_success",
        "classical_confidence",
        "classical_time_s",
        "cnn_ber",
        "cnn_success",
        "cnn_time_s",
        "num_bits",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict[str, object]]) -> None:
    """Print a compact comparison summary."""
    classical_mean = float(np.mean([r["classical_ber"] for r in rows]))
    cnn_mean = float(np.mean([r["cnn_ber"] for r in rows]))
    classical_ok = sum(1 for r in rows if r["classical_success"])
    cnn_ok = sum(1 for r in rows if r["cnn_success"])
    total = len(rows)
    print("\nCNN Benchmark Summary")
    print("=====================")
    print(f"Rows:                 {total}")
    print(f"Classical mean BER:   {classical_mean:.4f}")
    print(f"CNN mean BER:         {cnn_mean:.4f}")
    print(f"Classical successes:  {classical_ok}/{total}")
    print(f"CNN successes:        {cnn_ok}/{total}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-assisted extraction.")
    parser.add_argument("eval_images", nargs="+", help="Evaluation image files or directories")
    parser.add_argument(
        "--train-images",
        nargs="+",
        default=None,
        help="Training image files or directories. Defaults to eval images for smoke tests.",
    )
    parser.add_argument("-o", "--output", default="cnn_results.csv")
    parser.add_argument("--model-output", default=None, help="Optional path to save the trained .keras model")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--samples-per-image", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--payload-bits", type=int, default=128)
    parser.add_argument("--payload-seed", type=int, default=42)
    parser.add_argument("--coefficient-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wavelet", default="haar", choices=["haar", "db4"])
    parser.add_argument("--delta", type=float, default=16.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.train_images is None:
        args.train_images = args.eval_images
        print("[warn] --train-images not set; using eval images as a smoke test only.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for the CNN benchmark. Install the ml extra "
            "in a TensorFlow-compatible Python environment, for example: "
            "py -3.11 -m pip install -e \".[dev,ml]\""
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)
    attacks = get_default_attacks()

    train_inputs, train_targets = build_training_set(args, attacks)
    train_inputs, mean, std = normalize_inputs(train_inputs)

    model = build_cnn_decoder(
        input_shape=tuple(train_inputs.shape[1:]),
        output_bits=args.payload_bits,
        learning_rate=args.learning_rate,
    )
    model.fit(
        train_inputs,
        train_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2 if len(train_inputs) >= 10 else 0.0,
        verbose=1,
    )

    if args.model_output:
        model.save(args.model_output)
        print(f"Saved model: {args.model_output}")

    rows = evaluate(model, args, attacks, mean, std)
    write_csv(args.output, rows)
    print_summary(rows)
    print(f"Saved results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
