"""Run 1: deliberately overfit the thesis baseline CNN on 32 clean examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.embedding import _get_embedding_locations
from watermark.extraction import compute_ber, extract_from_image
from watermark.models.cnn_decoder import build_cnn_decoder
from watermark.preprocessor import extract_y_channel, load_image, rgb_to_ycbcr

from run_cnn_benchmark import embed_image, normalize_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/curated/train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/run1_clean_fixed_seed_overfit",
    )
    parser.add_argument("--source-images", type=int, default=4)
    parser.add_argument("--payloads-per-image", type=int, default=8)
    parser.add_argument("--payload-bits", type=int, default=128)
    parser.add_argument("--payload-seed", type=int, default=20260808)
    parser.add_argument("--coefficient-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=16.0)
    parser.add_argument("--wavelet", default="haar", choices=["haar"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--stop-ber", type=float, default=0.001)
    parser.add_argument("--stable-zero-epochs", type=int, default=5)
    return parser.parse_args()


def summarize_probabilities(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.source_images != 4 or args.payloads_per_image != 8:
        raise SystemExit("Run 1 requires exactly 4 source images and 8 payloads per image.")
    if args.coefficient_seed != 42 or args.delta != 16.0:
        raise SystemExit("Run 1 requires coefficient seed 42 and provisional delta 16.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Run 1.") from exc

    candidate_paths = sorted(args.train_dir.glob("*.png"))
    if len(candidate_paths) < args.source_images:
        raise SystemExit(f"Expected at least {args.source_images} PNGs in {args.train_dir}.")

    payload_rng = np.random.default_rng(args.payload_seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    classical_bers: list[float] = []
    sample_rows: list[dict] = []
    candidate_rows: list[dict] = []
    selected_paths: list[Path] = []

    for image_path in candidate_paths:
        image = resize_square(center_crop_square(load_image(image_path)), size=TARGET_SIZE)
        y = extract_y_channel(rgb_to_ycbcr(image))
        candidate_inputs: list[np.ndarray] = []
        candidate_targets: list[np.ndarray] = []
        candidate_bers: list[float] = []
        for payload_index in range(args.payloads_per_image):
            bits = payload_rng.integers(0, 2, args.payload_bits, dtype=np.uint8)
            watermarked = embed_image(
                image, bits, args.delta, args.wavelet, args.coefficient_seed
            )
            cnn_input = prepare_cnn_input_from_image(watermarked, wavelet=args.wavelet)
            extracted, _confidence = extract_from_image(
                watermarked,
                num_bits=args.payload_bits,
                seed=args.coefficient_seed,
                delta=args.delta,
                wavelet=args.wavelet,
                target_subbands=("lh2", "hl2"),
            )
            ber = compute_ber(bits, extracted)
            candidate_inputs.append(cnn_input)
            candidate_targets.append(bits.astype(np.float32))
            candidate_bers.append(ber)

        candidate_array = np.asarray(candidate_bers)
        qualifies = bool(np.all(candidate_array == 0.0))
        candidate_rows.append(
            {
                "screening_order": len(candidate_rows) + 1,
                "source_image": image_path.name,
                "mean_clean_ber": float(candidate_array.mean()),
                "maximum_clean_ber": float(candidate_array.max()),
                "perfect_payload_count": int(np.sum(candidate_array == 0.0)),
                "qualifies": qualifies,
                "pct_y_eq_255": float(100.0 * np.mean(y == 255.0)),
                "pct_y_ge_247": float(100.0 * np.mean(y >= 247.0)),
                "pct_y_eq_0": float(100.0 * np.mean(y == 0.0)),
                "pct_y_le_8": float(100.0 * np.mean(y <= 8.0)),
            }
        )
        print(
            f"[screen] {len(candidate_rows)} {image_path.name} "
            f"perfect={int(np.sum(candidate_array == 0.0))}/8 qualifies={qualifies}"
        )
        if not qualifies:
            continue

        selected_paths.append(image_path)
        for payload_index, (cnn_input, bits, ber) in enumerate(
            zip(candidate_inputs, candidate_targets, candidate_bers)
        ):
            inputs.append(cnn_input)
            targets.append(bits)
            classical_bers.append(ber)
            sample_rows.append(
                {
                    "sample_index": len(sample_rows),
                    "source_image": image_path.name,
                    "payload_index": payload_index,
                    "coefficient_seed": args.coefficient_seed,
                    "classical_ber": ber,
                }
            )
        if len(selected_paths) == args.source_images:
            break

    if len(selected_paths) != args.source_images:
        raise SystemExit("Training split exhausted before four qualifying images were found.")

    train_inputs = np.stack(inputs).astype(np.float32)
    train_targets = np.stack(targets).astype(np.float32)
    half = args.payload_bits // 2
    lh_locations = _get_embedding_locations(train_inputs.shape[1:3], half, args.coefficient_seed)
    hl_locations = _get_embedding_locations(
        train_inputs.shape[1:3], args.payload_bits - half, args.coefficient_seed + 1
    )

    assert train_inputs.shape == (32, 128, 128, 2)
    assert train_targets.shape == (32, 128)
    assert len(lh_locations) == 64 and len(hl_locations) == 64

    classical = np.asarray(classical_bers)
    target_ones = int(train_targets.sum())
    target_total = int(train_targets.size)
    payload_strings = {"".join(map(str, row.astype(np.uint8))) for row in train_targets}
    preflight = {
        "mean_classical_ber": float(classical.mean()),
        "max_classical_ber": float(classical.max()),
        "perfect_classical_recoveries": int(np.sum(classical == 0.0)),
        "samples": int(len(classical)),
        "payload_shape": list(train_targets[0].shape),
        "cnn_input_shape": list(train_inputs[0].shape),
        "dwt_subband_shape": list(train_inputs.shape[1:3]),
        "payload_bit_count": args.payload_bits,
        "lh2_selected_locations": int(len(lh_locations)),
        "hl2_selected_locations": int(len(hl_locations)),
        "coefficient_seed": args.coefficient_seed,
        "target_zeros": target_total - target_ones,
        "target_ones": target_ones,
        "target_one_proportion": target_ones / target_total,
        "unique_payloads": len(payload_strings),
        "candidate_images_screened": len(candidate_rows),
        "selected_source_images": [path.name for path in selected_paths],
    }
    print("Preflight checks")
    print(json.dumps(preflight, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "samples.csv",
        list(sample_rows[0]),
        sample_rows,
    )
    write_rows(
        args.output_dir / "candidate_screening.csv",
        list(candidate_rows[0]),
        candidate_rows,
    )
    with (args.output_dir / "preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2)
    if preflight["perfect_classical_recoveries"] != 32:
        aborted_config = {
            "experiment": "run1_clean_fixed_seed_overfit",
            "source_images": [path.name for path in selected_paths],
            "source_image_count": args.source_images,
            "payloads_per_image": args.payloads_per_image,
            "payload_bits": args.payload_bits,
            "payload_rng_seed": args.payload_seed,
            "coefficient_seed": args.coefficient_seed,
            "model_initialization_seed": args.model_seed,
            "delta": args.delta,
            "wavelet": args.wavelet,
            "dwt_level": 2,
            "subbands": ["LH2", "HL2"],
            "attacks": [],
            "dropout_rate": 0.0,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "maximum_epochs": args.max_epochs,
            "epochs_actually_trained": 0,
        }
        failure = {
            "status": "PREFLIGHT_FAILED",
            "reason": "not all clean samples have classical BER 0",
            **preflight,
        }
        with (args.output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
        with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
            json.dump(aborted_config, handle, indent=2)
        raise SystemExit("Preflight failed: not all clean samples have classical BER 0.")

    train_inputs, norm_mean, norm_std = normalize_inputs(train_inputs)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.model_seed)
    model = build_cnn_decoder(
        input_shape=tuple(train_inputs.shape[1:]),
        output_bits=args.payload_bits,
        learning_rate=args.learning_rate,
        dropout_rate=0.0,
    )

    history_rows: list[dict[str, float | int]] = []

    class BerHistory(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.consecutive_zero_epochs = 0

        def on_epoch_end(self, epoch, logs=None):
            probabilities = self.model.predict(train_inputs, batch_size=args.batch_size, verbose=0)
            predictions = probabilities >= 0.5
            sample_bers = np.mean(predictions != train_targets, axis=1)
            ber = float(sample_bers.mean())
            perfect = int(np.sum(sample_bers == 0.0))
            loss = float((logs or {}).get("loss", np.nan))
            history_rows.append(
                {"epoch": epoch + 1, "loss": loss, "ber": ber, "perfect_recoveries": perfect}
            )
            if epoch == 0 or (epoch + 1) % 10 == 0 or ber == 0.0:
                print(
                    f"epoch={epoch + 1} loss={loss:.8f} "
                    f"ber={ber:.8f} perfect={perfect}/32"
                )
            self.consecutive_zero_epochs = self.consecutive_zero_epochs + 1 if ber == 0.0 else 0
            if self.consecutive_zero_epochs >= args.stable_zero_epochs:
                self.model.stop_training = True

    history_callback = BerHistory()
    model.fit(
        train_inputs,
        train_targets,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        shuffle=True,
        callbacks=[history_callback],
        verbose=0,
    )

    probabilities = np.asarray(model.predict(train_inputs, batch_size=args.batch_size, verbose=0))
    predictions = (probabilities >= 0.5).astype(np.uint8)
    target_bits = train_targets.astype(np.uint8)
    errors = predictions != target_bits
    sample_bers = errors.mean(axis=1)
    per_bit_ber = errors.mean(axis=0)
    predicted_one_frequency = predictions.mean(axis=0)
    target_one_frequency = target_bits.mean(axis=0)
    correct = ~errors

    summary = {
        "status": "PASS" if float(sample_bers.mean()) <= 0.001 else "FAIL",
        "mean_training_ber": float(sample_bers.mean()),
        "median_training_ber": float(np.median(sample_bers)),
        "max_training_ber": float(sample_bers.max()),
        "perfect_recoveries": int(np.sum(sample_bers == 0.0)),
        "samples": int(len(sample_bers)),
        "epochs_trained": len(history_rows),
        "final_bce_loss": float(history_rows[-1]["loss"]),
        "probabilities": {
            **summarize_probabilities(probabilities.ravel()),
            "proportion_0.45_to_0.55": float(np.mean((probabilities >= 0.45) & (probabilities <= 0.55))),
        },
        "correct_prediction_probabilities": summarize_probabilities(probabilities[correct]),
        "incorrect_prediction_probabilities": summarize_probabilities(probabilities[errors]),
        "target_zero_probabilities": summarize_probabilities(probabilities[target_bits == 0]),
        "target_one_probabilities": summarize_probabilities(probabilities[target_bits == 1]),
    }
    config = {
        "experiment": "run1_clean_fixed_seed_overfit",
        "source_images": [path.name for path in selected_paths],
        "candidate_images_screened": len(candidate_rows),
        "source_image_count": args.source_images,
        "payloads_per_image": args.payloads_per_image,
        "payload_bits": args.payload_bits,
        "payload_rng_seed": args.payload_seed,
        "coefficient_seed": args.coefficient_seed,
        "model_initialization_seed": args.model_seed,
        "delta": args.delta,
        "wavelet": args.wavelet,
        "dwt_level": 2,
        "subbands": ["LH2", "HL2"],
        "attacks": [],
        "dropout_rate": 0.0,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "maximum_epochs": args.max_epochs,
        "stop_ber": args.stop_ber,
        "stable_zero_epochs": args.stable_zero_epochs,
        "epochs_actually_trained": len(history_rows),
        "normalization_mean": norm_mean.reshape(-1).astype(float).tolist(),
        "normalization_std": norm_std.reshape(-1).astype(float).tolist(),
    }
    bit_rows = [
        {
            "bit_index": bit,
            "per_bit_ber": float(per_bit_ber[bit]),
            "predicted_one_frequency": float(predicted_one_frequency[bit]),
            "target_one_frequency": float(target_one_frequency[bit]),
        }
        for bit in range(args.payload_bits)
    ]
    write_rows(
        args.output_dir / "training_history.csv",
        ["epoch", "loss", "ber", "perfect_recoveries"],
        history_rows,
    )
    write_rows(
        args.output_dir / "per_bit_diagnostics.csv",
        ["bit_index", "per_bit_ber", "predicted_one_frequency", "target_one_frequency"],
        bit_rows,
    )
    with (args.output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    model.save(args.output_dir / "baseline_cnn_overfit.keras")
    print("Final results")
    print(json.dumps(summary, indent=2))
    print(f"Saved artifacts to {args.output_dir}")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
