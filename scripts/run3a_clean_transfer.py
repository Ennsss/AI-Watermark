"""Run 3A: evaluation-only clean transfer of the frozen Run 2 decoder."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import compute_ber, extract_from_image
from watermark.preprocessor import extract_y_channel, load_image, rgb_to_ycbcr

from diagnose_run1_signal_localization import location_rows
from run_cnn_benchmark import embed_image


VALIDATION_DIR = ROOT / "data/curated/val"
RUN2_DIR = ROOT / "experiments/run2_seed_aware_overfit"
OUTPUT_DIR = ROOT / "experiments/run3a_clean_transfer"
MODEL_PATH = RUN2_DIR / "seed_aware_overfit.keras"
IMAGE_COUNT = 100
PAYLOADS_PER_IMAGE = 2
PAYLOAD_SEED = 20260809
COEFFICIENT_SEED = 42
PAYLOAD_BITS = 128
DELTA = 16.0
THRESHOLD = 0.5


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(values.size),
        "mean_ber": float(values.mean()),
        "median_ber": float(np.median(values)),
        "standard_deviation": float(values.std()),
        "minimum_ber": float(values.min()),
        "maximum_ber": float(values.max()),
        "perfect_count": int(np.sum(values == 0.0)),
        "perfect_rate": float(np.mean(values == 0.0)),
    }


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "median": None,
        }
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "median": float(np.median(values)),
    }


def construct_features(coefficient_map: np.ndarray, locations: list[dict]) -> np.ndarray:
    features = np.empty((PAYLOAD_BITS, 4), dtype=np.float32)
    for location in locations:
        bit = location["bit_index"]
        coefficient = float(
            coefficient_map[location["row"], location["column"], location["channel"]]
        )
        scaled = coefficient / DELTA
        features[bit] = [
            scaled,
            np.sin(np.pi * scaled),
            np.cos(np.pi * scaled),
            float(location["channel"]),
        ]
    return features


def main() -> int:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for frozen Run 3A inference.") from exc
    if not MODEL_PATH.exists():
        raise SystemExit(f"Frozen Run 2 checkpoint not found: {MODEL_PATH}")

    validation_paths = sorted(VALIDATION_DIR.glob("*.png"))[:IMAGE_COUNT]
    if len(validation_paths) != IMAGE_COUNT:
        raise SystemExit(f"Expected {IMAGE_COUNT} validation PNGs, found {len(validation_paths)}.")
    locations = location_rows((128, 128))
    rng = np.random.default_rng(PAYLOAD_SEED)
    feature_tensors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    classical_predictions: list[np.ndarray] = []
    sample_rows: list[dict] = []
    payload_rows: list[dict] = []

    for image_index, image_path in enumerate(validation_paths, start=1):
        image = resize_square(center_crop_square(load_image(image_path)), TARGET_SIZE)
        original_y = extract_y_channel(rgb_to_ycbcr(image))
        print(f"[validation] {image_index}/{IMAGE_COUNT} {image_path.name}")
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            classical_bits, _ = extract_from_image(
                watermarked,
                PAYLOAD_BITS,
                COEFFICIENT_SEED,
                DELTA,
                "haar",
                target_subbands=("lh2", "hl2"),
            )
            classical_ber = compute_ber(bits, classical_bits)
            coefficient_map = prepare_cnn_input_from_image(watermarked, wavelet="haar")
            feature_tensors.append(construct_features(coefficient_map, locations))
            targets.append(bits)
            classical_predictions.append(classical_bits)
            sample_rows.append(
                {
                    "sample_id": len(sample_rows),
                    "image_id": image_path.stem,
                    "image_filename": image_path.name,
                    "payload_index": payload_index,
                    "payload_fingerprint": fingerprint,
                    "classical_ber": classical_ber,
                    "classical_perfect": classical_ber == 0.0,
                    "original_pct_y_eq_255": float(100.0 * np.mean(original_y == 255.0)),
                    "original_pct_y_ge_247": float(100.0 * np.mean(original_y >= 247.0)),
                    "original_pct_y_eq_0": float(100.0 * np.mean(original_y == 0.0)),
                    "original_pct_y_le_8": float(100.0 * np.mean(original_y <= 8.0)),
                }
            )
            payload_rows.append(
                {
                    "sample_id": len(payload_rows),
                    "image_filename": image_path.name,
                    "payload_index": payload_index,
                    "payload_rng_seed": PAYLOAD_SEED,
                    "payload_fingerprint": fingerprint,
                    "target_zeros": int(np.sum(bits == 0)),
                    "target_ones": int(np.sum(bits == 1)),
                }
            )

    features = np.stack(feature_tensors).astype(np.float32)
    target_array = np.stack(targets).astype(np.uint8)
    classical_array = np.stack(classical_predictions).astype(np.uint8)
    if features.shape != (200, 128, 4) or target_array.shape != (200, 128):
        raise RuntimeError(f"Unexpected Run 3A shapes: {features.shape}, {target_array.shape}")
    if np.isnan(features).any() or np.isinf(features).any():
        raise RuntimeError("Run 3A feature representation contains non-finite values.")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    if tuple(model.input_shape[1:]) != (128, 4) or tuple(model.output_shape[1:]) != (128,):
        raise RuntimeError(f"Frozen checkpoint shape mismatch: {model.input_shape}, {model.output_shape}")
    probabilities = np.asarray(model.predict(features, batch_size=8, verbose=0))
    cnn_predictions = (probabilities >= THRESHOLD).astype(np.uint8)
    classical_errors = classical_array != target_array
    cnn_errors = cnn_predictions != target_array
    classical_bers = classical_errors.mean(axis=1)
    cnn_bers = cnn_errors.mean(axis=1)
    confidences = np.abs(probabilities - 0.5) * 2.0
    for index, row in enumerate(sample_rows):
        row.update(
            {
                "cnn_ber": float(cnn_bers[index]),
                "cnn_perfect": bool(cnn_bers[index] == 0.0),
                "cnn_mean_confidence": float(confidences[index].mean()),
            }
        )

    clean_mask = classical_bers == 0.0
    impaired_mask = ~clean_mask
    all_summary = {
        "classical": metric_summary(classical_bers),
        "cnn": metric_summary(cnn_bers),
    }
    clean_subset = {
        "subset_condition": "classical BER = 0",
        "sample_count": int(clean_mask.sum()),
        "cnn": metric_summary(cnn_bers[clean_mask]),
    }
    impaired_subset = {
        "subset_condition": "classical BER > 0",
        "sample_count": int(impaired_mask.sum()),
        "classical_mean_ber": float(classical_bers[impaired_mask].mean()),
        "cnn_mean_ber": float(cnn_bers[impaired_mask].mean()),
        "cnn_better_count": int(np.sum(cnn_bers[impaired_mask] < classical_bers[impaired_mask])),
        "equal_count": int(np.sum(cnn_bers[impaired_mask] == classical_bers[impaired_mask])),
        "cnn_worse_count": int(np.sum(cnn_bers[impaired_mask] > classical_bers[impaired_mask])),
    }

    per_bit_cnn_ber = cnn_errors.mean(axis=0)
    per_bit_classical_ber = classical_errors.mean(axis=0)
    target_one_frequency = target_array.mean(axis=0)
    predicted_one_frequency = cnn_predictions.mean(axis=0)
    bit_rows = [
        {
            "bit_index": bit,
            "cnn_ber": float(per_bit_cnn_ber[bit]),
            "classical_ber": float(per_bit_classical_ber[bit]),
            "target_one_frequency": float(target_one_frequency[bit]),
            "predicted_one_frequency": float(predicted_one_frequency[bit]),
        }
        for bit in range(PAYLOAD_BITS)
    ]
    worst_bits = sorted(bit_rows, key=lambda row: (-row["cnn_ber"], row["bit_index"]))[:10]
    best_bits = sorted(bit_rows, key=lambda row: (row["cnn_ber"], row["bit_index"]))[:10]

    correct_probabilities = probabilities[~cnn_errors]
    incorrect_probabilities = probabilities[cnn_errors]
    probability_summary = {
        "all_raw_probabilities": distribution(probabilities.ravel()),
        "fraction_0.45_to_0.55": float(
            np.mean((probabilities >= 0.45) & (probabilities <= 0.55))
        ),
        "all_confidences": distribution(confidences.ravel()),
        "correct_raw_probabilities": distribution(correct_probabilities),
        "correct_confidences": distribution(confidences[~cnn_errors]),
        "incorrect_raw_probabilities": distribution(incorrect_probabilities),
        "incorrect_confidences": distribution(confidences[cnn_errors]),
    }

    cnn_mean = float(cnn_bers.mean())
    if cnn_mean < 0.001:
        classification = "STRONG TRANSFER"
    elif cnn_mean < 0.01:
        classification = "PASS"
    elif cnn_mean < 0.20:
        classification = "PARTIAL TRANSFER"
    else:
        classification = "FAIL / LITTLE GENERALIZATION"

    with (RUN2_DIR / "summary_metrics.json").open(encoding="utf-8") as handle:
        run2_summary = json.load(handle)
    comparison_rows = [
        {
            "run": "Run 2 memorization",
            "images": 4,
            "examples": 32,
            "payloads_seen_during_training": True,
            "mean_ber": run2_summary["mean_training_ber"],
            "perfect_recovery_rate": run2_summary["perfect_payloads"] / 32,
            "model_parameters": run2_summary["parameter_count"],
            "attacks": "none",
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
        },
        {
            "run": "Run 3A validation transfer",
            "images": IMAGE_COUNT,
            "examples": len(sample_rows),
            "payloads_seen_during_training": False,
            "mean_ber": cnn_mean,
            "perfect_recovery_rate": float(np.mean(cnn_bers == 0.0)),
            "model_parameters": int(model.count_params()),
            "attacks": "none",
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
        },
    ]
    selected_image_rows = [
        {"selection_order": index, "image_id": path.stem, "image_filename": path.name}
        for index, path in enumerate(validation_paths, start=1)
    ]
    summary = {
        "classification": classification,
        "all_validation_examples": all_summary,
        "classically_clean_subset": clean_subset,
        "classically_impaired_subset": impaired_subset,
        "payloads": {
            "rng_seed": PAYLOAD_SEED,
            "unique_payloads": len({bits.tobytes() for bits in target_array}),
            "target_zeros": int(np.sum(target_array == 0)),
            "target_ones": int(np.sum(target_array == 1)),
            "overall_target_one_frequency": float(target_array.mean()),
        },
        "per_bit": {
            "cnn_zero_ber_bits": int(np.sum(per_bit_cnn_ber == 0.0)),
            "outputs_always_zero": int(np.sum(predicted_one_frequency == 0.0)),
            "outputs_always_one": int(np.sum(predicted_one_frequency == 1.0)),
            "overall_predicted_one_frequency": float(cnn_predictions.mean()),
            "worst_10": worst_bits,
            "best_10": best_bits,
        },
        "probability_diagnostics": probability_summary,
    }
    config = {
        "experiment": "run3a_clean_zero_training_transfer",
        "evaluation_only": True,
        "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "selection": "first 100 PNG files from data/curated/val in sorted filename order",
        "image_count": IMAGE_COUNT,
        "payloads_per_image": PAYLOADS_PER_IMAGE,
        "examples": len(sample_rows),
        "payload_rng": "numpy.random.default_rng",
        "payload_rng_seed": PAYLOAD_SEED,
        "payload_bits": PAYLOAD_BITS,
        "coefficient_seed": COEFFICIENT_SEED,
        "delta": DELTA,
        "wavelet": "haar",
        "dwt_level": 2,
        "subbands": ["LH2", "HL2"],
        "attacks": [],
        "threshold": THRESHOLD,
        "features_in_order": [
            "coefficient / delta",
            "sin(pi * coefficient / delta)",
            "cos(pi * coefficient / delta)",
            "subband identifier (LH2=0, HL2=1)",
        ],
        "training_or_fine_tuning_performed": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "selected_validation_images.csv", selected_image_rows)
    write_csv(OUTPUT_DIR / "validation_payloads.csv", payload_rows)
    write_csv(OUTPUT_DIR / "per_sample_results.csv", sample_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(
        OUTPUT_DIR / "probability_diagnostics.csv",
        [
            {"group": key, **value}
            for key, value in probability_summary.items()
            if isinstance(value, dict)
        ],
    )
    write_csv(OUTPUT_DIR / "comparison_to_run2.csv", comparison_rows)
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved Run 3A artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
