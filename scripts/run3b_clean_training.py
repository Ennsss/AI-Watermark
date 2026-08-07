"""Run 3B: proper clean seed-aware training and fixed held-out validation."""

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
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run2_seed_aware_overfit import build_model
from run3a_clean_transfer import construct_features, distribution, metric_summary
from run_cnn_benchmark import embed_image


TRAIN_DIR = ROOT / "data/curated/train"
VAL_DIR = ROOT / "data/curated/val"
RUN2_DIR = ROOT / "experiments/run2_seed_aware_overfit"
RUN3A_DIR = ROOT / "experiments/run3a_clean_transfer"
OUTPUT_DIR = ROOT / "experiments/run3b_clean_training"
TRAIN_IMAGE_COUNT = 500
VAL_IMAGE_COUNT = 100
PAYLOADS_PER_IMAGE = 2
TRAIN_PAYLOAD_SEED = 20260810
VAL_PAYLOAD_SEED = 20260809
MODEL_SEED = 42
COEFFICIENT_SEED = 42
PAYLOAD_BITS = 128
DELTA = 16.0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
THRESHOLD = 0.5


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    paths: list[Path], payload_seed: int, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(payload_seed)
    locations = location_rows((128, 128))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    classical_predictions: list[np.ndarray] = []
    rows: list[dict] = []
    for image_index, image_path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(image_path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[{split} data] {image_index}/{len(paths)} {image_path.name}")
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
            coefficient_map = prepare_cnn_input_from_image(watermarked, wavelet="haar")
            features.append(construct_features(coefficient_map, locations))
            targets.append(bits)
            classical_predictions.append(classical_bits)
            rows.append(
                {
                    "sample_id": len(rows),
                    "split": split,
                    "image_id": image_path.stem,
                    "image_filename": image_path.name,
                    "payload_index": payload_index,
                    "payload_rng_seed": payload_seed,
                    "payload_fingerprint": fingerprint,
                    "target_zeros": int(np.sum(bits == 0)),
                    "target_ones": int(np.sum(bits == 1)),
                    "classical_ber": compute_ber(bits, classical_bits),
                }
            )
    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.uint8),
        np.stack(classical_predictions).astype(np.uint8),
        rows,
    )


def epoch_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= THRESHOLD
    errors = predictions != targets
    sample_bers = errors.mean(axis=1)
    return {
        "ber": float(errors.mean()),
        "lh2_ber": float(errors[:, :64].mean()),
        "hl2_ber": float(errors[:, 64:].mean()),
        "perfect_payloads": int(np.sum(sample_bers == 0.0)),
    }


def main() -> int:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Run 3B.") from exc

    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    if len(train_paths) != TRAIN_IMAGE_COUNT or len(val_paths) != VAL_IMAGE_COUNT:
        raise SystemExit("Required deterministic train/validation image counts are unavailable.")

    train_x, train_y, train_classical, train_rows = generate_dataset(
        train_paths, TRAIN_PAYLOAD_SEED, "train"
    )
    val_x, val_y, val_classical, val_rows = generate_dataset(
        val_paths, VAL_PAYLOAD_SEED, "validation"
    )
    if train_x.shape != (1000, 128, 4) or val_x.shape != (200, 128, 4):
        raise RuntimeError(f"Unexpected feature shapes: {train_x.shape}, {val_x.shape}")
    if np.isnan(train_x).any() or np.isinf(train_x).any():
        raise RuntimeError("Training features contain non-finite values.")

    run3a_payloads = list(csv.DictReader((RUN3A_DIR / "validation_payloads.csv").open(encoding="utf-8")))
    expected_fingerprints = [row["payload_fingerprint"] for row in run3a_payloads]
    actual_fingerprints = [row["payload_fingerprint"] for row in val_rows]
    if actual_fingerprints != expected_fingerprints:
        raise RuntimeError("Run 3B validation payloads do not match Run 3A fingerprints.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model(input_features=4)
    if model.count_params() != 369:
        raise RuntimeError(f"Architecture changed unexpectedly: {model.count_params()} parameters")
    history_rows: list[dict] = []

    class ValidationBerCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_ber = float("inf")
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.best_weights = None
            self.epochs_without_ber_improvement = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_prob = self.model.predict(train_x, batch_size=BATCH_SIZE, verbose=0)
            val_prob = self.model.predict(val_x, batch_size=BATCH_SIZE, verbose=0)
            train_metrics = epoch_metrics(train_prob, train_y)
            val_metrics = epoch_metrics(val_prob, val_y)
            val_ber = val_metrics["ber"]
            val_loss = float(logs.get("val_loss", np.nan))
            strict_ber_improvement = val_ber < self.best_ber - 1e-12
            tie_loss_improvement = abs(val_ber - self.best_ber) <= 1e-12 and val_loss < self.best_loss
            if strict_ber_improvement:
                self.epochs_without_ber_improvement = 0
            else:
                self.epochs_without_ber_improvement += 1
            if strict_ber_improvement or tie_loss_improvement:
                self.best_ber = val_ber
                self.best_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_bce": float(logs.get("loss", np.nan)),
                    "train_ber": train_metrics["ber"],
                    "train_lh2_ber": train_metrics["lh2_ber"],
                    "train_hl2_ber": train_metrics["hl2_ber"],
                    "train_perfect_payloads": train_metrics["perfect_payloads"],
                    "validation_bce": val_loss,
                    "validation_ber": val_metrics["ber"],
                    "validation_lh2_ber": val_metrics["lh2_ber"],
                    "validation_hl2_ber": val_metrics["hl2_ber"],
                    "validation_perfect_payloads": val_metrics["perfect_payloads"],
                    "is_best_checkpoint": self.best_epoch == epoch + 1,
                }
            )
            print(
                f"epoch={epoch + 1} train_ber={train_metrics['ber']:.8f} "
                f"val_ber={val_ber:.8f} val_lh2={val_metrics['lh2_ber']:.8f} "
                f"val_hl2={val_metrics['hl2_ber']:.8f} val_perfect={val_metrics['perfect_payloads']}/200"
            )
            if self.epochs_without_ber_improvement >= PATIENCE:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise RuntimeError("No best Run 3B weights were captured.")
            self.model.set_weights(self.best_weights)

    checkpoint = ValidationBerCheckpoint()
    model.fit(
        train_x,
        train_y.astype(np.float32),
        validation_data=(val_x, val_y.astype(np.float32)),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=[checkpoint],
        verbose=0,
    )
    model.save(OUTPUT_DIR / "best_seed_aware_clean.keras") if OUTPUT_DIR.exists() else None

    val_probabilities = np.asarray(model.predict(val_x, batch_size=BATCH_SIZE, verbose=0))
    val_predictions = (val_probabilities >= THRESHOLD).astype(np.uint8)
    cnn_errors = val_predictions != val_y
    classical_errors = val_classical != val_y
    cnn_bers = cnn_errors.mean(axis=1)
    classical_bers = classical_errors.mean(axis=1)
    confidences = np.abs(val_probabilities - 0.5) * 2.0
    for index, row in enumerate(val_rows):
        row.update(
            {
                "cnn_ber": float(cnn_bers[index]),
                "classical_perfect": bool(classical_bers[index] == 0.0),
                "cnn_perfect": bool(cnn_bers[index] == 0.0),
                "cnn_mean_confidence": float(confidences[index].mean()),
            }
        )

    clean_mask = classical_bers == 0.0
    impaired_mask = ~clean_mask
    per_bit_cnn = cnn_errors.mean(axis=0)
    per_bit_classical = classical_errors.mean(axis=0)
    target_frequency = val_y.mean(axis=0)
    predicted_frequency = val_predictions.mean(axis=0)
    bit_rows = [
        {
            "bit_index": bit,
            "subband": "LH2" if bit < 64 else "HL2",
            "cnn_ber": float(per_bit_cnn[bit]),
            "classical_ber": float(per_bit_classical[bit]),
            "target_one_frequency": float(target_frequency[bit]),
            "predicted_one_frequency": float(predicted_frequency[bit]),
        }
        for bit in range(PAYLOAD_BITS)
    ]
    incorrect_probabilities = val_probabilities[cnn_errors]
    incorrect_targets = val_y[cnn_errors]
    probability_summary = {
        "all_probabilities": distribution(val_probabilities.ravel()),
        "fraction_0.45_to_0.55": float(
            np.mean((val_probabilities >= 0.45) & (val_probabilities <= 0.55))
        ),
        "all_confidences": distribution(confidences.ravel()),
        "incorrect_probabilities": distribution(incorrect_probabilities),
        "incorrect_confidences": distribution(confidences[cnn_errors]),
        "incorrect_target_zero_count": int(np.sum(incorrect_targets == 0)),
        "incorrect_target_one_count": int(np.sum(incorrect_targets == 1)),
    }

    overall_cnn = metric_summary(cnn_bers)
    overall_classical = metric_summary(classical_bers)
    mean_ber = overall_cnn["mean_ber"]
    if mean_ber < 0.001:
        classification = "STRONG PASS"
    elif mean_ber < 0.01:
        classification = "PASS"
    elif mean_ber < 0.05:
        classification = "PARTIAL"
    else:
        classification = "FAIL"
    clean_subset_cnn = metric_summary(cnn_bers[clean_mask])
    impaired = {
        "sample_count": int(impaired_mask.sum()),
        "classical_mean_ber": float(classical_bers[impaired_mask].mean()),
        "cnn_mean_ber": float(cnn_bers[impaired_mask].mean()),
        "cnn_better_count": int(np.sum(cnn_bers[impaired_mask] < classical_bers[impaired_mask])),
        "equal_count": int(np.sum(cnn_bers[impaired_mask] == classical_bers[impaired_mask])),
        "cnn_worse_count": int(np.sum(cnn_bers[impaired_mask] > classical_bers[impaired_mask])),
    }
    subbands = {
        "LH2": {
            "cnn_mean_ber": float(cnn_errors[:, :64].mean()),
            "classical_mean_ber": float(classical_errors[:, :64].mean()),
            "zero_ber_output_positions": int(np.sum(per_bit_cnn[:64] == 0.0)),
        },
        "HL2": {
            "cnn_mean_ber": float(cnn_errors[:, 64:].mean()),
            "classical_mean_ber": float(classical_errors[:, 64:].mean()),
            "zero_ber_output_positions": int(np.sum(per_bit_cnn[64:] == 0.0)),
        },
    }
    worst_bits = sorted(bit_rows, key=lambda row: (-row["cnn_ber"], row["bit_index"]))[:10]
    best_bits = sorted(bit_rows, key=lambda row: (row["cnn_ber"], row["bit_index"]))[:10]

    with (RUN2_DIR / "summary_metrics.json").open(encoding="utf-8") as handle:
        run2 = json.load(handle)
    with (RUN3A_DIR / "summary_metrics.json").open(encoding="utf-8") as handle:
        run3a = json.load(handle)
    comparison_rows = [
        {
            "run": "Run 2",
            "training_images": 4,
            "training_examples": 32,
            "validation_images": "N/A",
            "model_parameters": run2["parameter_count"],
            "overall_validation_ber": "N/A",
            "lh2_ber": "N/A",
            "hl2_ber": "N/A",
            "perfect_recovery_rate": 1.0,
            "attacks": "none",
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
        },
        {
            "run": "Run 3A",
            "training_images": 4,
            "training_examples": 32,
            "validation_images": 100,
            "model_parameters": run2["parameter_count"],
            "overall_validation_ber": run3a["all_validation_examples"]["cnn"]["mean_ber"],
            "lh2_ber": 0.000234375,
            "hl2_ber": 0.138828125,
            "perfect_recovery_rate": run3a["all_validation_examples"]["cnn"]["perfect_rate"],
            "attacks": "none",
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
        },
        {
            "run": "Run 3B",
            "training_images": TRAIN_IMAGE_COUNT,
            "training_examples": len(train_y),
            "validation_images": VAL_IMAGE_COUNT,
            "model_parameters": int(model.count_params()),
            "overall_validation_ber": overall_cnn["mean_ber"],
            "lh2_ber": subbands["LH2"]["cnn_mean_ber"],
            "hl2_ber": subbands["HL2"]["cnn_mean_ber"],
            "perfect_recovery_rate": overall_cnn["perfect_rate"],
            "attacks": "none",
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
        },
    ]
    summary = {
        "classification": classification,
        "best_epoch": checkpoint.best_epoch,
        "epochs_trained": len(history_rows),
        "best_validation_ber_during_training": checkpoint.best_ber,
        "best_validation_bce_tiebreak": checkpoint.best_loss,
        "cnn_all_validation": overall_cnn,
        "classical_all_validation": overall_classical,
        "paired_counts_all": {
            "cnn_better": int(np.sum(cnn_bers < classical_bers)),
            "equal": int(np.sum(cnn_bers == classical_bers)),
            "cnn_worse": int(np.sum(cnn_bers > classical_bers)),
        },
        "subbands": subbands,
        "classically_clean_subset": {
            "sample_count": int(clean_mask.sum()),
            "cnn_mean_ber": clean_subset_cnn["mean_ber"],
            "cnn_perfect_recovery_rate": clean_subset_cnn["perfect_rate"],
        },
        "classically_impaired_subset": impaired,
        "payload_balance": {
            "training_unique_payloads": len({row.tobytes() for row in train_y}),
            "training_zeros": int(np.sum(train_y == 0)),
            "training_ones": int(np.sum(train_y == 1)),
            "training_one_frequency": float(train_y.mean()),
            "validation_unique_payloads": len({row.tobytes() for row in val_y}),
            "validation_payload_fingerprints_match_run3a": True,
            "validation_one_frequency": float(val_y.mean()),
        },
        "per_bit": {
            "zero_ber_bits": int(np.sum(per_bit_cnn == 0.0)),
            "always_zero_outputs": int(np.sum(predicted_frequency == 0.0)),
            "always_one_outputs": int(np.sum(predicted_frequency == 1.0)),
            "overall_target_one_frequency": float(val_y.mean()),
            "overall_predicted_one_frequency": float(val_predictions.mean()),
            "worst_10": worst_bits,
            "best_10": best_bits,
        },
        "probability_diagnostics": probability_summary,
    }
    config = {
        "experiment": "run3b_clean_training",
        "train_selection": "first 500 sorted PNG filenames from data/curated/train",
        "validation_selection": "first 100 sorted PNG filenames from data/curated/val",
        "training_payload_seed": TRAIN_PAYLOAD_SEED,
        "validation_payload_seed": VAL_PAYLOAD_SEED,
        "validation_payloads_reused_from_run3a": True,
        "coefficient_seed": COEFFICIENT_SEED,
        "payload_bits": PAYLOAD_BITS,
        "delta": DELTA,
        "wavelet": "haar",
        "dwt_level": 2,
        "subbands": ["LH2", "HL2"],
        "attacks": [],
        "features": ["c/delta", "sin(pi*c/delta)", "cos(pi*c/delta)", "subband_id"],
        "architecture": ["Conv1D(16,1,ReLU)", "Conv1D(16,1,ReLU)", "Conv1D(1,1,sigmoid)"],
        "parameters": int(model.count_params()),
        "fresh_model_initialization": True,
        "model_seed": MODEL_SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_selection": "lowest validation BER; validation BCE breaks BER ties",
        "threshold": THRESHOLD,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "best_seed_aware_clean.keras")
    write_csv(
        OUTPUT_DIR / "selected_training_images.csv",
        [
            {"selection_order": index, "image_id": path.stem, "image_filename": path.name}
            for index, path in enumerate(train_paths, start=1)
        ],
    )
    write_csv(
        OUTPUT_DIR / "selected_validation_images.csv",
        [
            {"selection_order": index, "image_id": path.stem, "image_filename": path.name}
            for index, path in enumerate(val_paths, start=1)
        ],
    )
    write_csv(OUTPUT_DIR / "training_payloads.csv", train_rows)
    write_csv(OUTPUT_DIR / "validation_payloads.csv", val_rows)
    write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "per_sample_validation_results.csv", val_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(
        OUTPUT_DIR / "probability_diagnostics.csv",
        [
            {"group": key, **value}
            for key, value in probability_summary.items()
            if isinstance(value, dict)
        ],
    )
    write_csv(OUTPUT_DIR / "comparison_run2_run3a_run3b.csv", comparison_rows)
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved Run 3B artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
