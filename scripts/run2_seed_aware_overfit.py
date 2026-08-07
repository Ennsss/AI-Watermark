"""Run 2: overfit a minimal seed-aware shared Conv1D decoder."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from watermark.embedding import qim_extract_bit

from diagnose_run1_signal_localization import (
    COEFFICIENT_SEED,
    DELTA,
    PAYLOAD_BITS,
    PAYLOAD_SEED,
    SELECTED_IMAGES,
    build_exact_samples,
    location_rows,
)


OUTPUT_DIR = ROOT / "experiments/run2_seed_aware_overfit"
RUN1_DIR = ROOT / "experiments/run1_clean_fixed_seed_overfit"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
MODEL_SEED = 42
STABLE_ZERO_EPOCHS = 5


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probability_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "fraction_0.45_to_0.55": float(np.mean((values >= 0.45) & (values <= 0.55))),
    }


def build_representation(
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    locations = location_rows(raw_inputs.shape[1:3])
    features = np.empty((len(raw_inputs), PAYLOAD_BITS, 4), dtype=np.float32)
    rows: list[dict] = []
    classical_bits = np.empty_like(targets)
    for sample_index, (coefficient_map, target, meta) in enumerate(
        zip(raw_inputs, targets, metadata)
    ):
        for location in locations:
            bit = location["bit_index"]
            channel = location["channel"]
            row = location["row"]
            column = location["column"]
            coefficient = float(coefficient_map[row, column, channel])
            scaled = coefficient / DELTA
            sin_phase = float(np.sin(np.pi * scaled))
            cos_phase = float(np.cos(np.pi * scaled))
            subband_id = float(channel)
            features[sample_index, bit] = [scaled, sin_phase, cos_phase, subband_id]
            classical_bits[sample_index, bit] = qim_extract_bit(coefficient, DELTA)
            rows.append(
                {
                    **meta,
                    "bit_index": bit,
                    "target_bit": int(target[bit]),
                    "subband": location["subband"],
                    "subband_id": int(channel),
                    "row": row,
                    "column": column,
                    "raw_coefficient": coefficient,
                    "coefficient_over_delta": scaled,
                    "sin_pi_c_over_delta": sin_phase,
                    "cos_pi_c_over_delta": cos_phase,
                    "classical_extracted_bit_verification": int(classical_bits[sample_index, bit]),
                }
            )
    verification = [
        {
            "sample_id": sample_index,
            "source_image": metadata[sample_index]["source_image"],
            "payload_index": metadata[sample_index]["payload_index"],
            "classical_ber": float(np.mean(classical_bits[sample_index] != targets[sample_index])),
        }
        for sample_index in range(len(targets))
    ]
    return features, rows, verification


def build_model(input_features: int):
    from tensorflow import keras

    inputs = keras.layers.Input(shape=(PAYLOAD_BITS, input_features), name="seed_aware_features")
    x = keras.layers.Conv1D(16, 1, activation="relu", name="shared_conv1")(inputs)
    x = keras.layers.Conv1D(16, 1, activation="relu", name="shared_conv2")(x)
    x = keras.layers.Conv1D(1, 1, activation="sigmoid", name="shared_probability")(x)
    outputs = keras.layers.Reshape((PAYLOAD_BITS,), name="payload_probabilities")(x)
    model = keras.Model(inputs, outputs, name="run2_seed_aware_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
    )
    return model


def main() -> int:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Run 2.") from exc

    raw_inputs, targets, metadata = build_exact_samples()
    features, representation_rows, classical_rows = build_representation(raw_inputs, targets, metadata)
    classical_bers = np.asarray([row["classical_ber"] for row in classical_rows])
    locations = location_rows(raw_inputs.shape[1:3])
    payload_hash = hashlib.sha256(targets.tobytes()).hexdigest()
    verification = {
        "input_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "samples": len(targets),
        "payload_bits": PAYLOAD_BITS,
        "lh2_coefficient_count": sum(row["subband"] == "LH2" for row in locations),
        "hl2_coefficient_count": sum(row["subband"] == "HL2" for row in locations),
        "coefficient_seed": COEFFICIENT_SEED,
        "payload_rng_seed": PAYLOAD_SEED,
        "unique_payloads": len({row.tobytes() for row in targets}),
        "payload_sha256": payload_hash,
        "mean_classical_ber": float(classical_bers.mean()),
        "maximum_classical_ber": float(classical_bers.max()),
        "perfect_classical_recoveries": int(np.sum(classical_bers == 0.0)),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "feature_minimums": features.min(axis=(0, 1)).astype(float).tolist(),
        "feature_maximums": features.max(axis=(0, 1)).astype(float).tolist(),
        "bit_indices_in_order": bool([row["bit_index"] for row in locations] == list(range(128))),
    }
    print("Representation verification")
    print(json.dumps(verification, indent=2))
    if verification["perfect_classical_recoveries"] != 32:
        raise SystemExit("Run 2 preflight failed: classical recovery is not 32/32.")
    if verification["nan_count"] or verification["inf_count"]:
        raise SystemExit("Run 2 preflight failed: non-finite features.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "representation_samples.csv", representation_rows)
    write_csv(OUTPUT_DIR / "classical_preflight.csv", classical_rows)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model(features.shape[-1])
    parameter_count = int(model.count_params())
    history_rows: list[dict] = []

    class BerHistory(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.consecutive_zero = 0

        def on_epoch_end(self, epoch, logs=None):
            probabilities = self.model.predict(features, batch_size=BATCH_SIZE, verbose=0)
            predictions = probabilities >= 0.5
            sample_bers = np.mean(predictions != targets, axis=1)
            mean_ber = float(sample_bers.mean())
            perfect = int(np.sum(sample_bers == 0.0))
            loss = float((logs or {}).get("loss", np.nan))
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "mean_training_ber": mean_ber,
                    "perfect_payloads": perfect,
                }
            )
            if epoch == 0 or (epoch + 1) % 10 == 0 or mean_ber == 0.0:
                print(
                    f"epoch={epoch + 1} loss={loss:.8f} "
                    f"ber={mean_ber:.8f} perfect={perfect}/32"
                )
            self.consecutive_zero = self.consecutive_zero + 1 if mean_ber == 0.0 else 0
            if self.consecutive_zero >= STABLE_ZERO_EPOCHS:
                self.model.stop_training = True

    model.fit(
        features,
        targets.astype(np.float32),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=[BerHistory()],
        verbose=0,
    )
    probabilities = np.asarray(model.predict(features, batch_size=BATCH_SIZE, verbose=0))
    predictions = (probabilities >= 0.5).astype(np.uint8)
    errors = predictions != targets
    sample_bers = errors.mean(axis=1)
    per_bit_ber = errors.mean(axis=0)
    target_one_frequency = targets.mean(axis=0)
    predicted_one_frequency = predictions.mean(axis=0)
    status = "PASS" if float(sample_bers.mean()) <= 0.001 else "FAIL"
    summary = {
        "status": status,
        "mean_training_ber": float(sample_bers.mean()),
        "median_training_ber": float(np.median(sample_bers)),
        "maximum_sample_ber": float(sample_bers.max()),
        "minimum_sample_ber": float(sample_bers.min()),
        "perfect_payloads": int(np.sum(sample_bers == 0.0)),
        "samples": len(sample_bers),
        "output_bits_with_ber_zero": int(np.sum(per_bit_ber == 0.0)),
        "overall_target_one_frequency": float(targets.mean()),
        "overall_predicted_one_frequency": float(predictions.mean()),
        "probabilities": probability_summary(probabilities),
        "epochs_trained": len(history_rows),
        "final_bce_loss": float(history_rows[-1]["loss"]),
        "parameter_count": parameter_count,
    }
    config = {
        "experiment": "run2_seed_aware_overfit",
        "hypothesis": "gathering seed-selected coefficients in payload order makes the clean task learnable",
        "source_images": SELECTED_IMAGES,
        "payload_sha256": payload_hash,
        "payload_rng_seed": PAYLOAD_SEED,
        "coefficient_seed": COEFFICIENT_SEED,
        "payload_bits": PAYLOAD_BITS,
        "delta": DELTA,
        "wavelet": "haar",
        "dwt_level": 2,
        "subbands": ["LH2", "HL2"],
        "attacks": [],
        "input_shape": [128, 4],
        "features_in_order": [
            "coefficient / delta",
            "sin(pi * coefficient / delta)",
            "cos(pi * coefficient / delta)",
            "subband identifier (LH2=0, HL2=1)",
        ],
        "periodic_feature_rationale": (
            "Production bit-0 states lie at c/delta=k and bit-1 states at k+1/2; "
            "sin(pi*c/delta) and cos(pi*c/delta) continuously encode this 2*delta-periodic phase."
        ),
        "model": [
            "Conv1D(16, kernel_size=1, ReLU)",
            "Conv1D(16, kernel_size=1, ReLU)",
            "Conv1D(1, kernel_size=1, sigmoid)",
            "Reshape(128)",
        ],
        "parameter_count": parameter_count,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "epochs_actually_trained": len(history_rows),
        "validation": None,
        "pooling": False,
        "dropout": False,
        "stable_zero_epochs": STABLE_ZERO_EPOCHS,
    }
    per_bit_rows = [
        {
            "bit_index": bit,
            "per_bit_ber": float(per_bit_ber[bit]),
            "target_one_frequency": float(target_one_frequency[bit]),
            "predicted_one_frequency": float(predicted_one_frequency[bit]),
        }
        for bit in range(PAYLOAD_BITS)
    ]

    model.save(OUTPUT_DIR / "seed_aware_overfit.keras")
    with (RUN1_DIR / "summary_metrics.json").open(encoding="utf-8") as handle:
        run1_summary = json.load(handle)
    run1_model = keras.models.load_model(RUN1_DIR / "baseline_cnn_overfit.keras")
    comparison = [
        {
            "run": "Run 1 full-map CNN",
            "input_representation": "128x128x2 full LH2/HL2 maps",
            "parameter_count": int(run1_model.count_params()),
            "final_bce": run1_summary["final_bce_loss"],
            "final_mean_ber": run1_summary["mean_training_ber"],
            "perfect_payloads": f"{run1_summary['perfect_recoveries']}/32",
            "epochs": run1_summary["epochs_trained"],
            "pooling_used": True,
            "seed_locations_explicit": False,
        },
        {
            "run": "Run 2 seed-aware CNN",
            "input_representation": "128x4 selected-coefficient features in payload order",
            "parameter_count": parameter_count,
            "final_bce": summary["final_bce_loss"],
            "final_mean_ber": summary["mean_training_ber"],
            "perfect_payloads": f"{summary['perfect_payloads']}/32",
            "epochs": summary["epochs_trained"],
            "pooling_used": False,
            "seed_locations_explicit": True,
        },
    ]
    write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", per_bit_rows)
    write_csv(OUTPUT_DIR / "comparison_to_run1.csv", comparison)
    with (OUTPUT_DIR / "representation_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2)
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Final results")
    print(json.dumps(summary, indent=2))
    print(f"Saved Run 2 artifacts to {OUTPUT_DIR}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
