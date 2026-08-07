"""Stage 2B: full-factorial JPEG/re-encode exposure for every training pair."""

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
from run3a_clean_transfer import construct_features, distribution
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import (
    BATCH_SIZE,
    COEFFICIENT_SEED,
    CONDITIONS,
    DELTA,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_SEED,
    PATIENCE,
    PAYLOAD_BITS,
    PAYLOADS_PER_IMAGE,
    RUN3B_DIR,
    THRESHOLD,
    TRAIN_DIR,
    TRAIN_IMAGE_COUNT,
    TRAIN_PAYLOAD_SEED,
    VAL_DIR,
    VAL_IMAGE_COUNT,
    VAL_PAYLOAD_SEED,
    apply_condition,
    condition_metrics,
    generate_validation,
    prediction_metrics,
    write_csv,
)


STAGE2_DIR = ROOT / "experiments/stage2_jpeg_reencode_training"
OUTPUT_DIR = ROOT / "experiments/stage2b_full_factorial_jpeg_reencode"
SHUFFLE_SEED = 20260812


def generate_full_factorial_training(
    paths: list[Path], expected_fingerprints: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(TRAIN_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    condition_ids: list[int] = []
    metadata: list[dict] = []
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[full-factorial train] {image_index}/{len(paths)} {path.name}")
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected_fingerprints[base_pair]:
                raise RuntimeError("Training payload mismatch with Run 3B/Stage 2.")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                features.append(construct_features(coefficient_map, locations))
                targets.append(bits)
                condition_ids.append(condition_id)
                metadata.append(
                    {
                        "unshuffled_row": len(metadata),
                        "base_pair_index": base_pair,
                        "image_filename": path.name,
                        "payload_index": payload_index,
                        "payload_fingerprint": fingerprint,
                        "condition": condition,
                    }
                )
            base_pair += 1
    feature_array = np.stack(features).astype(np.float32)
    target_array = np.stack(targets).astype(np.uint8)
    condition_array = np.asarray(condition_ids, dtype=np.int8)
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(len(feature_array))
    for shuffled_row, original_row in enumerate(permutation):
        metadata[original_row]["shuffled_row"] = shuffled_row
    return (
        feature_array[permutation],
        target_array[permutation],
        condition_array[permutation],
        metadata,
    )


def main() -> int:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Stage 2B.") from exc

    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    run3b_train_payloads = list(
        csv.DictReader((RUN3B_DIR / "training_payloads.csv").open(encoding="utf-8"))
    )
    run3b_val_payloads = list(
        csv.DictReader((RUN3B_DIR / "validation_payloads.csv").open(encoding="utf-8"))
    )
    train_x, train_y, train_condition_ids, train_metadata = generate_full_factorial_training(
        train_paths, [row["payload_fingerprint"] for row in run3b_train_payloads]
    )
    val_x, val_y, val_classical, val_condition_ids, val_rows = generate_validation(
        val_paths, [row["payload_fingerprint"] for row in run3b_val_payloads]
    )
    if train_x.shape != (7000, 128, 4) or val_x.shape != (1400, 128, 4):
        raise RuntimeError(f"Unexpected Stage 2B shapes: {train_x.shape}, {val_x.shape}")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("Stage 2B feature cache contains NaN or Inf.")

    stage2_rows = list(
        csv.DictReader((STAGE2_DIR / "per_sample_validation_results.csv").open(encoding="utf-8"))
    )
    if len(stage2_rows) != len(val_rows):
        raise RuntimeError("Stage 2 validation row count mismatch.")
    for current, previous in zip(val_rows, stage2_rows):
        identity = ("image_filename", "payload_index", "payload_fingerprint", "condition")
        if any(str(current[key]) != str(previous[key]) for key in identity):
            raise RuntimeError("Stage 2B validation grid identity differs from Stage 2.")
        if not np.isclose(float(current["classical_ber"]), float(previous["classical_ber"])):
            raise RuntimeError("Stage 2B validation classical BER differs from Stage 2.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model(input_features=4)
    if model.count_params() != 369:
        raise RuntimeError("Stage 2B architecture changed unexpectedly.")
    history_rows: list[dict] = []

    class MacroCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_macro = float("inf")
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.best_weights = None
            self.stale_epochs = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_prob = self.model.predict(train_x, batch_size=BATCH_SIZE, verbose=0)
            val_prob = self.model.predict(val_x, batch_size=BATCH_SIZE, verbose=0)
            train_overall = prediction_metrics(train_prob, train_y)
            train_conditions = condition_metrics(train_prob, train_y, train_condition_ids)
            val_conditions = condition_metrics(val_prob, val_y, val_condition_ids)
            macro = float(np.mean([val_conditions[name]["ber"] for name in CONDITIONS]))
            val_loss = float(logs.get("val_loss", np.nan))
            strict = macro < self.best_macro - 1e-12
            tie_loss = abs(macro - self.best_macro) <= 1e-12 and val_loss < self.best_loss
            self.stale_epochs = 0 if strict else self.stale_epochs + 1
            if strict or tie_loss:
                self.best_macro = macro
                self.best_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
            row = {
                "epoch": epoch + 1,
                "train_bce": float(logs.get("loss", np.nan)),
                "train_ber": train_overall["ber"],
                "train_lh2_ber": train_overall["lh2_ber"],
                "train_hl2_ber": train_overall["hl2_ber"],
                "validation_bce": val_loss,
                "validation_macro_ber": macro,
                "is_best_checkpoint": self.best_epoch == epoch + 1,
            }
            for condition in CONDITIONS:
                for prefix, metrics in [
                    ("train", train_conditions[condition]),
                    ("validation", val_conditions[condition]),
                ]:
                    row[f"{prefix}_{condition}_ber"] = metrics["ber"]
                    row[f"{prefix}_{condition}_lh2_ber"] = metrics["lh2_ber"]
                    row[f"{prefix}_{condition}_hl2_ber"] = metrics["hl2_ber"]
                    row[f"{prefix}_{condition}_perfect_payloads"] = metrics["perfect_payloads"]
            history_rows.append(row)
            print(
                f"epoch={epoch + 1} train={train_overall['ber']:.6f} macro={macro:.6f} "
                f"clean={val_conditions['clean']['ber']:.6f} jpeg70={val_conditions['jpeg70']['ber']:.6f} "
                f"jpeg50={val_conditions['jpeg50']['ber']:.6f} re3={val_conditions['reencode3']['ber']:.6f}"
            )
            if self.stale_epochs >= PATIENCE:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise RuntimeError("No Stage 2B checkpoint captured.")
            self.model.set_weights(self.best_weights)

    checkpoint = MacroCheckpoint()
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

    probabilities = np.asarray(model.predict(val_x, batch_size=BATCH_SIZE, verbose=0))
    predictions = (probabilities >= THRESHOLD).astype(np.uint8)
    cnn_errors = predictions != val_y
    classical_errors = val_classical != val_y
    cnn_sample_bers = cnn_errors.mean(axis=1)
    classical_sample_bers = classical_errors.mean(axis=1)
    confidences = np.abs(probabilities - 0.5) * 2.0
    for index, row in enumerate(val_rows):
        row.update(
            {
                "stage2_cnn_ber": float(stage2_rows[index]["cnn_ber"]),
                "stage2b_cnn_ber": float(cnn_sample_bers[index]),
                "stage2b_perfect": bool(cnn_sample_bers[index] == 0.0),
                "stage2b_mean_confidence": float(confidences[index].mean()),
            }
        )

    stage2_condition_rows = {
        row["condition"]: row
        for row in csv.DictReader((STAGE2_DIR / "per_condition_summary.csv").open(encoding="utf-8"))
    }
    condition_rows: list[dict] = []
    bit_rows: list[dict] = []
    probability_rows: list[dict] = []
    for condition_id, condition in enumerate(CONDITIONS):
        mask = val_condition_ids == condition_id
        errors = cnn_errors[mask]
        classical_condition_errors = classical_errors[mask]
        condition_bers = cnn_sample_bers[mask]
        classical_bers = classical_sample_bers[mask]
        stage2_bers = np.asarray([float(row["cnn_ber"]) for row in np.asarray(stage2_rows, dtype=object)[mask]])
        stage2_mean = float(stage2_condition_rows[condition]["cnn_mean_ber"])
        stage2b_mean = float(errors.mean())
        absolute_improvement = stage2_mean - stage2b_mean
        relative_improvement = absolute_improvement / stage2_mean if stage2_mean > 0 else None
        condition_rows.append(
            {
                "condition": condition,
                "classical_ber": float(classical_condition_errors.mean()),
                "stage2_cnn_ber": stage2_mean,
                "stage2b_cnn_ber": stage2b_mean,
                "absolute_improvement_over_stage2": absolute_improvement,
                "relative_improvement_over_stage2": relative_improvement,
                "stage2b_minus_classical": float(stage2b_mean - classical_condition_errors.mean()),
                "stage2b_perfect_rate": float(np.mean(condition_bers == 0.0)),
                "stage2b_lh2_ber": float(errors[:, :64].mean()),
                "stage2b_hl2_ber": float(errors[:, 64:].mean()),
                "stage2b_better_than_classical": int(np.sum(condition_bers < classical_bers)),
                "stage2b_equal_classical": int(np.sum(condition_bers == classical_bers)),
                "stage2b_worse_than_classical": int(np.sum(condition_bers > classical_bers)),
                "stage2b_better_than_stage2": int(np.sum(condition_bers < stage2_bers)),
                "stage2b_equal_stage2": int(np.sum(condition_bers == stage2_bers)),
                "stage2b_worse_than_stage2": int(np.sum(condition_bers > stage2_bers)),
            }
        )
        per_bit = errors.mean(axis=0)
        target_frequency = val_y[mask].mean(axis=0)
        predicted_frequency = predictions[mask].mean(axis=0)
        worst = set(np.argsort(-per_bit, kind="stable")[:10].tolist())
        for bit in range(PAYLOAD_BITS):
            bit_rows.append(
                {
                    "condition": condition,
                    "bit_index": bit,
                    "subband": "LH2" if bit < 64 else "HL2",
                    "cnn_ber": float(per_bit[bit]),
                    "classical_ber": float(classical_condition_errors.mean(axis=0)[bit]),
                    "target_one_frequency": float(target_frequency[bit]),
                    "predicted_one_frequency": float(predicted_frequency[bit]),
                    "zero_ber": bool(per_bit[bit] == 0.0),
                    "is_worst_10": bit in worst,
                }
            )
        condition_prob = probabilities[mask]
        condition_targets = val_y[mask]
        condition_conf = confidences[mask]
        incorrect = errors
        incorrect_prob = condition_prob[incorrect]
        probability_rows.append(
            {
                "condition": condition,
                **{f"probability_{key}": value for key, value in distribution(condition_prob.ravel()).items()},
                "fraction_0.45_to_0.55": float(
                    np.mean((condition_prob >= 0.45) & (condition_prob <= 0.55))
                ),
                "correct_confidence": float(condition_conf[~incorrect].mean()),
                "incorrect_confidence": float(condition_conf[incorrect].mean()) if incorrect.any() else None,
                **{
                    f"incorrect_probability_{key}": value
                    for key, value in distribution(incorrect_prob).items()
                },
                "false_zero_count": int(np.sum(incorrect & (condition_targets == 1))),
                "false_one_count": int(np.sum(incorrect & (condition_targets == 0))),
                "zero_ber_positions": int(np.sum(per_bit == 0.0)),
                "always_zero_outputs": int(np.sum(predicted_frequency == 0.0)),
                "always_one_outputs": int(np.sum(predicted_frequency == 1.0)),
            }
        )

    classical_macro_all = float(np.mean([row["classical_ber"] for row in condition_rows]))
    stage2_macro_all = float(np.mean([row["stage2_cnn_ber"] for row in condition_rows]))
    stage2b_macro_all = float(np.mean([row["stage2b_cnn_ber"] for row in condition_rows]))
    classical_attack_macro = float(np.mean([row["classical_ber"] for row in condition_rows[1:]]))
    stage2_attack_macro = float(np.mean([row["stage2_cnn_ber"] for row in condition_rows[1:]]))
    stage2b_attack_macro = float(np.mean([row["stage2b_cnn_ber"] for row in condition_rows[1:]]))
    improvement_vs_stage2 = stage2_attack_macro - stage2b_attack_macro
    improvement_vs_classical = classical_attack_macro - stage2b_attack_macro
    clean_ber = condition_rows[0]["stage2b_cnn_ber"]
    clean_retention = "Excellent" if clean_ber < 0.001 else "Acceptable" if clean_ber < 0.005 else "Concern"
    moderate_clear = any(
        row["absolute_improvement_over_stage2"] > 0.01
        for row in condition_rows
        if row["condition"] in {"jpeg70", "reencode1", "reencode2", "reencode3"}
    )
    no_new_subband_failure = all(
        max(row["stage2b_lh2_ber"], row["stage2b_hl2_ber"])
        <= max(
            float(stage2_condition_rows[row["condition"]]["cnn_lh2_ber"]),
            float(stage2_condition_rows[row["condition"]]["cnn_hl2_ber"]),
        )
        + 0.02
        for row in condition_rows
    )
    material_macro = improvement_vs_stage2 > 0.01
    if clean_ber < 0.001 and material_macro and moderate_clear and no_new_subband_failure:
        classification = "STRONG PASS"
    elif clean_ber < 0.005 and material_macro and no_new_subband_failure:
        classification = "PASS"
    elif improvement_vs_stage2 > 0 or clean_ber < 0.005:
        classification = "PARTIAL"
    else:
        classification = "FAIL"

    cache_metadata = {
        "cache_type": "in-memory precomputed feature tensors",
        "training_feature_shape": list(train_x.shape),
        "training_target_shape": list(train_y.shape),
        "training_condition_shape": list(train_condition_ids.shape),
        "validation_feature_shape": list(val_x.shape),
        "validation_target_shape": list(val_y.shape),
        "training_nan_count": int(np.isnan(train_x).sum()),
        "training_inf_count": int(np.isinf(train_x).sum()),
        "validation_nan_count": int(np.isnan(val_x).sum()),
        "validation_inf_count": int(np.isinf(val_x).sum()),
        "shuffle_seed": SHUFFLE_SEED,
    }
    summary = {
        "classification": classification,
        "best_epoch": checkpoint.best_epoch,
        "epochs_trained": len(history_rows),
        "best_macro_validation_ber": checkpoint.best_macro,
        "best_validation_bce_tiebreak": checkpoint.best_loss,
        "clean_retention": clean_retention,
        "macro_all_seven": {
            "classical": classical_macro_all,
            "stage2": stage2_macro_all,
            "stage2b": stage2b_macro_all,
        },
        "macro_attacks_only": {
            "classical": classical_attack_macro,
            "stage2": stage2_attack_macro,
            "stage2b": stage2b_attack_macro,
            "stage2b_absolute_improvement_vs_stage2": improvement_vs_stage2,
            "stage2b_relative_improvement_vs_stage2": improvement_vs_stage2 / stage2_attack_macro,
            "stage2b_absolute_improvement_vs_classical": improvement_vs_classical,
            "stage2b_relative_improvement_vs_classical": improvement_vs_classical / classical_attack_macro,
        },
        "checks": {
            "material_macro_improvement_threshold_gt_0.01": material_macro,
            "moderate_or_reencode_clear_improvement_gt_0.01": moderate_clear,
            "no_new_subband_failure": no_new_subband_failure,
        },
    }
    config = {
        "experiment": "stage2b_full_factorial_jpeg_reencode",
        "hypothesis": "full-factorial attack exposure improves robustness without changing the model",
        "base_training_pairs": 1000,
        "conditions_per_pair": 7,
        "training_examples": 7000,
        "condition_counts": {condition: 1000 for condition in CONDITIONS},
        "training_payload_seed": TRAIN_PAYLOAD_SEED,
        "validation_payload_seed": VAL_PAYLOAD_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "model_seed": MODEL_SEED,
        "model_parameters": 369,
        "features": ["c/delta", "sin(pi*c/delta)", "cos(pi*c/delta)", "subband_id"],
        "coefficient_seed": COEFFICIENT_SEED,
        "delta": DELTA,
        "wavelet": "haar",
        "dwt_level": 2,
        "subbands": ["LH2", "HL2"],
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint": "lowest seven-condition macro validation BER; validation BCE tie-break",
        "attack_parameters_unchanged_from_stage2": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "best_seed_aware_full_factorial.keras")
    write_csv(
        OUTPUT_DIR / "condition_counts.csv",
        [
            {"split": "training", "condition": condition, "count": 1000, "fraction": 1 / 7}
            for condition in CONDITIONS
        ]
        + [
            {"split": "validation", "condition": condition, "count": 200, "fraction": 1 / 7}
            for condition in CONDITIONS
        ],
    )
    write_csv(OUTPUT_DIR / "training_example_metadata.csv", train_metadata)
    write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "per_condition_summary.csv", condition_rows)
    write_csv(OUTPUT_DIR / "per_sample_validation_results.csv", val_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(OUTPUT_DIR / "probability_diagnostics.csv", probability_rows)
    write_csv(OUTPUT_DIR / "comparison_stage2_stage2b.csv", condition_rows)
    with (OUTPUT_DIR / "cache_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(cache_metadata, handle, indent=2)
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved Stage 2B artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
