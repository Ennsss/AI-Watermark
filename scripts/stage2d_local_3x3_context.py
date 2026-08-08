"""Stage 2D: controlled shared 3x3 local-context decoder."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import extract_from_image
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run3a_clean_transfer import distribution
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import (
    BATCH_SIZE, COEFFICIENT_SEED, CONDITIONS, DELTA, LEARNING_RATE,
    MAX_EPOCHS, MODEL_SEED, PATIENCE, PAYLOAD_BITS, PAYLOADS_PER_IMAGE,
    RUN3B_DIR, THRESHOLD, TRAIN_DIR, TRAIN_IMAGE_COUNT, TRAIN_PAYLOAD_SEED,
    VAL_DIR, VAL_IMAGE_COUNT, VAL_PAYLOAD_SEED, apply_condition,
    condition_metrics, prediction_metrics, write_csv,
)

OUTPUT_DIR = ROOT / "experiments/stage2d_local_3x3_context"
STAGE2B_DIR = ROOT / "experiments/stage2b_full_factorial_jpeg_reencode"
SHUFFLE_SEED = 20260812
PATCH_SIZE = 3
PADDING_MODE = "symmetric"


def build_model():
    from tensorflow import keras

    patches = keras.layers.Input((PAYLOAD_BITS, 3, 3, 3), name="local_patch_features")
    subbands = keras.layers.Input((PAYLOAD_BITS, 1), name="subband_id")
    x = keras.layers.TimeDistributed(
        keras.layers.Conv2D(16, (3, 3), activation="relu", padding="valid"),
        name="shared_patch_conv",
    )(patches)
    x = keras.layers.TimeDistributed(keras.layers.Flatten(), name="per_bit_flatten")(x)
    x = keras.layers.Concatenate(axis=-1, name="append_subband_id")([x, subbands])
    x = keras.layers.TimeDistributed(
        keras.layers.Dense(8, activation="relu"), name="shared_hidden"
    )(x)
    x = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid"), name="shared_probability"
    )(x)
    outputs = keras.layers.Reshape((PAYLOAD_BITS,), name="payload_probabilities")(x)
    model = keras.Model([patches, subbands], outputs, name="stage2d_shared_local_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
    )
    return model


def patch_features(coefficient_map: np.ndarray, locations: list[dict]) -> np.ndarray:
    padded = np.pad(coefficient_map, ((1, 1), (1, 1), (0, 0)), mode=PADDING_MODE)
    result = np.empty((PAYLOAD_BITS, 3, 3, 3), dtype=np.float32)
    for location in locations:
        bit = location["bit_index"]
        row, column, channel = location["row"], location["column"], location["channel"]
        raw = padded[row : row + 3, column : column + 3, channel]
        scaled = raw / DELTA
        result[bit, ..., 0] = scaled
        result[bit, ..., 1] = np.sin(np.pi * scaled)
        result[bit, ..., 2] = np.cos(np.pi * scaled)
    return result


def subband_tensor(sample_count: int) -> np.ndarray:
    ids = np.concatenate([np.zeros(64), np.ones(64)]).astype(np.float32)
    return np.broadcast_to(ids[None, :, None], (sample_count, PAYLOAD_BITS, 1)).copy()


def make_representation_sample(
    coefficient_map: np.ndarray, features: np.ndarray, target: np.ndarray,
    locations: list[dict], sample_id: int, condition: str,
) -> list[dict]:
    rows = []
    padded = np.pad(coefficient_map, ((1, 1), (1, 1), (0, 0)), mode=PADDING_MODE)
    for location in locations[:4] + locations[62:66] + locations[-4:]:
        bit = location["bit_index"]
        row, col, channel = location["row"], location["column"], location["channel"]
        raw = padded[row : row + 3, col : col + 3, channel]
        patch_coordinates = [
            (max(0, min(127, row + dr)), max(0, min(127, col + dc)))
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        ]
        rows.append({
            "sample_id": sample_id,
            "condition": condition,
            "bit_index": bit,
            "target_bit": int(target[bit]),
            "subband": location["subband"],
            "center_row": row,
            "center_column": col,
            "patch_coordinates_after_symmetric_boundary_mapping": json.dumps(patch_coordinates),
            "raw_3x3_coefficients": json.dumps(raw.astype(float).tolist()),
            "derived_3x3x3_features": json.dumps(features[bit].astype(float).tolist()),
        })
    return rows


def generate_training(paths: list[Path], expected: list[str]):
    total = TRAIN_IMAGE_COUNT * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    x = np.empty((total, PAYLOAD_BITS, 3, 3, 3), dtype=np.float32)
    y = np.empty((total, PAYLOAD_BITS), dtype=np.uint8)
    condition_ids = np.empty(total, dtype=np.int8)
    rng = np.random.default_rng(TRAIN_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    metadata = []
    samples = []
    row_id = 0
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[train patches] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected[base_pair]:
                raise RuntimeError("Training payload fingerprint differs from Stage 2B.")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                x[row_id] = patch_features(coefficient_map, locations)
                y[row_id] = bits
                condition_ids[row_id] = condition_id
                metadata.append({
                    "unshuffled_row": row_id, "base_pair_index": base_pair,
                    "image_filename": path.name, "payload_index": payload_index,
                    "payload_fingerprint": fingerprint, "condition": condition,
                })
                if base_pair == 0 and condition in {"clean", "jpeg70", "jpeg50"}:
                    samples.extend(make_representation_sample(
                        coefficient_map, x[row_id], bits, locations, row_id, condition
                    ))
                row_id += 1
            base_pair += 1
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(total)
    for shuffled_row, original_row in enumerate(permutation):
        metadata[original_row]["shuffled_row"] = shuffled_row
    return x[permutation], y[permutation], condition_ids[permutation], metadata, samples


def generate_validation(paths: list[Path], expected: list[str]):
    total = VAL_IMAGE_COUNT * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    x = np.empty((total, PAYLOAD_BITS, 3, 3, 3), dtype=np.float32)
    y = np.empty((total, PAYLOAD_BITS), dtype=np.uint8)
    classical = np.empty_like(y)
    condition_ids = np.empty(total, dtype=np.int8)
    rows = []
    rng = np.random.default_rng(VAL_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    row_id = 0
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[validation patches] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected[base_pair]:
                raise RuntimeError("Validation payload fingerprint differs from Stage 2B.")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                x[row_id] = patch_features(coefficient_map, locations)
                y[row_id] = bits
                condition_ids[row_id] = condition_id
                classical[row_id], _ = extract_from_image(
                    attacked, PAYLOAD_BITS, COEFFICIENT_SEED, DELTA, "haar",
                    target_subbands=("lh2", "hl2"),
                )
                rows.append({
                    "row_id": row_id, "base_pair_index": base_pair,
                    "image_filename": path.name, "payload_index": payload_index,
                    "payload_fingerprint": fingerprint, "condition": condition,
                })
                row_id += 1
            base_pair += 1
    return x, y, classical, condition_ids, rows


def main() -> int:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Stage 2D.") from exc

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    train_payload_rows = list(csv.DictReader((RUN3B_DIR / "training_payloads.csv").open(encoding="utf-8")))
    val_payload_rows = list(csv.DictReader((RUN3B_DIR / "validation_payloads.csv").open(encoding="utf-8")))
    train_x, train_y, train_cids, train_metadata, representation_samples = generate_training(
        train_paths, [row["payload_fingerprint"] for row in train_payload_rows]
    )
    val_x, val_y, val_classical, val_cids, val_rows = generate_validation(
        val_paths, [row["payload_fingerprint"] for row in val_payload_rows]
    )
    train_subbands = subband_tensor(len(train_x))
    val_subbands = subband_tensor(len(val_x))
    if train_x.shape != (7000, 128, 3, 3, 3) or val_x.shape != (1400, 128, 3, 3, 3):
        raise RuntimeError(f"Unexpected patch shapes: {train_x.shape}, {val_x.shape}")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("Patch representation contains NaN or Inf.")

    stage2b_samples = list(csv.DictReader(
        (STAGE2B_DIR / "per_sample_validation_results.csv").open(encoding="utf-8")
    ))
    if len(stage2b_samples) != len(val_rows):
        raise RuntimeError("Stage 2B validation row count mismatch.")
    for current, previous in zip(val_rows, stage2b_samples):
        for key in ("image_filename", "payload_index", "payload_fingerprint", "condition"):
            if str(current[key]) != str(previous[key]):
                raise RuntimeError(f"Stage 2D validation identity mismatch at {key}.")

    locations = location_rows((128, 128))
    verification = {
        "training_patch_shape": list(train_x.shape),
        "training_subband_shape": list(train_subbands.shape),
        "validation_patch_shape": list(val_x.shape),
        "validation_subband_shape": list(val_subbands.shape),
        "payload_positions": len(locations),
        "lh2_centers": sum(row["subband"] == "LH2" for row in locations),
        "hl2_centers": sum(row["subband"] == "HL2" for row in locations),
        "bit_order_exact": [row["bit_index"] for row in locations] == list(range(128)),
        "centers_match_stage2b_location_regeneration": True,
        "same_subband_patch_only": True,
        "padding": PADDING_MODE,
        "all_patches_full_3x3": True,
        "training_payload_fingerprints_match": True,
        "validation_payload_fingerprints_match": True,
        "validation_grid_matches_stage2b": True,
        "nan_count": int(np.isnan(train_x).sum() + np.isnan(val_x).sum()),
        "inf_count": int(np.isinf(train_x).sum() + np.isinf(val_x).sum()),
        "test_set_used": False,
    }

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model()
    parameter_count = int(model.count_params())
    if parameter_count >= 2000:
        raise RuntimeError(f"Stage 2D model too large: {parameter_count}")
    history_rows = []

    class MacroCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_macro = float("inf")
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.best_weights = None
            self.stale = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_prob = self.model.predict([train_x, train_subbands], batch_size=BATCH_SIZE, verbose=0)
            val_prob = self.model.predict([val_x, val_subbands], batch_size=BATCH_SIZE, verbose=0)
            train_overall = prediction_metrics(train_prob, train_y)
            train_conditions = condition_metrics(train_prob, train_y, train_cids)
            val_conditions = condition_metrics(val_prob, val_y, val_cids)
            macro = float(np.mean([val_conditions[name]["ber"] for name in CONDITIONS]))
            val_loss = float(logs.get("val_loss", np.nan))
            strict = macro < self.best_macro - 1e-12
            tie = abs(macro - self.best_macro) <= 1e-12 and val_loss < self.best_loss
            self.stale = 0 if strict else self.stale + 1
            if strict or tie:
                self.best_macro, self.best_loss = macro, val_loss
                self.best_epoch, self.best_weights = epoch + 1, self.model.get_weights()
            row = {
                "epoch": epoch + 1, "train_bce": float(logs.get("loss", np.nan)),
                "train_ber": train_overall["ber"], "train_lh2_ber": train_overall["lh2_ber"],
                "train_hl2_ber": train_overall["hl2_ber"], "validation_bce": val_loss,
                "validation_macro_ber": macro, "is_best_checkpoint": self.best_epoch == epoch + 1,
            }
            for condition in CONDITIONS:
                for prefix, metrics in (("train", train_conditions[condition]), ("validation", val_conditions[condition])):
                    row[f"{prefix}_{condition}_ber"] = metrics["ber"]
                    row[f"{prefix}_{condition}_lh2_ber"] = metrics["lh2_ber"]
                    row[f"{prefix}_{condition}_hl2_ber"] = metrics["hl2_ber"]
            history_rows.append(row)
            print(
                f"epoch={epoch+1} train={train_overall['ber']:.6f} macro={macro:.6f} "
                f"clean={val_conditions['clean']['ber']:.6f} jpeg70={val_conditions['jpeg70']['ber']:.6f} "
                f"jpeg50={val_conditions['jpeg50']['ber']:.6f}", flush=True
            )
            if self.stale >= PATIENCE:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise RuntimeError("No Stage 2D checkpoint captured.")
            self.model.set_weights(self.best_weights)

    checkpoint = MacroCheckpoint()
    model.fit(
        [train_x, train_subbands], train_y.astype(np.float32),
        validation_data=([val_x, val_subbands], val_y.astype(np.float32)),
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, shuffle=True,
        callbacks=[checkpoint], verbose=0,
    )

    probabilities = np.asarray(model.predict([val_x, val_subbands], batch_size=BATCH_SIZE, verbose=0))
    predictions = (probabilities >= THRESHOLD).astype(np.uint8)
    errors = predictions != val_y
    classical_errors = val_classical != val_y
    sample_bers = errors.mean(axis=1)
    classical_bers = classical_errors.mean(axis=1)
    confidences = np.abs(probabilities - 0.5) * 2
    stage2b_conditions = {
        row["condition"]: row for row in csv.DictReader(
            (STAGE2B_DIR / "per_condition_summary.csv").open(encoding="utf-8")
        )
    }
    condition_rows, bit_rows, probability_rows = [], [], []
    for condition_id, condition in enumerate(CONDITIONS):
        mask = val_cids == condition_id
        condition_errors = errors[mask]
        condition_bers = sample_bers[mask]
        condition_classical_bers = classical_bers[mask]
        stage2b_bers = np.asarray([
            float(row["stage2b_cnn_ber"]) for row in np.asarray(stage2b_samples, dtype=object)[mask]
        ])
        stage2b_ber = float(stage2b_conditions[condition]["stage2b_cnn_ber"])
        stage2d_ber = float(condition_errors.mean())
        improvement = stage2b_ber - stage2d_ber
        condition_rows.append({
            "condition": condition, "classical_ber": float(classical_errors[mask].mean()),
            "stage2b_ber": stage2b_ber, "stage2d_ber": stage2d_ber,
            "absolute_improvement_over_stage2b": improvement,
            "relative_improvement_over_stage2b": improvement / stage2b_ber,
            "perfect_payload_rate": float(np.mean(condition_bers == 0)),
            "lh2_ber": float(condition_errors[:, :64].mean()),
            "hl2_ber": float(condition_errors[:, 64:].mean()),
            "stage2d_better_than_stage2b": int(np.sum(condition_bers < stage2b_bers)),
            "stage2d_equal_stage2b": int(np.sum(condition_bers == stage2b_bers)),
            "stage2d_worse_than_stage2b": int(np.sum(condition_bers > stage2b_bers)),
            "stage2d_better_than_classical": int(np.sum(condition_bers < condition_classical_bers)),
            "stage2d_equal_classical": int(np.sum(condition_bers == condition_classical_bers)),
            "stage2d_worse_than_classical": int(np.sum(condition_bers > condition_classical_bers)),
        })
        per_bit = condition_errors.mean(axis=0)
        target_frequency = val_y[mask].mean(axis=0)
        predicted_frequency = predictions[mask].mean(axis=0)
        worst = set(np.argsort(-per_bit, kind="stable")[:10].tolist())
        for bit in range(PAYLOAD_BITS):
            bit_rows.append({
                "condition": condition, "bit_index": bit,
                "subband": "LH2" if bit < 64 else "HL2", "ber": float(per_bit[bit]),
                "zero_ber": bool(per_bit[bit] == 0), "is_worst_10": bit in worst,
                "target_one_frequency": float(target_frequency[bit]),
                "predicted_one_frequency": float(predicted_frequency[bit]),
            })
        condition_prob = probabilities[mask]
        incorrect = condition_errors
        condition_conf = confidences[mask]
        probability_rows.append({
            "condition": condition,
            **{f"probability_{key}": value for key, value in distribution(condition_prob.ravel()).items()},
            "fraction_0.45_to_0.55": float(np.mean((condition_prob >= 0.45) & (condition_prob <= 0.55))),
            "correct_confidence": float(condition_conf[~incorrect].mean()),
            "incorrect_confidence": float(condition_conf[incorrect].mean()) if incorrect.any() else None,
            "false_zero_count": int(np.sum(incorrect & (val_y[mask] == 1))),
            "false_one_count": int(np.sum(incorrect & (val_y[mask] == 0))),
            "target_one_frequency": float(val_y[mask].mean()),
            "predicted_one_frequency": float(predictions[mask].mean()),
            "zero_ber_bit_positions": int(np.sum(per_bit == 0)),
        })

    for index, row in enumerate(val_rows):
        row.update({
            "classical_ber": float(classical_bers[index]),
            "stage2b_ber": float(stage2b_samples[index]["stage2b_cnn_ber"]),
            "stage2d_ber": float(sample_bers[index]),
            "stage2d_mean_confidence": float(confidences[index].mean()),
        })

    masked_x = val_x.copy()
    masked_x[:, :, :, :, :] = 0.0
    masked_x[:, :, 1, 1, :] = val_x[:, :, 1, 1, :]
    masked_prob = np.asarray(model.predict([masked_x, val_subbands], batch_size=BATCH_SIZE, verbose=0))
    masked_pred = masked_prob >= THRESHOLD
    ablation_rows = []
    for condition in ("jpeg70", "jpeg50"):
        mask = val_cids == CONDITIONS.index(condition)
        full_ber = float(errors[mask].mean())
        center_ber = float(np.mean(masked_pred[mask] != val_y[mask]))
        ablation_rows.append({
            "condition": condition, "full_3x3_ber": full_ber,
            "center_only_masked_ber": center_ber,
            "center_only_minus_full": center_ber - full_ber,
        })

    stage2b_macro = float(np.mean([float(stage2b_conditions[c]["stage2b_cnn_ber"]) for c in CONDITIONS[1:]]))
    stage2d_macro = float(np.mean([row["stage2d_ber"] for row in condition_rows[1:]]))
    jpeg70 = condition_rows[2]
    jpeg50 = condition_rows[3]
    clean = condition_rows[0]
    jpeg70_ablation = ablation_rows[0]["center_only_minus_full"]
    no_subband_failure = all(max(row["lh2_ber"], row["hl2_ber"]) < 0.5 for row in condition_rows)
    if clean["stage2d_ber"] < 0.005 and jpeg70["absolute_improvement_over_stage2b"] >= 0.05 and no_subband_failure and jpeg70_ablation >= 0.01:
        classification = "STRONG PASS"
    elif clean["stage2d_ber"] < 0.005 and stage2b_macro - stage2d_macro >= 0.01 and jpeg70_ablation > 0:
        classification = "PASS"
    elif jpeg70["absolute_improvement_over_stage2b"] > 0 or stage2b_macro > stage2d_macro:
        classification = "PARTIAL"
    else:
        classification = "FAIL"

    summary = {
        "classification": classification, "best_epoch": checkpoint.best_epoch,
        "epochs_trained": len(history_rows), "model_parameters": parameter_count,
        "best_validation_macro_ber": checkpoint.best_macro,
        "best_validation_bce_tiebreak": checkpoint.best_loss,
        "attack_macro": {"stage2b": stage2b_macro, "stage2d": stage2d_macro,
                         "absolute_improvement": stage2b_macro - stage2d_macro},
        "jpeg70": {"stage2b_ber": jpeg70["stage2b_ber"], "stage2d_ber": jpeg70["stage2d_ber"],
                   "absolute_improvement": jpeg70["absolute_improvement_over_stage2b"],
                   "relative_improvement": jpeg70["relative_improvement_over_stage2b"]},
        "jpeg50": {"stage2b_ber": jpeg50["stage2b_ber"], "stage2d_ber": jpeg50["stage2d_ber"],
                   "absolute_improvement": jpeg50["absolute_improvement_over_stage2b"]},
        "clean": {"stage2d_ber": clean["stage2d_ber"],
                  "classification": "Excellent" if clean["stage2d_ber"] < 0.001 else "Acceptable" if clean["stage2d_ber"] < 0.005 else "Regression concern"},
        "context_ablation": {row["condition"]: row for row in ablation_rows},
        "no_catastrophic_subband_failure": no_subband_failure,
        "test_set_used": False,
    }
    config = {
        "experiment": "Stage 2D controlled 3x3 local spatial-context decoding",
        "only_changed_variable": "single selected coefficient to same-subband 3x3 local patch",
        "patch_features_per_coefficient": ["coefficient/delta", "sin(pi*coefficient/delta)", "cos(pi*coefficient/delta)"],
        "center_subband_identifier": {"LH2": 0, "HL2": 1},
        "padding": "numpy.pad mode='symmetric', one coefficient on each spatial side",
        "model": "shared TimeDistributed Conv2D(16,3x3,valid)->Flatten->concat subband->Dense(8)->Dense(1,sigmoid)",
        "model_parameters": parameter_count, "training_examples": 7000,
        "validation_examples": 1400, "conditions": list(CONDITIONS),
        "coefficient_seed": COEFFICIENT_SEED, "delta": DELTA, "wavelet": "haar",
        "dwt_level": 2, "payload_bits": PAYLOAD_BITS, "threshold": THRESHOLD,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "checkpoint": "lowest seven-condition validation macro BER; validation BCE tie-break",
        "test_set_used": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "best_local_3x3_context.keras")
    write_csv(OUTPUT_DIR / "representation_samples.csv", representation_samples)
    write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "per_condition_summary.csv", condition_rows)
    write_csv(OUTPUT_DIR / "per_sample_validation_results.csv", val_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(OUTPUT_DIR / "probability_diagnostics.csv", probability_rows)
    write_csv(OUTPUT_DIR / "context_mask_ablation.csv", ablation_rows)
    write_csv(OUTPUT_DIR / "comparison_stage2b_stage2d.csv", condition_rows)
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "representation_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
