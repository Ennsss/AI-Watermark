"""Stage 2: balanced JPEG/re-encode training for the seed-aware decoder."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.suite import jpeg_compression, reencode_jpeg
from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import compute_ber, extract_from_image
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run2_seed_aware_overfit import build_model
from run3a_clean_transfer import construct_features, distribution
from run3b_clean_training import metric_summary
from run_cnn_benchmark import embed_image


TRAIN_DIR = ROOT / "data/curated/train"
VAL_DIR = ROOT / "data/curated/val"
RUN3B_DIR = ROOT / "experiments/run3b_clean_training"
OUTPUT_DIR = ROOT / "experiments/stage2_jpeg_reencode_training"
TRAIN_IMAGE_COUNT = 500
VAL_IMAGE_COUNT = 100
PAYLOADS_PER_IMAGE = 2
TRAIN_PAYLOAD_SEED = 20260810
VAL_PAYLOAD_SEED = 20260809
SCHEDULE_SEED = 20260811
MODEL_SEED = 42
PAYLOAD_BITS = 128
COEFFICIENT_SEED = 42
DELTA = 16.0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
THRESHOLD = 0.5
CONDITIONS = ("clean", "jpeg90", "jpeg70", "jpeg50", "reencode1", "reencode2", "reencode3")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_condition(image: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return image.copy()
    if condition.startswith("jpeg"):
        return jpeg_compression(image, quality=int(condition.removeprefix("jpeg"))).image
    if condition.startswith("reencode"):
        return reencode_jpeg(
            image, passes=int(condition.removeprefix("reencode")), quality=85
        ).image
    raise ValueError(f"Unknown Stage 2 condition: {condition}")


def create_balanced_schedule(sample_count: int) -> list[str]:
    schedule = [CONDITIONS[index % len(CONDITIONS)] for index in range(sample_count)]
    rng = np.random.default_rng(SCHEDULE_SEED)
    rng.shuffle(schedule)
    return schedule


def make_example(
    image: np.ndarray, bits: np.ndarray, condition: str, locations: list[dict]
) -> tuple[np.ndarray, np.ndarray, float]:
    watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
    attacked = apply_condition(watermarked, condition)
    classical_bits, _ = extract_from_image(
        attacked,
        PAYLOAD_BITS,
        COEFFICIENT_SEED,
        DELTA,
        "haar",
        target_subbands=("lh2", "hl2"),
    )
    coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
    return (
        construct_features(coefficient_map, locations),
        classical_bits,
        compute_ber(bits, classical_bits),
    )


def generate_training(
    paths: list[Path], schedule: list[str]
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(TRAIN_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[dict] = []
    sample_id = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[train data] {image_index}/{len(paths)} {path.name}")
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            condition = schedule[sample_id]
            feature, _classical_bits, classical_ber = make_example(
                image, bits, condition, locations
            )
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            features.append(feature)
            targets.append(bits)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_filename": path.name,
                    "payload_index": payload_index,
                    "payload_seed": TRAIN_PAYLOAD_SEED,
                    "payload_fingerprint": fingerprint,
                    "condition": condition,
                    "target_zeros": int(np.sum(bits == 0)),
                    "target_ones": int(np.sum(bits == 1)),
                    "classical_ber": classical_ber,
                }
            )
            sample_id += 1
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.uint8), rows


def generate_validation(
    paths: list[Path], expected_payload_fingerprints: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(VAL_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    classical: list[np.ndarray] = []
    condition_ids: list[int] = []
    rows: list[dict] = []
    base_pair_index = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[validation grid] {image_index}/{len(paths)} {path.name}")
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected_payload_fingerprints[base_pair_index]:
                raise RuntimeError("Validation payload fingerprint mismatch with Run 3B.")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                classical_bits, _ = extract_from_image(
                    attacked,
                    PAYLOAD_BITS,
                    COEFFICIENT_SEED,
                    DELTA,
                    "haar",
                    target_subbands=("lh2", "hl2"),
                )
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                features.append(construct_features(coefficient_map, locations))
                targets.append(bits)
                classical.append(classical_bits)
                condition_ids.append(condition_id)
                rows.append(
                    {
                        "row_id": len(rows),
                        "base_pair_index": base_pair_index,
                        "image_filename": path.name,
                        "payload_index": payload_index,
                        "payload_fingerprint": fingerprint,
                        "condition": condition,
                        "classical_ber": compute_ber(bits, classical_bits),
                    }
                )
            base_pair_index += 1
    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.uint8),
        np.stack(classical).astype(np.uint8),
        np.asarray(condition_ids, dtype=np.int8),
        rows,
    )


def prediction_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = (probabilities >= THRESHOLD) != targets
    return {
        "ber": float(errors.mean()),
        "lh2_ber": float(errors[:, :64].mean()),
        "hl2_ber": float(errors[:, 64:].mean()),
    }


def condition_metrics(
    probabilities: np.ndarray, targets: np.ndarray, condition_ids: np.ndarray
) -> dict[str, dict[str, float | int]]:
    result = {}
    predictions = probabilities >= THRESHOLD
    for condition_id, condition in enumerate(CONDITIONS):
        mask = condition_ids == condition_id
        errors = predictions[mask] != targets[mask]
        sample_bers = errors.mean(axis=1)
        result[condition] = {
            "ber": float(errors.mean()),
            "lh2_ber": float(errors[:, :64].mean()),
            "hl2_ber": float(errors[:, 64:].mean()),
            "perfect_payloads": int(np.sum(sample_bers == 0.0)),
        }
    return result


def main() -> int:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Stage 2.") from exc

    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    run3b_train_paths = [
        row["image_filename"]
        for row in csv.DictReader((RUN3B_DIR / "selected_training_images.csv").open(encoding="utf-8"))
    ]
    run3b_val_paths = [
        row["image_filename"]
        for row in csv.DictReader((RUN3B_DIR / "selected_validation_images.csv").open(encoding="utf-8"))
    ]
    if [path.name for path in train_paths] != run3b_train_paths:
        raise RuntimeError("Stage 2 training image selection does not match Run 3B.")
    if [path.name for path in val_paths] != run3b_val_paths:
        raise RuntimeError("Stage 2 validation image selection does not match Run 3B.")
    run3b_train_payload_rows = list(
        csv.DictReader((RUN3B_DIR / "training_payloads.csv").open(encoding="utf-8"))
    )
    run3b_val_payload_rows = list(
        csv.DictReader((RUN3B_DIR / "validation_payloads.csv").open(encoding="utf-8"))
    )
    schedule = create_balanced_schedule(TRAIN_IMAGE_COUNT * PAYLOADS_PER_IMAGE)
    train_x, train_y, train_rows = generate_training(train_paths, schedule)
    if [row["payload_fingerprint"] for row in train_rows] != [
        row["payload_fingerprint"] for row in run3b_train_payload_rows
    ]:
        raise RuntimeError("Stage 2 training payloads do not match Run 3B.")
    val_x, val_y, val_classical, val_condition_ids, val_rows = generate_validation(
        val_paths, [row["payload_fingerprint"] for row in run3b_val_payload_rows]
    )
    if train_x.shape != (1000, 128, 4) or val_x.shape != (1400, 128, 4):
        raise RuntimeError(f"Unexpected Stage 2 shapes: {train_x.shape}, {val_x.shape}")

    condition_counts = Counter(schedule)
    condition_count_rows = [
        {
            "split": "training",
            "condition": condition,
            "sample_count": condition_counts[condition],
            "fraction": condition_counts[condition] / len(schedule),
        }
        for condition in CONDITIONS
    ] + [
        {
            "split": "validation",
            "condition": condition,
            "sample_count": int(np.sum(val_condition_ids == index)),
            "fraction": float(np.mean(val_condition_ids == index)),
        }
        for index, condition in enumerate(CONDITIONS)
    ]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model(input_features=4)
    if model.count_params() != 369:
        raise RuntimeError("Stage 2 architecture parameter count changed.")
    history_rows: list[dict] = []

    class MacroBerCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_macro = float("inf")
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.best_weights = None
            self.epochs_without_macro_improvement = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_prob = self.model.predict(train_x, batch_size=BATCH_SIZE, verbose=0)
            val_prob = self.model.predict(val_x, batch_size=BATCH_SIZE, verbose=0)
            train_metric = prediction_metrics(train_prob, train_y)
            per_condition = condition_metrics(val_prob, val_y, val_condition_ids)
            macro = float(np.mean([per_condition[name]["ber"] for name in CONDITIONS]))
            val_loss = float(logs.get("val_loss", np.nan))
            strict_improvement = macro < self.best_macro - 1e-12
            tie_loss_improvement = abs(macro - self.best_macro) <= 1e-12 and val_loss < self.best_loss
            self.epochs_without_macro_improvement = (
                0 if strict_improvement else self.epochs_without_macro_improvement + 1
            )
            if strict_improvement or tie_loss_improvement:
                self.best_macro = macro
                self.best_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
            row = {
                "epoch": epoch + 1,
                "train_bce": float(logs.get("loss", np.nan)),
                "train_ber": train_metric["ber"],
                "train_lh2_ber": train_metric["lh2_ber"],
                "train_hl2_ber": train_metric["hl2_ber"],
                "validation_bce": val_loss,
                "validation_macro_ber": macro,
                "is_best_checkpoint": self.best_epoch == epoch + 1,
            }
            for condition in CONDITIONS:
                metrics = per_condition[condition]
                row[f"{condition}_ber"] = metrics["ber"]
                row[f"{condition}_lh2_ber"] = metrics["lh2_ber"]
                row[f"{condition}_hl2_ber"] = metrics["hl2_ber"]
                row[f"{condition}_perfect_payloads"] = metrics["perfect_payloads"]
                row[f"train_count_{condition}"] = condition_counts[condition]
            history_rows.append(row)
            print(
                f"epoch={epoch + 1} train={train_metric['ber']:.6f} macro={macro:.6f} "
                f"clean={per_condition['clean']['ber']:.6f} jpeg50={per_condition['jpeg50']['ber']:.6f} "
                f"reencode3={per_condition['reencode3']['ber']:.6f}"
            )
            if self.epochs_without_macro_improvement >= PATIENCE:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise RuntimeError("No Stage 2 best weights captured.")
            self.model.set_weights(self.best_weights)

    checkpoint = MacroBerCheckpoint()
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
    confidences = np.abs(probabilities - 0.5) * 2.0
    cnn_sample_bers = cnn_errors.mean(axis=1)
    classical_sample_bers = classical_errors.mean(axis=1)
    for index, row in enumerate(val_rows):
        row.update(
            {
                "cnn_ber": float(cnn_sample_bers[index]),
                "classical_perfect": bool(classical_sample_bers[index] == 0.0),
                "cnn_perfect": bool(cnn_sample_bers[index] == 0.0),
                "cnn_mean_confidence": float(confidences[index].mean()),
            }
        )

    condition_summary_rows: list[dict] = []
    bit_rows: list[dict] = []
    probability_rows: list[dict] = []
    subset_rows: list[dict] = []
    for condition_id, condition in enumerate(CONDITIONS):
        mask = val_condition_ids == condition_id
        condition_cnn_errors = cnn_errors[mask]
        condition_classical_errors = classical_errors[mask]
        condition_cnn_bers = cnn_sample_bers[mask]
        condition_classical_bers = classical_sample_bers[mask]
        condition_prob = probabilities[mask]
        condition_pred = predictions[mask]
        condition_targets = val_y[mask]
        condition_conf = confidences[mask]
        condition_summary_rows.append(
            {
                "condition": condition,
                "sample_count": int(mask.sum()),
                "classical_mean_ber": float(condition_classical_errors.mean()),
                "cnn_mean_ber": float(condition_cnn_errors.mean()),
                "classical_perfect_rate": float(np.mean(condition_classical_bers == 0.0)),
                "cnn_perfect_rate": float(np.mean(condition_cnn_bers == 0.0)),
                "cnn_lh2_ber": float(condition_cnn_errors[:, :64].mean()),
                "cnn_hl2_ber": float(condition_cnn_errors[:, 64:].mean()),
                "cnn_minus_classical_ber": float(
                    condition_cnn_errors.mean() - condition_classical_errors.mean()
                ),
                "cnn_better_count": int(np.sum(condition_cnn_bers < condition_classical_bers)),
                "equal_count": int(np.sum(condition_cnn_bers == condition_classical_bers)),
                "cnn_worse_count": int(np.sum(condition_cnn_bers > condition_classical_bers)),
            }
        )
        per_bit_cnn = condition_cnn_errors.mean(axis=0)
        per_bit_classical = condition_classical_errors.mean(axis=0)
        target_frequency = condition_targets.mean(axis=0)
        predicted_frequency = condition_pred.mean(axis=0)
        worst_indices = set(np.argsort(-per_bit_cnn, kind="stable")[:10].tolist())
        for bit in range(PAYLOAD_BITS):
            bit_rows.append(
                {
                    "condition": condition,
                    "bit_index": bit,
                    "subband": "LH2" if bit < 64 else "HL2",
                    "cnn_ber": float(per_bit_cnn[bit]),
                    "classical_ber": float(per_bit_classical[bit]),
                    "target_one_frequency": float(target_frequency[bit]),
                    "predicted_one_frequency": float(predicted_frequency[bit]),
                    "is_worst_10": bit in worst_indices,
                    "cnn_zero_ber": bool(per_bit_cnn[bit] == 0.0),
                }
            )
        incorrect = condition_cnn_errors
        probability_rows.append(
            {
                "condition": condition,
                **{f"probability_{key}": value for key, value in distribution(condition_prob.ravel()).items()},
                "fraction_0.45_to_0.55": float(
                    np.mean((condition_prob >= 0.45) & (condition_prob <= 0.55))
                ),
                "mean_correct_confidence": float(condition_conf[~incorrect].mean()),
                "mean_incorrect_confidence": (
                    float(condition_conf[incorrect].mean()) if incorrect.any() else None
                ),
                "false_zero_count": int(np.sum(incorrect & (condition_targets == 1))),
                "false_one_count": int(np.sum(incorrect & (condition_targets == 0))),
                "zero_ber_bit_positions": int(np.sum(per_bit_cnn == 0.0)),
                "always_zero_outputs": int(np.sum(predicted_frequency == 0.0)),
                "always_one_outputs": int(np.sum(predicted_frequency == 1.0)),
            }
        )
        classical_clean = condition_classical_bers == 0.0
        for subset_name, subset_mask in [
            ("classically_clean", classical_clean),
            ("classically_impaired", ~classical_clean),
        ]:
            subset_count = int(subset_mask.sum())
            subset_rows.append(
                {
                    "condition": condition,
                    "subset": subset_name,
                    "sample_count": subset_count,
                    "classical_mean_ber": (
                        float(condition_classical_bers[subset_mask].mean())
                        if subset_count
                        else None
                    ),
                    "cnn_mean_ber": (
                        float(condition_cnn_bers[subset_mask].mean())
                        if subset_count
                        else None
                    ),
                }
            )

    classical_condition_bers = np.asarray(
        [row["classical_mean_ber"] for row in condition_summary_rows]
    )
    cnn_condition_bers = np.asarray([row["cnn_mean_ber"] for row in condition_summary_rows])
    attack_classical_macro = float(classical_condition_bers[1:].mean())
    attack_cnn_macro = float(cnn_condition_bers[1:].mean())
    attack_absolute_improvement = attack_classical_macro - attack_cnn_macro
    attack_relative_improvement = attack_absolute_improvement / attack_classical_macro
    clean_ber = float(cnn_condition_bers[0])
    clean_retention = (
        "Excellent retention"
        if clean_ber < 0.001
        else "Acceptable retention"
        if clean_ber < 0.005
        else "Regression concern"
    )
    no_random_collapse = bool(
        all(
            row["cnn_mean_ber"] < 0.45 or row["classical_mean_ber"] >= 0.45
            for row in condition_summary_rows[1:]
        )
    )
    no_subband_collapse = bool(
        all(max(row["cnn_lh2_ber"], row["cnn_hl2_ber"]) < 0.139 for row in condition_summary_rows)
    )
    lower_conditions = sum(
        row["cnn_mean_ber"] < row["classical_mean_ber"] for row in condition_summary_rows[1:]
    )
    if (
        clean_ber < 0.001
        and lower_conditions >= 5
        and attack_cnn_macro < attack_classical_macro
        and no_random_collapse
        and no_subband_collapse
    ):
        classification = "STRONG PASS"
    elif (
        clean_ber < 0.005
        and attack_cnn_macro < attack_classical_macro
        and no_random_collapse
        and no_subband_collapse
    ):
        classification = "PASS"
    elif attack_cnn_macro < attack_classical_macro or clean_ber < 0.005:
        classification = "PARTIAL"
    else:
        classification = "FAIL"

    with (RUN3B_DIR / "summary_metrics.json").open(encoding="utf-8") as handle:
        run3b = json.load(handle)
    overall_stage2_errors = cnn_errors
    comparison_rows = [
        {
            "run": "Run 3B clean-trained",
            "training_images": 500,
            "training_conditions": "clean",
            "parameters": 369,
            "clean_ber": run3b["cnn_all_validation"]["mean_ber"],
            "jpeg90_ber": "N/A",
            "jpeg70_ber": "N/A",
            "jpeg50_ber": "N/A",
            "reencode1_ber": "N/A",
            "reencode2_ber": "N/A",
            "reencode3_ber": "N/A",
            "macro_ber": "N/A",
            "perfect_recovery_rate": run3b["cnn_all_validation"]["perfect_rate"],
            "lh2_ber": run3b["subbands"]["LH2"]["cnn_mean_ber"],
            "hl2_ber": run3b["subbands"]["HL2"]["cnn_mean_ber"],
        },
        {
            "run": "Stage 2 attack-trained",
            "training_images": 500,
            "training_conditions": "clean+jpeg90+jpeg70+jpeg50+reencode1+reencode2+reencode3",
            "parameters": 369,
            "clean_ber": clean_ber,
            "jpeg90_ber": cnn_condition_bers[1],
            "jpeg70_ber": cnn_condition_bers[2],
            "jpeg50_ber": cnn_condition_bers[3],
            "reencode1_ber": cnn_condition_bers[4],
            "reencode2_ber": cnn_condition_bers[5],
            "reencode3_ber": cnn_condition_bers[6],
            "macro_ber": float(cnn_condition_bers.mean()),
            "perfect_recovery_rate": float(np.mean(overall_stage2_errors.mean(axis=1) == 0.0)),
            "lh2_ber": float(overall_stage2_errors[:, :64].mean()),
            "hl2_ber": float(overall_stage2_errors[:, 64:].mean()),
        },
    ]
    summary = {
        "classification": classification,
        "best_epoch": checkpoint.best_epoch,
        "epochs_trained": len(history_rows),
        "best_validation_macro_ber": checkpoint.best_macro,
        "best_validation_bce_tiebreak": checkpoint.best_loss,
        "clean_retention": clean_retention,
        "run3b_clean_ber": run3b["cnn_all_validation"]["mean_ber"],
        "stage2_clean_ber": clean_ber,
        "macro_all_seven": {
            "classical": float(classical_condition_bers.mean()),
            "cnn": float(cnn_condition_bers.mean()),
        },
        "macro_attacks_only": {
            "classical": attack_classical_macro,
            "cnn": attack_cnn_macro,
            "absolute_improvement": attack_absolute_improvement,
            "relative_improvement": attack_relative_improvement,
        },
        "worst_condition": {
            "classical_condition": CONDITIONS[int(np.argmax(classical_condition_bers))],
            "classical_ber": float(classical_condition_bers.max()),
            "cnn_condition": CONDITIONS[int(np.argmax(cnn_condition_bers))],
            "cnn_ber": float(cnn_condition_bers.max()),
        },
        "checks": {
            "no_random_collapse": no_random_collapse,
            "no_subband_collapse": no_subband_collapse,
            "attacked_conditions_cnn_lower_count": lower_conditions,
        },
        "payloads": {
            "training_seed": TRAIN_PAYLOAD_SEED,
            "validation_seed": VAL_PAYLOAD_SEED,
            "training_fingerprints_match_run3b": True,
            "validation_fingerprints_match_run3b": True,
            "training_unique_payloads": len({row.tobytes() for row in train_y}),
            "training_one_frequency": float(train_y.mean()),
        },
    }
    config = {
        "experiment": "stage2_jpeg_reencode_training",
        "model": "fresh 369-parameter Run 2 seed-aware Conv1D",
        "features": ["c/delta", "sin(pi*c/delta)", "cos(pi*c/delta)", "subband_id"],
        "training_selection": "same first 500 sorted train PNGs as Run 3B",
        "validation_selection": "same first 100 sorted validation PNGs as Run 3B",
        "training_payload_seed": TRAIN_PAYLOAD_SEED,
        "validation_payload_seed": VAL_PAYLOAD_SEED,
        "schedule_seed": SCHEDULE_SEED,
        "conditions": {
            "clean": "no degradation",
            "jpeg90": "one OpenCV JPEG encode/decode at quality 90",
            "jpeg70": "one OpenCV JPEG encode/decode at quality 70",
            "jpeg50": "one OpenCV JPEG encode/decode at quality 50",
            "reencode1": "one sequential OpenCV JPEG encode/decode pass at quality 85; identical to JPEG Q85",
            "reencode2": "two sequential OpenCV JPEG encode/decode passes, each quality 85",
            "reencode3": "three sequential OpenCV JPEG encode/decode passes, each quality 85",
        },
        "balanced_schedule": "1000 condition labels formed by cyclic allocation then deterministically shuffled",
        "validation_grid_examples": 1400,
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
        "checkpoint": "lowest macro-average validation BER across seven conditions; validation BCE tie-break",
        "excluded_attacks": ["resize", "crop", "noise", "filter", "rotation", "sharpen", "combined"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "best_seed_aware_jpeg_reencode.keras")
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
    write_csv(OUTPUT_DIR / "payload_reproducibility.csv", train_rows)
    write_csv(OUTPUT_DIR / "condition_counts.csv", condition_count_rows)
    write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "per_condition_summary.csv", condition_summary_rows)
    write_csv(OUTPUT_DIR / "per_sample_validation_results.csv", val_rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(OUTPUT_DIR / "probability_diagnostics.csv", probability_rows)
    write_csv(OUTPUT_DIR / "classical_subset_diagnostics.csv", subset_rows)
    write_csv(OUTPUT_DIR / "comparison_to_run3b.csv", comparison_rows)
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved Stage 2 artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
