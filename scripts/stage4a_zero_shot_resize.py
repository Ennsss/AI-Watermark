"""Stage 4A: zero-shot resize evaluation of the frozen delta-24 decoder."""

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

from attacks.suite import resize_scale
from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import extract_from_image
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run3a_clean_transfer import distribution
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import write_csv
from stage3a_delta_calibration import features_from_map


OUTPUT_DIR = ROOT / "experiments/stage4a_zero_shot_resize"
STAGE3A_DIR = ROOT / "experiments/stage3a_delta_calibration"
CHECKPOINT = STAGE3A_DIR / "delta_24/best_seed_aware_delta24.keras"
VAL_DIR = ROOT / "data/curated/val"
VAL_IMAGES = 100
PAYLOADS_PER_IMAGE = 2
PAYLOAD_SEED = 20260809
PAYLOAD_BITS = 128
COEFFICIENT_SEED = 42
DELTA = 24.0
THRESHOLD = 0.5
BATCH_SIZE = 32
CONDITIONS = ("clean", "resize75", "resize50", "resize25")
SCALES = {"clean": 1.0, "resize75": 0.75, "resize50": 0.50, "resize25": 0.25}


def apply_condition(image: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return image.copy()
    return resize_scale(image, scale=SCALES[condition]).image


def selected_coefficients(coefficient_map: np.ndarray, locations: list[dict]) -> np.ndarray:
    return np.asarray([
        coefficient_map[row["row"], row["column"], row["channel"]]
        for row in locations
    ], dtype=np.float32)


def circular_phase_displacement(clean: np.ndarray, attacked: np.ndarray) -> np.ndarray:
    clean_phase = np.mod(clean / DELTA, 1.0)
    attacked_phase = np.mod(attacked / DELTA, 1.0)
    difference = np.abs(clean_phase - attacked_phase)
    return np.minimum(difference, 1.0 - difference)


def metric_summary(sample_bers: np.ndarray) -> dict:
    return {
        "mean_ber": float(sample_bers.mean()),
        "median_ber": float(np.median(sample_bers)),
        "std_ber": float(sample_bers.std()),
        "minimum_ber": float(sample_bers.min()),
        "maximum_ber": float(sample_bers.max()),
        "perfect_payload_count": int(np.sum(sample_bers == 0)),
        "perfect_payload_rate": float(np.mean(sample_bers == 0)),
    }


def transfer_label(ber: float) -> str:
    if ber < 0.01:
        return "EXCELLENT ZERO-SHOT"
    if ber < 0.05:
        return "GOOD ZERO-SHOT"
    if ber < 0.20:
        return "PARTIAL ZERO-SHOT"
    if ber < 0.40:
        return "POOR ZERO-SHOT"
    return "NEAR RANDOM"


def main() -> int:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required to load the frozen Stage 3A checkpoint.") from exc

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Frozen delta-24 checkpoint not found: {CHECKPOINT}")

    stage3a_config = json.loads((STAGE3A_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    if 24 not in stage3a_config["deltas"] or stage3a_config["coefficient_seed"] != COEFFICIENT_SEED:
        raise RuntimeError("Stage 3A configuration does not verify delta 24 and seed 42.")
    model = keras.models.load_model(CHECKPOINT, compile=False)
    if model.count_params() != 369 or model.input_shape != (None, 128, 4):
        raise RuntimeError(f"Frozen model verification failed: {model.count_params()}, {model.input_shape}")

    paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGES]
    stage3a_payloads = [
        row for row in csv.DictReader(
            (STAGE3A_DIR / "payload_reproducibility_metadata.csv").open(encoding="utf-8")
        ) if row["split"] == "validation"
    ]
    if len(paths) != VAL_IMAGES or len(stage3a_payloads) != VAL_IMAGES * PAYLOADS_PER_IMAGE:
        raise RuntimeError("Stage 4A validation identity source is incomplete.")

    total = VAL_IMAGES * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    features = np.empty((total, PAYLOAD_BITS, 4), dtype=np.float32)
    targets = np.empty((total, PAYLOAD_BITS), dtype=np.uint8)
    classical = np.empty_like(targets)
    coefficients = np.empty((total, PAYLOAD_BITS), dtype=np.float32)
    condition_ids = np.empty(total, dtype=np.int8)
    rows, payload_rows = [], []
    rng = np.random.default_rng(PAYLOAD_SEED)
    locations = location_rows((128, 128))
    row_id = 0
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[Stage 4A] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            expected = stage3a_payloads[base_pair]
            if fingerprint != expected["payload_fingerprint"] or path.name != expected["image_filename"]:
                raise RuntimeError("Stage 4A payload/image identity differs from Stage 3A.")
            payload_rows.append({
                "base_pair_index": base_pair, "image_filename": path.name,
                "payload_index": payload_index, "payload_seed": PAYLOAD_SEED,
                "payload_fingerprint": fingerprint,
            })
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                features[row_id] = features_from_map(coefficient_map, locations, DELTA)
                coefficients[row_id] = selected_coefficients(coefficient_map, locations)
                targets[row_id] = bits
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

    if features.shape != (800, 128, 4) or not np.isfinite(features).all():
        raise RuntimeError("Stage 4A feature verification failed.")
    probabilities = np.asarray(model.predict(features, batch_size=BATCH_SIZE, verbose=0))
    predictions = (probabilities >= THRESHOLD).astype(np.uint8)
    cnn_errors = predictions != targets
    classical_errors = classical != targets
    cnn_sample_bers = cnn_errors.mean(axis=1)
    classical_sample_bers = classical_errors.mean(axis=1)
    confidences = np.abs(probabilities - 0.5) * 2.0

    condition_rows, bit_rows, probability_rows, displacement_rows = [], [], [], []
    clean_coefficients = coefficients[condition_ids == 0]
    clean_classical = classical[condition_ids == 0]
    for condition_id, condition in enumerate(CONDITIONS):
        mask = condition_ids == condition_id
        cnn_condition_errors = cnn_errors[mask]
        classical_condition_errors = classical_errors[mask]
        cnn_bers = cnn_sample_bers[mask]
        classical_bers = classical_sample_bers[mask]
        classical_stats = metric_summary(classical_bers)
        cnn_stats = metric_summary(cnn_bers)
        cnn_mean = cnn_stats["mean_ber"]
        classical_mean = classical_stats["mean_ber"]
        condition_rows.append({
            "condition": condition,
            **{f"classical_{key}": value for key, value in classical_stats.items()},
            **{f"cnn_{key}": value for key, value in cnn_stats.items()},
            "cnn_minus_classical_ber": cnn_mean - classical_mean,
            "absolute_cnn_improvement": classical_mean - cnn_mean,
            "relative_cnn_ber_reduction": (classical_mean - cnn_mean) / classical_mean,
            "classical_lh2_ber": float(classical_condition_errors[:, :64].mean()),
            "classical_hl2_ber": float(classical_condition_errors[:, 64:].mean()),
            "cnn_lh2_ber": float(cnn_condition_errors[:, :64].mean()),
            "cnn_hl2_ber": float(cnn_condition_errors[:, 64:].mean()),
            "cnn_better_count": int(np.sum(cnn_bers < classical_bers)),
            "equal_count": int(np.sum(cnn_bers == classical_bers)),
            "cnn_worse_count": int(np.sum(cnn_bers > classical_bers)),
            "zero_shot_classification": transfer_label(cnn_mean),
        })
        per_bit = cnn_condition_errors.mean(axis=0)
        per_bit_classical = classical_condition_errors.mean(axis=0)
        target_frequency = targets[mask].mean(axis=0)
        predicted_frequency = predictions[mask].mean(axis=0)
        worst = set(np.argsort(-per_bit, kind="stable")[:10].tolist())
        for bit in range(PAYLOAD_BITS):
            bit_rows.append({
                "condition": condition, "bit_index": bit,
                "subband": "LH2" if bit < 64 else "HL2",
                "cnn_ber": float(per_bit[bit]),
                "classical_ber": float(per_bit_classical[bit]),
                "zero_ber": bool(per_bit[bit] == 0), "is_worst_10": bit in worst,
                "target_one_frequency": float(target_frequency[bit]),
                "predicted_one_frequency": float(predicted_frequency[bit]),
                "always_zero": bool(predicted_frequency[bit] == 0),
                "always_one": bool(predicted_frequency[bit] == 1),
            })
        condition_prob = probabilities[mask]
        incorrect = cnn_condition_errors
        condition_confidence = confidences[mask]
        probability_rows.append({
            "condition": condition,
            **{f"probability_{key}": value for key, value in distribution(condition_prob.ravel()).items()},
            "fraction_0.45_to_0.55": float(np.mean((condition_prob >= 0.45) & (condition_prob <= 0.55))),
            "correct_prediction_confidence": float(condition_confidence[~incorrect].mean()),
            "incorrect_prediction_confidence": float(condition_confidence[incorrect].mean()) if incorrect.any() else None,
            "false_zero_count": int(np.sum(incorrect & (targets[mask] == 1))),
            "false_one_count": int(np.sum(incorrect & (targets[mask] == 0))),
            "target_one_frequency": float(targets[mask].mean()),
            "predicted_one_frequency": float(predictions[mask].mean()),
            "zero_ber_bit_positions": int(np.sum(per_bit == 0)),
            "always_zero_positions": int(np.sum(predicted_frequency == 0)),
            "always_one_positions": int(np.sum(predicted_frequency == 1)),
        })
        displacement = np.abs(coefficients[mask] - clean_coefficients)
        phase_displacement = circular_phase_displacement(clean_coefficients, coefficients[mask])
        displacement_rows.append({
            "condition": condition, "count": int(displacement.size),
            "median_absolute_coefficient_displacement": float(np.median(displacement)),
            "mean_absolute_coefficient_displacement": float(np.mean(displacement)),
            "p90_absolute_coefficient_displacement": float(np.quantile(displacement, 0.90)),
            "fraction_displacement_gt_delta_over_4": float(np.mean(displacement > DELTA / 4)),
            "fraction_displacement_gt_delta_over_2": float(np.mean(displacement > DELTA / 2)),
            "median_circular_phase_displacement": float(np.median(phase_displacement)),
            "mean_circular_phase_displacement": float(np.mean(phase_displacement)),
            "p90_circular_phase_displacement": float(np.quantile(phase_displacement, 0.90)),
            "classical_decision_flip_rate_vs_clean": float(np.mean(classical[mask] != clean_classical)),
        })

    for index, row in enumerate(rows):
        row.update({
            "classical_ber": float(classical_sample_bers[index]),
            "cnn_ber": float(cnn_sample_bers[index]),
            "cnn_mean_probability": float(probabilities[index].mean()),
            "cnn_mean_confidence": float(confidences[index].mean()),
        })

    stage3a_conditions = {
        row["condition"]: row for row in csv.DictReader(
            (STAGE3A_DIR / "per_condition_summary.csv").open(encoding="utf-8")
        ) if int(row["delta"]) == 24
    }
    clean_current = condition_rows[0]
    clean_cnn_reference = float(stage3a_conditions["clean"]["cnn_ber"])
    clean_classical_reference = float(stage3a_conditions["clean"]["classical_ber"])
    comparison_clean = {
        "stage3a_delta24_cnn_clean_ber": clean_cnn_reference,
        "stage4a_cnn_clean_ber": clean_current["cnn_mean_ber"],
        "cnn_absolute_difference": clean_current["cnn_mean_ber"] - clean_cnn_reference,
        "stage3a_delta24_classical_clean_ber": clean_classical_reference,
        "stage4a_classical_clean_ber": clean_current["classical_mean_ber"],
        "classical_absolute_difference": clean_current["classical_mean_ber"] - clean_classical_reference,
        "material_difference_threshold": 0.001,
        "reproducible": bool(
            abs(clean_current["cnn_mean_ber"] - clean_cnn_reference) < 0.001
            and abs(clean_current["classical_mean_ber"] - clean_classical_reference) < 0.001
        ),
    }
    if not comparison_clean["reproducible"]:
        raise RuntimeError(f"Clean reproducibility failed: {comparison_clean}")

    classical_trend = [row["classical_mean_ber"] for row in condition_rows]
    cnn_trend = [row["cnn_mean_ber"] for row in condition_rows]
    monotonically_increasing_classical = all(a <= b for a, b in zip(classical_trend, classical_trend[1:]))
    monotonically_increasing_cnn = all(a <= b for a, b in zip(cnn_trend, cnn_trend[1:]))
    resize_rows = condition_rows[1:]
    if all(row["cnn_mean_ber"] < 0.05 and row["cnn_mean_ber"] < row["classical_mean_ber"] for row in resize_rows):
        overall = "STRONG ZERO-SHOT ROBUSTNESS"
        stage4b = False
        decision_reason = "All resize levels remain low-BER and the CNN consistently improves classical extraction."
    elif resize_rows[-1]["cnn_mean_ber"] >= 0.40 and resize_rows[-1]["classical_mean_ber"] >= 0.40:
        overall = "SEVERE RESIZE LIMITATION"
        stage4b = False
        decision_reason = "Resize25 is near random for both branches; diagnose alignment/information effects before training."
    elif any(row["cnn_mean_ber"] >= 0.05 for row in resize_rows):
        overall = "PARTIAL ZERO-SHOT ROBUSTNESS" if resize_rows[0]["cnn_mean_ber"] < 0.20 else "RESIZE DISTRIBUTION SHIFT"
        stage4b = True
        decision_reason = "At least one resize level degrades materially while remaining below the both-branches-near-random stop condition."
    else:
        overall = "RESIZE DISTRIBUTION SHIFT"
        stage4b = True
        decision_reason = "Resize introduces a measurable distribution shift that warrants controlled exposure."

    verification = {
        "validation_images": len(paths), "payloads_per_image": PAYLOADS_PER_IMAGE,
        "base_pairs": len(payload_rows), "validation_grid_examples": len(rows),
        "image_filenames_match_stage3a": True, "payload_fingerprints_match_stage3a": True,
        "payload_seed": PAYLOAD_SEED, "coefficient_seed": COEFFICIENT_SEED,
        "delta": DELTA, "payload_bits": PAYLOAD_BITS,
        "selected_locations": len(locations),
        "lh2_locations": sum(row["subband"] == "LH2" for row in locations),
        "hl2_locations": sum(row["subband"] == "HL2" for row in locations),
        "features_shape": list(features.shape), "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()), "test_directory_accessed": False,
    }
    resize_implementation = {
        "function": "attacks.suite.resize_scale",
        "library": "OpenCV cv2.resize",
        "color_handling": "RGB to BGR before resize; BGR to RGB after restore",
        "dimension_rule": "max(1, int(round(original_dimension * scale)))",
        "downscale_interpolation": "cv2.INTER_AREA",
        "upscale_interpolation": "cv2.INTER_LANCZOS4",
        "original_dimensions": [512, 512],
        "intermediate_dimensions": {"resize75": [384, 384], "resize50": [256, 256], "resize25": [128, 128]},
        "restored_dimensions": [512, 512],
    }
    config = {
        "experiment": "Stage 4A zero-shot resize robustness evaluation",
        "evaluation_only": True, "training_performed": False,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "model_parameters": int(model.count_params()), "delta": DELTA,
        "coefficient_seed": COEFFICIENT_SEED, "payload_bits": PAYLOAD_BITS,
        "wavelet": "haar", "dwt_level": 2, "subbands": ["LH2", "HL2"],
        "representation": ["coefficient/delta", "sin(pi*coefficient/delta)", "cos(pi*coefficient/delta)", "subband_id"],
        "threshold": THRESHOLD, "conditions": list(CONDITIONS),
        "excluded": ["training", "fine-tuning", "JPEG", "crop", "combined attacks", "test set"],
    }
    summary = {
        "overall_outcome": overall,
        "condition_classifications": {row["condition"]: row["zero_shot_classification"] for row in resize_rows},
        "classical_ber_monotonically_increases_with_severity": monotonically_increasing_classical,
        "cnn_ber_monotonically_increases_with_severity": monotonically_increasing_cnn,
        "stage4b_resize_training_justified": stage4b,
        "decision_reason": decision_reason,
        "clean_reproducibility": comparison_clean,
        "training_performed": False, "test_set_used": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "selected_validation_images.csv", [
        {"selection_order": index, "image_filename": path.name}
        for index, path in enumerate(paths, 1)
    ])
    write_csv(OUTPUT_DIR / "payload_reproducibility_metadata.csv", payload_rows)
    write_csv(OUTPUT_DIR / "per_condition_summary.csv", condition_rows)
    write_csv(OUTPUT_DIR / "per_sample_results.csv", rows)
    write_csv(OUTPUT_DIR / "per_bit_diagnostics.csv", bit_rows)
    write_csv(OUTPUT_DIR / "probability_diagnostics.csv", probability_rows)
    write_csv(OUTPUT_DIR / "coefficient_displacement_summary.csv", displacement_rows)
    write_csv(OUTPUT_DIR / "comparison_classical_cnn.csv", condition_rows)
    write_csv(OUTPUT_DIR / "comparison_to_stage3a_clean.csv", [comparison_clean])
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "data_identity_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "resize_implementation.json").write_text(json.dumps(resize_implementation, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": condition_rows, "displacement": displacement_rows, "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
