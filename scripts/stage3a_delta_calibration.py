"""Stage 3A: controlled QIM delta calibration with fresh Stage 2B decoders."""

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
from evaluation.metrics import compute_psnr, compute_ssim
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import extract_from_image
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run2_seed_aware_overfit import build_model
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import (
    BATCH_SIZE, COEFFICIENT_SEED, CONDITIONS, LEARNING_RATE, MAX_EPOCHS,
    MODEL_SEED, PATIENCE, PAYLOAD_BITS, PAYLOADS_PER_IMAGE, RUN3B_DIR,
    THRESHOLD, TRAIN_DIR, TRAIN_IMAGE_COUNT, TRAIN_PAYLOAD_SEED, VAL_DIR,
    VAL_IMAGE_COUNT, VAL_PAYLOAD_SEED, apply_condition, condition_metrics,
    prediction_metrics, write_csv,
)

OUTPUT_DIR = ROOT / "experiments/stage3a_delta_calibration"
STAGE2B_DIR = ROOT / "experiments/stage2b_full_factorial_jpeg_reencode"
DELTAS = (8.0, 16.0, 24.0, 32.0)
SHUFFLE_SEEDS = {8.0: 20260820, 16.0: 20260821, 24.0: 20260822, 32.0: 20260823}
MEANINGFUL_MACRO_GAIN = 0.01


def features_from_map(coefficient_map: np.ndarray, locations: list[dict], delta: float) -> np.ndarray:
    features = np.empty((PAYLOAD_BITS, 4), dtype=np.float32)
    for location in locations:
        bit = location["bit_index"]
        coefficient = float(coefficient_map[
            location["row"], location["column"], location["channel"]
        ])
        scaled = coefficient / delta
        features[bit] = [
            scaled, np.sin(np.pi * scaled), np.cos(np.pi * scaled),
            float(location["channel"]),
        ]
    return features


def generate_training(paths: list[Path], expected: list[str], delta: float):
    total = TRAIN_IMAGE_COUNT * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    x = np.empty((total, PAYLOAD_BITS, 4), dtype=np.float32)
    y = np.empty((total, PAYLOAD_BITS), dtype=np.uint8)
    condition_ids = np.empty(total, dtype=np.int8)
    rng = np.random.default_rng(TRAIN_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    payload_rows = []
    row_id = 0
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 50 == 0 or image_index == len(paths):
            print(f"[delta {delta:g} train] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected[base_pair]:
                raise RuntimeError(f"Delta {delta:g} training payload mismatch.")
            if delta == DELTAS[0]:
                payload_rows.append({
                    "split": "training", "base_pair_index": base_pair,
                    "image_filename": path.name, "payload_index": payload_index,
                    "payload_seed": TRAIN_PAYLOAD_SEED, "payload_fingerprint": fingerprint,
                })
            watermarked = embed_image(image, bits, delta, "haar", COEFFICIENT_SEED)
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                x[row_id] = features_from_map(coefficient_map, locations, delta)
                y[row_id] = bits
                condition_ids[row_id] = condition_id
                row_id += 1
            base_pair += 1
    permutation = np.random.default_rng(SHUFFLE_SEEDS[delta]).permutation(total)
    return x[permutation], y[permutation], condition_ids[permutation], payload_rows


def generate_validation(paths: list[Path], expected: list[str], delta: float):
    total = VAL_IMAGE_COUNT * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    x = np.empty((total, PAYLOAD_BITS, 4), dtype=np.float32)
    y = np.empty((total, PAYLOAD_BITS), dtype=np.uint8)
    classical = np.empty_like(y)
    condition_ids = np.empty(total, dtype=np.int8)
    rng = np.random.default_rng(VAL_PAYLOAD_SEED)
    locations = location_rows((128, 128))
    rows, fidelity_rows, payload_rows = [], [], []
    row_id = 0
    base_pair = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[delta {delta:g} validation] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected[base_pair]:
                raise RuntimeError(f"Delta {delta:g} validation payload mismatch.")
            if delta == DELTAS[0]:
                payload_rows.append({
                    "split": "validation", "base_pair_index": base_pair,
                    "image_filename": path.name, "payload_index": payload_index,
                    "payload_seed": VAL_PAYLOAD_SEED, "payload_fingerprint": fingerprint,
                })
            watermarked = embed_image(image, bits, delta, "haar", COEFFICIENT_SEED)
            fidelity_rows.append({
                "delta": int(delta), "base_pair_index": base_pair,
                "image_filename": path.name, "payload_index": payload_index,
                "payload_fingerprint": fingerprint,
                "psnr_rgb_db": compute_psnr(image, watermarked),
                "ssim_rgb": compute_ssim(image, watermarked),
            })
            for condition_id, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                x[row_id] = features_from_map(coefficient_map, locations, delta)
                y[row_id] = bits
                condition_ids[row_id] = condition_id
                classical[row_id], _ = extract_from_image(
                    attacked, PAYLOAD_BITS, COEFFICIENT_SEED, delta, "haar",
                    target_subbands=("lh2", "hl2"),
                )
                rows.append({
                    "delta": int(delta), "row_id_within_delta": row_id,
                    "base_pair_index": base_pair, "image_filename": path.name,
                    "payload_index": payload_index, "payload_fingerprint": fingerprint,
                    "condition": condition,
                })
                row_id += 1
            base_pair += 1
    return x, y, classical, condition_ids, rows, fidelity_rows, payload_rows


def stats(values: np.ndarray, prefix: str) -> dict:
    return {
        f"mean_{prefix}": float(np.mean(values)),
        f"median_{prefix}": float(np.median(values)),
        f"std_{prefix}": float(np.std(values)),
        f"minimum_{prefix}": float(np.min(values)),
        f"maximum_{prefix}": float(np.max(values)),
    }


def train_delta(train_x, train_y, train_cids, val_x, val_y, val_cids, delta: float):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model(input_features=4)
    if model.count_params() != 369:
        raise RuntimeError("Stage 3A model is not the frozen 369-parameter decoder.")
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
            train_prob = self.model.predict(train_x, batch_size=BATCH_SIZE, verbose=0)
            val_prob = self.model.predict(val_x, batch_size=BATCH_SIZE, verbose=0)
            train_metric = prediction_metrics(train_prob, train_y)
            val_conditions = condition_metrics(val_prob, val_y, val_cids)
            macro = float(np.mean([val_conditions[c]["ber"] for c in CONDITIONS]))
            val_loss = float(logs.get("val_loss", np.nan))
            strict = macro < self.best_macro - 1e-12
            tie = abs(macro - self.best_macro) <= 1e-12 and val_loss < self.best_loss
            self.stale = 0 if strict else self.stale + 1
            if strict or tie:
                self.best_macro, self.best_loss = macro, val_loss
                self.best_epoch, self.best_weights = epoch + 1, self.model.get_weights()
            row = {
                "delta": int(delta), "epoch": epoch + 1,
                "train_bce": float(logs.get("loss", np.nan)),
                "train_ber": train_metric["ber"],
                "train_lh2_ber": train_metric["lh2_ber"],
                "train_hl2_ber": train_metric["hl2_ber"],
                "validation_bce": val_loss, "validation_macro_ber": macro,
                "is_best_checkpoint": self.best_epoch == epoch + 1,
            }
            for condition in CONDITIONS:
                metrics = val_conditions[condition]
                row[f"validation_{condition}_ber"] = metrics["ber"]
                row[f"validation_{condition}_lh2_ber"] = metrics["lh2_ber"]
                row[f"validation_{condition}_hl2_ber"] = metrics["hl2_ber"]
            history_rows.append(row)
            print(
                f"delta={delta:g} epoch={epoch+1} train={train_metric['ber']:.6f} "
                f"macro={macro:.6f} clean={val_conditions['clean']['ber']:.6f} "
                f"jpeg70={val_conditions['jpeg70']['ber']:.6f} jpeg50={val_conditions['jpeg50']['ber']:.6f}",
                flush=True,
            )
            if self.stale >= PATIENCE:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise RuntimeError(f"No checkpoint captured for delta {delta:g}.")
            self.model.set_weights(self.best_weights)

    checkpoint = MacroCheckpoint()
    model.fit(
        train_x, train_y.astype(np.float32),
        validation_data=(val_x, val_y.astype(np.float32)),
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, shuffle=True,
        callbacks=[checkpoint], verbose=0,
    )
    return model, checkpoint, history_rows


def main() -> int:
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for Stage 3A.") from exc
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    train_payload_expected = list(csv.DictReader((RUN3B_DIR / "training_payloads.csv").open(encoding="utf-8")))
    val_payload_expected = list(csv.DictReader((RUN3B_DIR / "validation_payloads.csv").open(encoding="utf-8")))
    stage2b_rows = list(csv.DictReader((STAGE2B_DIR / "per_sample_validation_results.csv").open(encoding="utf-8")))
    stage2b_conditions = {
        row["condition"]: row for row in csv.DictReader(
            (STAGE2B_DIR / "per_condition_summary.csv").open(encoding="utf-8")
        )
    }

    all_fidelity, all_conditions, all_samples, all_payloads = [], [], [], []
    all_delta_summaries, comparison_rows = [], []
    for delta in DELTAS:
        print(f"=== DELTA {delta:g} ===", flush=True)
        train_x, train_y, train_cids, train_payload_rows = generate_training(
            train_paths, [row["payload_fingerprint"] for row in train_payload_expected], delta
        )
        val_x, val_y, val_classical, val_cids, val_rows, fidelity_rows, val_payload_rows = generate_validation(
            val_paths, [row["payload_fingerprint"] for row in val_payload_expected], delta
        )
        if train_x.shape != (7000, 128, 4) or val_x.shape != (1400, 128, 4):
            raise RuntimeError(f"Delta {delta:g} tensor shape mismatch.")
        if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
            raise RuntimeError(f"Delta {delta:g} contains non-finite features.")
        if delta == DELTAS[0]:
            all_payloads.extend(train_payload_rows + val_payload_rows)
        if len(val_rows) != len(stage2b_rows):
            raise RuntimeError("Validation grid row count differs from Stage 2B.")
        for current, previous in zip(val_rows, stage2b_rows):
            for key in ("image_filename", "payload_index", "payload_fingerprint", "condition"):
                if str(current[key]) != str(previous[key]):
                    raise RuntimeError(f"Delta {delta:g} validation identity mismatch at {key}.")

        model, checkpoint, history = train_delta(
            train_x, train_y, train_cids, val_x, val_y, val_cids, delta
        )
        probabilities = np.asarray(model.predict(val_x, batch_size=BATCH_SIZE, verbose=0))
        predictions = (probabilities >= THRESHOLD).astype(np.uint8)
        errors = predictions != val_y
        classical_errors = val_classical != val_y
        sample_bers = errors.mean(axis=1)
        classical_sample_bers = classical_errors.mean(axis=1)
        delta_condition_rows = []
        for condition_id, condition in enumerate(CONDITIONS):
            mask = val_cids == condition_id
            condition_errors = errors[mask]
            condition_classical = classical_errors[mask]
            condition_bers = sample_bers[mask]
            condition_classical_bers = classical_sample_bers[mask]
            target_frequency = float(val_y[mask].mean())
            predicted_frequency = float(predictions[mask].mean())
            row = {
                "delta": int(delta), "condition": condition,
                "classical_ber": float(condition_classical.mean()),
                "cnn_ber": float(condition_errors.mean()),
                "cnn_perfect_recovery_rate": float(np.mean(condition_bers == 0)),
                "cnn_lh2_ber": float(condition_errors[:, :64].mean()),
                "cnn_hl2_ber": float(condition_errors[:, 64:].mean()),
                "target_one_frequency": target_frequency,
                "predicted_one_frequency": predicted_frequency,
                "false_zero_count": int(np.sum(condition_errors & (val_y[mask] == 1))),
                "false_one_count": int(np.sum(condition_errors & (val_y[mask] == 0))),
                "cnn_better_than_classical": int(np.sum(condition_bers < condition_classical_bers)),
                "cnn_equal_classical": int(np.sum(condition_bers == condition_classical_bers)),
                "cnn_worse_than_classical": int(np.sum(condition_bers > condition_classical_bers)),
            }
            delta_condition_rows.append(row)
            all_conditions.append(row)
            if delta == 16.0:
                reference = float(stage2b_conditions[condition]["stage2b_cnn_ber"])
                comparison_rows.append({
                    "condition": condition, "stage2b_reference_ber": reference,
                    "stage3a_delta16_ber": row["cnn_ber"],
                    "absolute_difference": row["cnn_ber"] - reference,
                })
        for index, row in enumerate(val_rows):
            row.update({
                "classical_ber": float(classical_sample_bers[index]),
                "cnn_ber": float(sample_bers[index]),
                "cnn_mean_probability": float(probabilities[index].mean()),
            })
            all_samples.append(row)

        fidelity_array_psnr = np.asarray([row["psnr_rgb_db"] for row in fidelity_rows])
        fidelity_array_ssim = np.asarray([row["ssim_rgb"] for row in fidelity_rows])
        all_fidelity.extend(fidelity_rows)
        clean_classical_bers = classical_sample_bers[val_cids == 0]
        cnn_values = np.asarray([row["cnn_ber"] for row in delta_condition_rows])
        classical_values = np.asarray([row["classical_ber"] for row in delta_condition_rows])
        reencode_macro = float(cnn_values[4:7].mean())
        attack_macro = float(cnn_values[1:].mean())
        classical_attack_macro = float(classical_values[1:].mean())
        delta_summary = {
            "delta": int(delta), **stats(fidelity_array_psnr, "psnr_rgb_db"),
            **stats(fidelity_array_ssim, "ssim_rgb"),
            "classical_clean_mean_ber": float(clean_classical_bers.mean()),
            "classical_clean_median_ber": float(np.median(clean_classical_bers)),
            "classical_clean_maximum_ber": float(clean_classical_bers.max()),
            "classical_clean_perfect_payloads": int(np.sum(clean_classical_bers == 0)),
            "cnn_clean_ber": float(cnn_values[0]), "cnn_jpeg90_ber": float(cnn_values[1]),
            "cnn_jpeg70_ber": float(cnn_values[2]), "cnn_jpeg50_ber": float(cnn_values[3]),
            "cnn_reencode_macro_ber": reencode_macro,
            "cnn_seven_condition_macro_ber": float(cnn_values.mean()),
            "cnn_six_attack_macro_ber": attack_macro,
            "classical_six_attack_macro_ber": classical_attack_macro,
            "cnn_absolute_improvement_over_classical_attack_macro": classical_attack_macro - attack_macro,
            "cnn_relative_improvement_over_classical_attack_macro": (classical_attack_macro - attack_macro) / classical_attack_macro,
            "best_epoch": checkpoint.best_epoch, "epochs_trained": len(history),
            "best_validation_bce": checkpoint.best_loss,
            "shuffle_seed": SHUFFLE_SEEDS[delta],
        }
        all_delta_summaries.append(delta_summary)
        delta_dir = OUTPUT_DIR / f"delta_{int(delta)}"
        delta_dir.mkdir(parents=True, exist_ok=True)
        model.save(delta_dir / f"best_seed_aware_delta{int(delta)}.keras")
        write_csv(OUTPUT_DIR / f"training_history_delta{int(delta)}.csv", history)
        del train_x, train_y, train_cids, val_x, val_y, val_classical, probabilities, predictions

    by_delta = {row["delta"]: row for row in all_delta_summaries}
    reference = by_delta[16]
    eligible = [
        delta for delta in (24, 32)
        if reference["cnn_six_attack_macro_ber"] - by_delta[delta]["cnn_six_attack_macro_ber"]
        >= MEANINGFUL_MACRO_GAIN
    ]
    selected_delta = min(eligible) if eligible else 16
    selected = by_delta[selected_delta]
    fidelity_pair_rows = []
    fidelity_by_delta = {
        delta: sorted((row for row in all_fidelity if row["delta"] == delta), key=lambda r: r["base_pair_index"])
        for delta in (8, 16, 24, 32)
    }
    for candidate in (8, 24, 32):
        psnr_diff = np.asarray([
            right["psnr_rgb_db"] - left["psnr_rgb_db"]
            for left, right in zip(fidelity_by_delta[16], fidelity_by_delta[candidate])
        ])
        ssim_diff = np.asarray([
            right["ssim_rgb"] - left["ssim_rgb"]
            for left, right in zip(fidelity_by_delta[16], fidelity_by_delta[candidate])
        ])
        fidelity_pair_rows.append({
            "reference_delta": 16, "comparison_delta": candidate,
            "mean_paired_psnr_difference_comparison_minus_reference_db": float(psnr_diff.mean()),
            "mean_paired_ssim_difference_comparison_minus_reference": float(ssim_diff.mean()),
        })
    reproducibility_max_difference = max(abs(float(row["absolute_difference"])) for row in comparison_rows)
    jpeg50_ceiling = all(by_delta[d]["cnn_jpeg50_ber"] >= 0.40 for d in by_delta)
    decision = {
        "selection_rule": (
            "Select the smallest delta above 16 whose six-attack CNN macro BER improves "
            f"by at least {MEANINGFUL_MACRO_GAIN:.2f} absolute; otherwise retain 16. "
            "Report paired fidelity loss without inventing a cutoff."
        ),
        "selected_delta": selected_delta,
        "reason": (
            f"Delta {selected_delta} is the smallest value satisfying the fixed robustness rule."
            if eligible else "Neither delta 24 nor 32 satisfied the fixed meaningful robustness rule; delta 16 is retained."
        ),
        "delta16_reproducibility_max_condition_ber_difference": reproducibility_max_difference,
        "delta16_reasonably_reproducible": reproducibility_max_difference < 0.01,
        "jpeg50_practical_ceiling_across_tested_deltas": jpeg50_ceiling,
        "outcome_categories": [
            "CALIBRATION SUCCESS",
            *( ["SEVERE-JPEG CEILING"] if jpeg50_ceiling else [] ),
            *( ["FIDELITY-LIMITED"] if selected_delta > 16 else [] ),
        ],
        "paired_fidelity_comparisons": fidelity_pair_rows,
        "test_set_used": False,
    }
    config = {
        "experiment": "Stage 3A controlled QIM delta calibration",
        "deltas": [8, 16, 24, 32], "training_examples_per_delta": 7000,
        "validation_examples_per_delta": 1400,
        "training_payload_seed": TRAIN_PAYLOAD_SEED,
        "validation_payload_seed": VAL_PAYLOAD_SEED,
        "shuffle_seeds": {str(int(k)): v for k, v in SHUFFLE_SEEDS.items()},
        "coefficient_seed": COEFFICIENT_SEED, "payload_bits": PAYLOAD_BITS,
        "wavelet": "haar", "dwt_level": 2, "subbands": ["LH2", "HL2"],
        "conditions": list(CONDITIONS),
        "model": "fresh 369-parameter Stage 2B seed-aware Conv1D per delta",
        "features": ["coefficient/delta", "sin(pi*coefficient/delta)", "cos(pi*coefficient/delta)", "subband_id"],
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE, "threshold": THRESHOLD,
        "checkpoint": "lowest seven-condition validation macro BER; validation BCE tie-break",
        "fidelity": "full RGB uint8; skimage PSNR data_range=255; multichannel SSIM data_range=255 channel_axis=2",
        "test_set_used": False,
    }
    identity_verification = {
        "training_images": len(train_paths),
        "validation_images": len(val_paths),
        "training_base_pairs": len(train_payload_expected),
        "validation_base_pairs": len(val_payload_expected),
        "training_payload_fingerprints_match_run3b": True,
        "validation_payload_fingerprints_match_run3b": True,
        "validation_grid_matches_stage2b_for_all_deltas": True,
        "validation_examples_per_delta": 1400,
        "training_examples_per_delta": 7000,
        "coefficient_seed": COEFFICIENT_SEED,
        "test_set_used": False,
        "nonfinite_feature_count": 0,
    }
    write_csv(OUTPUT_DIR / "selected_training_images.csv", [
        {"selection_order": i, "image_filename": p.name} for i, p in enumerate(train_paths, 1)
    ])
    write_csv(OUTPUT_DIR / "selected_validation_images.csv", [
        {"selection_order": i, "image_filename": p.name} for i, p in enumerate(val_paths, 1)
    ])
    write_csv(OUTPUT_DIR / "payload_reproducibility_metadata.csv", all_payloads)
    write_csv(OUTPUT_DIR / "fidelity_metrics.csv", all_fidelity)
    write_csv(OUTPUT_DIR / "per_delta_summary.csv", all_delta_summaries)
    write_csv(OUTPUT_DIR / "per_condition_summary.csv", all_conditions)
    write_csv(OUTPUT_DIR / "per_sample_validation_results.csv", all_samples)
    write_csv(OUTPUT_DIR / "comparison_to_stage2b.csv", comparison_rows)
    write_csv(OUTPUT_DIR / "paired_fidelity_comparison.csv", fidelity_pair_rows)
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "data_identity_verification.json").write_text(
        json.dumps(identity_verification, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "calibration_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"per_delta": all_delta_summaries, "decision": decision}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
