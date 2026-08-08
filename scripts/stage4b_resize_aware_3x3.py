"""Stage 4B: one controlled resize-aware shared 3x3 decoder experiment."""

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
from stage2_jpeg_reencode_training import write_csv
from stage4a_zero_shot_resize import resize_scale


OUTPUT_DIR = ROOT / "experiments/stage4b_resize_aware_3x3"
RUN3B_DIR = ROOT / "experiments/run3b_clean_training"
STAGE4A_DIR = ROOT / "experiments/stage4a_zero_shot_resize"
TRAIN_DIR = ROOT / "data/curated/train"
VAL_DIR = ROOT / "data/curated/val"
CONDITIONS = ("clean", "resize75", "resize50", "resize25")
SCALES = {"resize75": 0.75, "resize50": 0.50, "resize25": 0.25}
STAGE4A_CNN = {"clean": 0.0001953125, "resize75": 0.094609375,
               "resize50": 0.1133984375, "resize25": 0.50515625}
STAGE4A_CLASSICAL = {"clean": 0.1278515625, "resize75": 0.2253515625,
                     "resize50": 0.2355078125, "resize25": 0.5066015625}
TRAIN_IMAGES, VAL_IMAGES = 500, 100
PAYLOADS_PER_IMAGE, PAYLOAD_BITS = 2, 128
TRAIN_PAYLOAD_SEED, VAL_PAYLOAD_SEED = 20260810, 20260809
COEFFICIENT_SEED, SHUFFLE_SEED, MODEL_SEED = 42, 20260830, 20260830
DELTA, THRESHOLD = 24.0, 0.5
BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE = 32, 100, 10, 0.001


def apply_condition(image: np.ndarray, condition: str) -> np.ndarray:
    return image if condition == "clean" else resize_scale(image, SCALES[condition]).image


def build_model():
    from tensorflow import keras
    patches = keras.layers.Input((PAYLOAD_BITS, 3, 3, 3), name="local_patch_features")
    subbands = keras.layers.Input((PAYLOAD_BITS, 1), name="subband_id")
    x = keras.layers.TimeDistributed(
        keras.layers.Conv2D(16, (3, 3), padding="valid", activation="relu"),
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
    output = keras.layers.Reshape((PAYLOAD_BITS,), name="payload_probabilities")(x)
    model = keras.Model([patches, subbands], output, name="stage4b_resize_aware_3x3")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy")
    return model


def patch_features(coefficient_map: np.ndarray, locations: list[dict]) -> np.ndarray:
    padded = np.pad(coefficient_map, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
    result = np.empty((PAYLOAD_BITS, 3, 3, 3), dtype=np.float32)
    for item in locations:
        bit, row, col, channel = (item["bit_index"], item["row"],
                                  item["column"], item["channel"])
        scaled = padded[row:row + 3, col:col + 3, channel] / DELTA
        result[bit, ..., 0] = scaled
        result[bit, ..., 1] = np.sin(np.pi * scaled)
        result[bit, ..., 2] = np.cos(np.pi * scaled)
    return result


def subband_tensor(count: int) -> np.ndarray:
    ids = np.r_[np.zeros(64), np.ones(64)].astype(np.float32)
    return np.broadcast_to(ids[None, :, None], (count, 128, 1)).copy()


def representation_samples(cmap, features, target, locations, sample_id, condition):
    padded = np.pad(cmap, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
    rows = []
    for item in locations[:4] + locations[62:66] + locations[-4:]:
        bit, row, col, channel = (item["bit_index"], item["row"],
                                  item["column"], item["channel"])
        raw = padded[row:row + 3, col:col + 3, channel]
        rows.append({
            "sample_id": sample_id, "condition": condition, "bit_index": bit,
            "target_bit": int(target[bit]), "subband": item["subband"],
            "center_row": row, "center_column": col,
            "raw_3x3_coefficients": json.dumps(raw.astype(float).tolist()),
            "derived_3x3x3_features": json.dumps(features[bit].astype(float).tolist()),
        })
    return rows


def generate(paths, expected, seed: int, split: str):
    total = len(paths) * PAYLOADS_PER_IMAGE * len(CONDITIONS)
    x = np.empty((total, 128, 3, 3, 3), np.float32)
    y = np.empty((total, 128), np.uint8)
    classical = np.empty_like(y) if split == "validation" else None
    cids = np.empty(total, np.int8)
    rows, samples, row_id, base_pair = [], [], 0, 0
    rng, locations = np.random.default_rng(seed), location_rows((128, 128))
    for image_index, path in enumerate(paths, 1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[{split}] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, 128, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            if fingerprint != expected[base_pair]:
                raise RuntimeError(f"{split} payload fingerprint mismatch at pair {base_pair}")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            for cid, condition in enumerate(CONDITIONS):
                attacked = apply_condition(watermarked, condition)
                cmap = prepare_cnn_input_from_image(attacked, wavelet="haar")
                x[row_id], y[row_id], cids[row_id] = patch_features(cmap, locations), bits, cid
                if classical is not None:
                    classical[row_id], _ = extract_from_image(
                        attacked, 128, COEFFICIENT_SEED, DELTA, "haar",
                        target_subbands=("lh2", "hl2"),
                    )
                rows.append({
                    "row_id": row_id, "base_pair_index": base_pair,
                    "image_filename": path.name, "payload_index": payload_index,
                    "payload_fingerprint": fingerprint, "condition": condition,
                })
                if base_pair == 0:
                    samples += representation_samples(cmap, x[row_id], bits, locations,
                                                      row_id, condition)
                row_id += 1
            base_pair += 1
    return x, y, classical, cids, rows, samples


def basic_metrics(prob, target):
    pred = prob >= THRESHOLD
    err = pred != target
    return {"ber": float(err.mean()), "lh2_ber": float(err[:, :64].mean()),
            "hl2_ber": float(err[:, 64:].mean())}


def probability_row(condition, prob, target, pred):
    err = pred != target
    confidence = np.abs(prob - 0.5) * 2
    row = {"condition": condition, **{f"probability_{k}": v
           for k, v in distribution(prob.ravel()).items()},
           "fraction_probability_0.45_to_0.55": float(np.mean((prob >= .45) & (prob <= .55))),
           "correct_mean_confidence": float(confidence[~err].mean()),
           "incorrect_mean_confidence": float(confidence[err].mean())}
    return row


def main() -> int:
    import tensorflow as tf
    from tensorflow import keras
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGES]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGES]
    train_payloads = list(csv.DictReader((RUN3B_DIR / "training_payloads.csv").open(encoding="utf-8")))
    val_payloads = list(csv.DictReader((RUN3B_DIR / "validation_payloads.csv").open(encoding="utf-8")))
    stage4a_samples = list(csv.DictReader((STAGE4A_DIR / "per_sample_results.csv").open(encoding="utf-8")))
    train_x, train_y, _, train_cids, train_rows, samples = generate(
        train_paths, [r["payload_fingerprint"] for r in train_payloads], TRAIN_PAYLOAD_SEED, "training")
    val_x, val_y, val_classical, val_cids, val_rows, val_samples = generate(
        val_paths, [r["payload_fingerprint"] for r in val_payloads], VAL_PAYLOAD_SEED, "validation")
    samples += val_samples
    if train_x.shape != (4000, 128, 3, 3, 3) or val_x.shape != (800, 128, 3, 3, 3):
        raise RuntimeError("Unexpected representation shape")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("Non-finite representation")
    if len(stage4a_samples) != len(val_rows):
        raise RuntimeError("Stage 4A validation row count mismatch")
    for current, previous in zip(val_rows, stage4a_samples):
        for key in ("image_filename", "payload_index", "payload_fingerprint", "condition"):
            if str(current[key]) != str(previous[key]):
                raise RuntimeError(f"Stage 4A identity mismatch: {key}")
    classical_reproduction = {}
    for cid, condition in enumerate(CONDITIONS):
        mask = val_cids == cid
        classical_reproduction[condition] = float((val_classical[mask] != val_y[mask]).mean())
        if abs(classical_reproduction[condition] - STAGE4A_CLASSICAL[condition]) > 1e-12:
            raise RuntimeError(f"Classical {condition} did not reproduce Stage 4A")
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(len(train_x))
    for shuffled, original in enumerate(permutation):
        train_rows[original]["shuffled_row"] = shuffled
    train_x, train_y, train_cids = train_x[permutation], train_y[permutation], train_cids[permutation]
    train_subbands, val_subbands = subband_tensor(len(train_x)), subband_tensor(len(val_x))
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = build_model()
    if model.count_params() != 601:
        raise RuntimeError(f"Expected 601 parameters, got {model.count_params()}")
    history_rows = []

    class MacroCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__(); self.best_macro = np.inf; self.best_bce = np.inf
            self.best_epoch = 0; self.best_weights = None; self.stale = 0
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            tp = self.model.predict([train_x, train_subbands], batch_size=BATCH_SIZE, verbose=0)
            vp = self.model.predict([val_x, val_subbands], batch_size=BATCH_SIZE, verbose=0)
            tm = basic_metrics(tp, train_y)
            per = {c: basic_metrics(vp[val_cids == i], val_y[val_cids == i])
                   for i, c in enumerate(CONDITIONS)}
            macro, bce = float(np.mean([per[c]["ber"] for c in CONDITIONS])), float(logs["val_loss"])
            improved = macro < self.best_macro - 1e-12
            tied = abs(macro - self.best_macro) <= 1e-12 and bce < self.best_bce
            self.stale = 0 if improved else self.stale + 1
            if improved or tied:
                self.best_macro, self.best_bce, self.best_epoch = macro, bce, epoch + 1
                self.best_weights = self.model.get_weights()
            row = {"epoch": epoch + 1, "train_bce": float(logs["loss"]),
                   "train_ber": tm["ber"], "train_lh2_ber": tm["lh2_ber"],
                   "train_hl2_ber": tm["hl2_ber"], "validation_bce": bce,
                   "validation_macro_ber": macro, "is_best_checkpoint": self.best_epoch == epoch + 1}
            for c in CONDITIONS:
                for key, value in per[c].items(): row[f"validation_{c}_{key}"] = value
            history_rows.append(row)
            print(f"epoch={epoch+1} train={tm['ber']:.6f} macro={macro:.6f} " +
                  " ".join(f"{c}={per[c]['ber']:.6f}" for c in CONDITIONS), flush=True)
            if self.stale >= PATIENCE: self.model.stop_training = True
        def on_train_end(self, logs=None):
            if self.best_weights is None: raise RuntimeError("No checkpoint captured")
            self.model.set_weights(self.best_weights)

    callback = MacroCheckpoint()
    model.fit([train_x, train_subbands], train_y.astype(np.float32),
              validation_data=([val_x, val_subbands], val_y.astype(np.float32)),
              batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, shuffle=False,
              callbacks=[callback], verbose=0)
    prob = np.asarray(model.predict([val_x, val_subbands], batch_size=BATCH_SIZE, verbose=0))
    pred, errors = (prob >= THRESHOLD).astype(np.uint8), None
    errors = pred != val_y
    classical_errors = val_classical != val_y
    sample_ber, classical_sample_ber = errors.mean(1), classical_errors.mean(1)
    stage4a_sample_ber = np.asarray([float(r["cnn_ber"]) for r in stage4a_samples])
    condition_rows, probability_rows, bit_rows, comparison_rows = [], [], [], []
    for cid, condition in enumerate(CONDITIONS):
        mask = val_cids == cid; e, p, t, pr = errors[mask], pred[mask], val_y[mask], prob[mask]
        sb, cb, z = sample_ber[mask], classical_sample_ber[mask], stage4a_sample_ber[mask]
        stage4b_ber = float(e.mean()); absolute = STAGE4A_CNN[condition] - stage4b_ber
        row = {"condition": condition, "classical_ber": float(classical_errors[mask].mean()),
               "stage4a_zero_shot_cnn_ber": STAGE4A_CNN[condition], "stage4b_cnn_ber": stage4b_ber,
               "absolute_improvement_over_stage4a": absolute,
               "relative_improvement_over_stage4a": absolute / STAGE4A_CNN[condition],
               "classical_perfect_rate": float(np.mean(cb == 0)), "stage4b_perfect_rate": float(np.mean(sb == 0)),
               "stage4b_lh2_ber": float(e[:, :64].mean()), "stage4b_hl2_ber": float(e[:, 64:].mean()),
               "target_one_frequency": float(t.mean()), "predicted_one_frequency": float(p.mean()),
               "false_zero_count": int(np.sum((t == 1) & (p == 0))),
               "false_one_count": int(np.sum((t == 0) & (p == 1))),
               "always_zero_positions": int(np.sum(np.all(p == 0, axis=0))),
               "always_one_positions": int(np.sum(np.all(p == 1, axis=0))),
               "zero_ber_output_positions": int(np.sum(np.all(e == 0, axis=0))),
               "stage4b_lower_than_stage4a_samples": int(np.sum(sb < z)),
               "stage4b_equal_stage4a_samples": int(np.sum(sb == z)),
               "stage4b_higher_than_stage4a_samples": int(np.sum(sb > z)),
               "stage4b_lower_than_classical_samples": int(np.sum(sb < cb)),
               "stage4b_equal_classical_samples": int(np.sum(sb == cb)),
               "stage4b_higher_than_classical_samples": int(np.sum(sb > cb))}
        condition_rows.append(row); probability_rows.append(probability_row(condition, pr, t, p))
        comparison_rows.append({k: row[k] for k in ("condition", "classical_ber", "stage4a_zero_shot_cnn_ber",
            "stage4b_cnn_ber", "absolute_improvement_over_stage4a", "relative_improvement_over_stage4a")})
        for bit in range(128):
            be, bp, bt = e[:, bit], p[:, bit], t[:, bit]
            bit_rows.append({"condition": condition, "bit_index": bit,
                "subband": "LH2" if bit < 64 else "HL2", "ber": float(be.mean()),
                "target_one_frequency": float(bt.mean()), "predicted_one_frequency": float(bp.mean()),
                "false_zero_count": int(np.sum((bt == 1) & (bp == 0))),
                "false_one_count": int(np.sum((bt == 0) & (bp == 1)))})
    for i, row in enumerate(val_rows):
        row.update({"classical_ber": float(classical_sample_ber[i]),
                    "stage4a_cnn_ber": float(stage4a_sample_ber[i]),
                    "stage4b_cnn_ber": float(sample_ber[i]),
                    "stage4b_mean_probability": float(prob[i].mean())})
    ablation_rows = []
    masked = val_x.copy()
    keep = masked[:, :, 1, 1, :].copy()
    masked.fill(0)
    masked[:, :, 1, 1, :] = keep
    masked_prob = model.predict([masked, val_subbands], batch_size=BATCH_SIZE, verbose=0)
    for condition in ("resize50", "resize25"):
        mask = val_cids == CONDITIONS.index(condition)
        ablation_rows.append({"condition": condition, "full_3x3_ber": float(errors[mask].mean()),
                              "center_only_masked_ber": float(((masked_prob[mask] >= .5) != val_y[mask]).mean())})
    values = {r["condition"]: r for r in condition_rows}
    stage4b_macro = float(np.mean([r["stage4b_cnn_ber"] for r in condition_rows]))
    stage4a_macro = float(np.mean(list(STAGE4A_CNN.values())))
    classical_macro = float(np.mean(list(STAGE4A_CLASSICAL.values())))
    r25_gain = STAGE4A_CNN["resize25"] - values["resize25"]["stage4b_cnn_ber"]
    clean_ok = values["clean"]["stage4b_cnn_ber"] < .005
    mask_worse = ablation_rows[1]["center_only_masked_ber"] - ablation_rows[1]["full_3x3_ber"]
    if clean_ok and values["resize25"]["stage4b_cnn_ber"] < .30 and mask_worse >= .05:
        classification = "STRONG PASS"
    elif clean_ok and r25_gain >= .10 and stage4b_macro < stage4a_macro and mask_worse > 0:
        classification = "PASS"
    elif r25_gain > 0 or stage4b_macro < stage4a_macro:
        classification = "PARTIAL"
    else: classification = "FAIL"
    summary = {"best_epoch": callback.best_epoch, "epochs_trained": len(history_rows),
        "best_validation_bce": callback.best_bce, "parameter_count": model.count_params(),
        "stage4a_macro_ber": stage4a_macro, "stage4b_macro_ber": stage4b_macro,
        "classical_macro_ber": classical_macro,
        "stage4b_absolute_improvement_over_stage4a_macro": stage4a_macro-stage4b_macro,
        "stage4b_relative_improvement_over_stage4a_macro": (stage4a_macro-stage4b_macro)/stage4a_macro,
        "stage4b_absolute_improvement_over_classical_macro": classical_macro-stage4b_macro,
        "stage4b_relative_improvement_over_classical_macro": (classical_macro-stage4b_macro)/classical_macro,
        "resize25_absolute_improvement": r25_gain,
        "resize25_relative_improvement": r25_gain/STAGE4A_CNN["resize25"],
        "classification": classification, "test_set_used": False}
    config = {"experiment": "Stage 4B controlled resize-aware 3x3 training", "conditions": list(CONDITIONS),
        "condition_examples": {c: 1000 for c in CONDITIONS}, "validation_condition_examples": {c: 200 for c in CONDITIONS},
        "delta": DELTA, "coefficient_seed": COEFFICIENT_SEED, "payload_bits": 128,
        "features": ["coefficient/24", "sin(pi*coefficient/24)", "cos(pi*coefficient/24)"],
        "subband_id": {"LH2": 0, "HL2": 1}, "padding": "symmetric", "shuffle_seed": SHUFFLE_SEED,
        "model_seed": MODEL_SEED, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "threshold": .5, "checkpoint": "lowest four-condition validation macro BER; BCE tie-break"}
    identity = {"training_images": 500, "training_base_pairs": 1000, "validation_images": 100,
        "validation_base_pairs": 200, "training_payload_fingerprints_match_run3b": True,
        "validation_payload_fingerprints_match_run3b": True, "validation_grid_matches_stage4a": True,
        "classical_ber_reproduction": classical_reproduction, "test_set_used": False}
    verification = {"training_patch_shape": list(train_x.shape), "validation_patch_shape": list(val_x.shape),
        "training_subband_shape": list(train_subbands.shape), "validation_subband_shape": list(val_subbands.shape),
        "payload_positions": 128, "lh2_centers": 64, "hl2_centers": 64, "same_subband_patch_only": True,
        "symmetric_padding": True, "nan_count": 0, "inf_count": 0, "model_parameters": model.count_params()}
    write_csv(OUTPUT_DIR/"representation_samples.csv", samples)
    write_csv(OUTPUT_DIR/"condition_counts.csv", [{"split": s, "condition": c, "count": n}
        for s, n in (("training",1000),("validation",200)) for c in CONDITIONS])
    write_csv(OUTPUT_DIR/"training_history.csv", history_rows)
    write_csv(OUTPUT_DIR/"per_condition_summary.csv", condition_rows)
    write_csv(OUTPUT_DIR/"per_sample_validation_results.csv", val_rows)
    write_csv(OUTPUT_DIR/"per_bit_diagnostics.csv", bit_rows)
    write_csv(OUTPUT_DIR/"probability_diagnostics.csv", probability_rows)
    write_csv(OUTPUT_DIR/"center_mask_ablation.csv", ablation_rows)
    write_csv(OUTPUT_DIR/"comparison_stage4a_stage4b.csv", comparison_rows)
    for name, value in (("experiment_config",config),("data_identity_verification",identity),
                        ("representation_verification",verification),("summary_metrics",summary)):
        (OUTPUT_DIR/f"{name}.json").write_text(json.dumps(value,indent=2),encoding="utf-8")
    model.save(OUTPUT_DIR/"best_resize_aware_3x3.keras")
    print(json.dumps({"conditions": condition_rows, "ablation": ablation_rows, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
