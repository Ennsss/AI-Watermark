"""Non-training signal-localization diagnostic for the frozen Run 1 samples."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.embedding import _get_embedding_locations, qim_extract_bit
from watermark.extraction import compute_ber, extract_from_image
from watermark.preprocessor import load_image

from run_cnn_benchmark import embed_image, normalize_inputs


RUN1_DIR = ROOT / "experiments/run1_clean_fixed_seed_overfit"
OUTPUT_DIR = ROOT / "experiments/run1_signal_localization"
TRAIN_DIR = ROOT / "data/curated/train"
PAYLOAD_SEED = 20260808
COEFFICIENT_SEED = 42
DELTA = 16.0
PAYLOAD_BITS = 128
SELECTED_IMAGES = ["5233584.png", "5233614.png", "5233676.png", "5233692.png"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def cohens_d(group_zero: np.ndarray, group_one: np.ndarray) -> float | None:
    if group_zero.size < 2 or group_one.size < 2:
        return None
    denominator = group_zero.size + group_one.size - 2
    pooled_var = (
        (group_zero.size - 1) * group_zero.var(ddof=1)
        + (group_one.size - 1) * group_one.var(ddof=1)
    ) / denominator
    if pooled_var <= 0:
        return None
    return float((group_one.mean() - group_zero.mean()) / np.sqrt(pooled_var))


def build_exact_samples() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reproduce screening RNG consumption and retain the four Run 1 qualifiers."""
    rng = np.random.default_rng(PAYLOAD_SEED)
    tensors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict] = []
    selected = set(SELECTED_IMAGES)
    found: list[str] = []
    for path in sorted(TRAIN_DIR.glob("*.png")):
        payloads = [rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8) for _ in range(8)]
        if path.name not in selected:
            continue
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        candidate: list[tuple[np.ndarray, np.ndarray]] = []
        for payload_index, bits in enumerate(payloads):
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            extracted, _ = extract_from_image(
                watermarked,
                PAYLOAD_BITS,
                COEFFICIENT_SEED,
                DELTA,
                "haar",
                target_subbands=("lh2", "hl2"),
            )
            if compute_ber(bits, extracted) != 0.0:
                raise RuntimeError(f"Frozen sample is no longer clean-recoverable: {path.name}")
            candidate.append((prepare_cnn_input_from_image(watermarked), bits))
        found.append(path.name)
        for payload_index, (tensor, bits) in enumerate(candidate):
            tensors.append(tensor)
            targets.append(bits)
            metadata.append(
                {
                    "sample_id": len(metadata),
                    "source_image": path.name,
                    "payload_index": payload_index,
                }
            )
        if len(found) == 4:
            break
    if found != SELECTED_IMAGES:
        raise RuntimeError(f"Run 1 image mismatch: reconstructed {found}, expected {SELECTED_IMAGES}")
    return np.stack(tensors).astype(np.float32), np.stack(targets).astype(np.uint8), metadata


def location_rows(shape: tuple[int, int]) -> list[dict]:
    half = PAYLOAD_BITS // 2
    lh = _get_embedding_locations(shape, half, COEFFICIENT_SEED)
    hl = _get_embedding_locations(shape, PAYLOAD_BITS - half, COEFFICIENT_SEED + 1)
    rows = []
    for bit_index, (row, col) in enumerate(lh):
        rows.append(
            {"bit_index": bit_index, "subband": "LH2", "channel": 0, "row": int(row), "column": int(col)}
        )
    for offset, (row, col) in enumerate(hl):
        rows.append(
            {
                "bit_index": half + offset,
                "subband": "HL2",
                "channel": 1,
                "row": int(row),
                "column": int(col),
            }
        )
    return rows


def collision_summary(locations: list[dict], divisor: int, targets: np.ndarray) -> dict:
    cells: dict[tuple[int, int, int], list[int]] = {}
    for location in locations:
        cell = (
            location["channel"],
            location["row"] // divisor,
            location["column"] // divisor,
        )
        cells.setdefault(cell, []).append(location["bit_index"])
    collision_cells = {cell: bits for cell, bits in cells.items() if len(bits) > 1}
    colliding_bits = {bit for bits in collision_cells.values() for bit in bits}
    collision_pairs = [pair for bits in collision_cells.values() for pair in itertools.combinations(bits, 2)]
    differing_per_sample = [
        sum(int(target[pair[0]] != target[pair[1]]) for pair in collision_pairs)
        for target in targets
    ]
    return {
        "region_size": divisor,
        "unique_cells_occupied": len(cells),
        "collision_cells": len(collision_cells),
        "selected_bits_sharing_a_cell": len(colliding_bits),
        "collision_pairs": len(collision_pairs),
        "mean_differing_target_collision_pairs_per_sample": float(np.mean(differing_per_sample)),
        "maximum_differing_target_collision_pairs_per_sample": int(max(differing_per_sample, default=0)),
    }


def survival_group(rows: list[dict], key: str, value: str | int | None = None) -> dict:
    selected = rows if value is None else [row for row in rows if row[key] == value]
    return {
        "group": "overall" if value is None else f"{key}={value}",
        "count": len(selected),
        "survive_2x2_count": sum(row["is_max_2x2"] for row in selected),
        "survive_2x2_pct": float(100.0 * np.mean([row["is_max_2x2"] for row in selected])),
        "discarded_2x2_pct": float(100.0 * np.mean([not row["is_max_2x2"] for row in selected])),
        "mean_rank_2x2": float(np.mean([row["rank_2x2"] for row in selected])),
        "survive_4x4_count": sum(row["is_max_4x4"] for row in selected),
        "survive_4x4_pct": float(100.0 * np.mean([row["is_max_4x4"] for row in selected])),
        "discarded_4x4_pct": float(100.0 * np.mean([not row["is_max_4x4"] for row in selected])),
        "mean_rank_4x4": float(np.mean([row["rank_4x4"] for row in selected])),
    }


def activation_diagnostics(normalized: np.ndarray, metadata: list[dict]) -> list[dict] | None:
    checkpoint = RUN1_DIR / "baseline_cnn_overfit.keras"
    if not checkpoint.exists():
        return None
    try:
        from tensorflow import keras

        model = keras.models.load_model(checkpoint)
        requested = [
            layer for layer in model.layers if layer.__class__.__name__ in {"Conv2D", "MaxPooling2D"}
        ][:4]
        extractor = keras.Model(model.inputs, [layer.output for layer in requested])
        outputs = extractor.predict(normalized, batch_size=8, verbose=0)
    except Exception as exc:
        return [{"status": "unavailable", "reason": str(exc)}]
    rows: list[dict] = []
    for layer, activations in zip(requested, outputs):
        image_variances = []
        pairwise_rms = []
        for image_name in SELECTED_IMAGES:
            indices = [i for i, item in enumerate(metadata) if item["source_image"] == image_name]
            group = activations[indices].astype(np.float64)
            image_variances.append(float(np.mean(np.var(group, axis=0))))
            for left, right in itertools.combinations(range(len(group)), 2):
                pairwise_rms.append(float(np.sqrt(np.mean((group[left] - group[right]) ** 2))))
        rows.append(
            {
                "status": "ok",
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "output_shape": str(tuple(activations.shape[1:])),
                "mean_payload_activation_variance": float(np.mean(image_variances)),
                "mean_pairwise_payload_rms_distance": float(np.mean(pairwise_rms)),
                "activation_std": float(activations.std()),
                "normalized_pairwise_distance": float(
                    np.mean(pairwise_rms) / activations.std() if activations.std() > 0 else 0.0
                ),
            }
        )
    return rows


def main() -> int:
    raw_inputs, targets, metadata = build_exact_samples()
    normalized, mean, std = normalize_inputs(raw_inputs)
    with (RUN1_DIR / "experiment_config.json").open(encoding="utf-8") as handle:
        run1_config = json.load(handle)
    saved_mean = np.asarray(run1_config["normalization_mean"])
    saved_std = np.asarray(run1_config["normalization_std"])
    if not np.allclose(mean.reshape(-1), saved_mean) or not np.allclose(std.reshape(-1), saved_std):
        raise RuntimeError("Reconstructed normalization statistics do not match Run 1.")

    locations = location_rows(raw_inputs.shape[1:3])
    selected_rows: list[dict] = []
    survival_rows: list[dict] = []
    raw_classical_bits = np.empty_like(targets)
    denormalized_bits = np.empty_like(targets)
    for sample_index, (raw_map, norm_map, target, meta) in enumerate(
        zip(raw_inputs, normalized, targets, metadata)
    ):
        for location in locations:
            bit_index = location["bit_index"]
            channel = location["channel"]
            row = location["row"]
            column = location["column"]
            raw = float(raw_map[row, column, channel])
            norm = float(norm_map[row, column, channel])
            d0 = abs(raw - DELTA * np.round(raw / DELTA))
            d1 = abs(raw - (DELTA * np.round((raw - DELTA / 2) / DELTA) + DELTA / 2))
            classical = qim_extract_bit(raw, DELTA)
            restored = norm * float(std.reshape(-1)[channel]) + float(mean.reshape(-1)[channel])
            normalized_restored_bit = qim_extract_bit(restored, DELTA)
            raw_classical_bits[sample_index, bit_index] = classical
            denormalized_bits[sample_index, bit_index] = normalized_restored_bit
            selected_rows.append(
                {
                    **meta,
                    "bit_index": bit_index,
                    "target_bit": int(target[bit_index]),
                    "subband": location["subband"],
                    "channel": channel,
                    "row": row,
                    "column": column,
                    "raw_coefficient": raw,
                    "coefficient_over_delta": raw / DELTA,
                    "normalized_coefficient": norm,
                    "classical_extracted_bit": classical,
                    "normalized_then_inverted_extracted_bit": normalized_restored_bit,
                    "distance_qim_lattice_0": float(d0),
                    "distance_qim_lattice_1": float(d1),
                    "decision_margin": float(abs(d0 - d1)),
                }
            )

            row0, col0 = (row // 2) * 2, (column // 2) * 2
            cell2 = norm_map[row0 : row0 + 2, col0 : col0 + 2, channel].reshape(-1)
            row4, col4 = (row // 4) * 4, (column // 4) * 4
            cell4 = norm_map[row4 : row4 + 4, col4 : col4 + 4, channel].reshape(-1)
            rank2 = int(1 + np.sum(cell2 > norm))
            rank4 = int(1 + np.sum(cell4 > norm))
            survival_rows.append(
                {
                    **meta,
                    "bit_index": bit_index,
                    "target_bit": int(target[bit_index]),
                    "subband": location["subband"],
                    "row": row,
                    "column": column,
                    "normalized_selected_value": norm,
                    "pool1_row": row // 2,
                    "pool1_column": column // 2,
                    "pool2_row": row // 4,
                    "pool2_column": column // 4,
                    "is_max_2x2": bool(norm >= cell2.max()),
                    "rank_2x2": rank2,
                    "difference_from_2x2_max": float(norm - cell2.max()),
                    "is_max_4x4": bool(norm >= cell4.max()),
                    "rank_4x4": rank4,
                    "difference_from_4x4_max": float(norm - cell4.max()),
                }
            )

    mapping_rows = [
        {
            **location,
            "pool1_row": location["row"] // 2,
            "pool1_column": location["column"] // 2,
            "pool2_row": location["row"] // 4,
            "pool2_column": location["column"] // 4,
        }
        for location in locations
    ]
    pool1 = collision_summary(locations, 2, targets)
    pool2 = collision_summary(locations, 4, targets)

    survival_summary = [survival_group(survival_rows, "", None)]
    survival_summary.extend(survival_group(survival_rows, "subband", value) for value in ["LH2", "HL2"])
    survival_summary.extend(survival_group(survival_rows, "target_bit", value) for value in [0, 1])

    dependence_rows: list[dict] = []
    dependence_rng = np.random.default_rng(PAYLOAD_SEED)
    occupied = {(item["channel"], item["row"], item["column"]) for item in locations}
    nonselected_by_channel: dict[int, list[tuple[int, int]]] = {}
    for channel in [0, 1]:
        available = [
            (row, col)
            for row in range(128)
            for col in range(128)
            if (channel, row, col) not in occupied
        ]
        chosen = dependence_rng.choice(len(available), size=64, replace=False)
        nonselected_by_channel[channel] = [available[index] for index in chosen]

    for image_name in SELECTED_IMAGES:
        sample_indices = [i for i, item in enumerate(metadata) if item["source_image"] == image_name]
        for location in locations:
            bit_index = location["bit_index"]
            channel = location["channel"]
            local_index = bit_index if channel == 0 else bit_index - 64
            non_row, non_col = nonselected_by_channel[channel][local_index]
            local_targets = targets[sample_indices, bit_index]
            selected_values = raw_inputs[sample_indices, location["row"], location["column"], channel]
            nonselected_values = raw_inputs[sample_indices, non_row, non_col, channel]
            group0 = local_targets == 0
            group1 = local_targets == 1
            valid = bool(group0.any() and group1.any())
            dependence_rows.append(
                {
                    "source_image": image_name,
                    "bit_index": bit_index,
                    "subband": location["subband"],
                    "target_zero_count": int(group0.sum()),
                    "target_one_count": int(group1.sum()),
                    "both_classes_present": valid,
                    "selected_mean_when_zero": float(selected_values[group0].mean()) if group0.any() else None,
                    "selected_mean_when_one": float(selected_values[group1].mean()) if group1.any() else None,
                    "selected_one_minus_zero": (
                        float(selected_values[group1].mean() - selected_values[group0].mean()) if valid else None
                    ),
                    "selected_cohens_d": cohens_d(selected_values[group0], selected_values[group1]) if valid else None,
                    "nonselected_row": non_row,
                    "nonselected_column": non_col,
                    "nonselected_mean_when_zero": float(nonselected_values[group0].mean()) if group0.any() else None,
                    "nonselected_mean_when_one": float(nonselected_values[group1].mean()) if group1.any() else None,
                    "nonselected_one_minus_zero": (
                        float(nonselected_values[group1].mean() - nonselected_values[group0].mean()) if valid else None
                    ),
                    "nonselected_cohens_d": cohens_d(nonselected_values[group0], nonselected_values[group1]) if valid else None,
                }
            )

    valid_dependence = [row for row in dependence_rows if row["both_classes_present"]]
    selected_differences = np.asarray([abs(row["selected_one_minus_zero"]) for row in valid_dependence])
    nonselected_differences = np.asarray([abs(row["nonselected_one_minus_zero"]) for row in valid_dependence])
    target_zero_rows = [row for row in selected_rows if row["target_bit"] == 0]
    target_one_rows = [row for row in selected_rows if row["target_bit"] == 1]
    raw_zero = np.asarray([row["coefficient_over_delta"] for row in target_zero_rows])
    raw_one = np.asarray([row["coefficient_over_delta"] for row in target_one_rows])
    norm_zero = np.asarray([row["normalized_coefficient"] for row in target_zero_rows])
    norm_one = np.asarray([row["normalized_coefficient"] for row in target_one_rows])
    margin_zero = np.asarray([row["decision_margin"] for row in target_zero_rows])
    margin_one = np.asarray([row["decision_margin"] for row in target_one_rows])

    correlations = {}
    for channel, name in [(0, "LH2"), (1, "HL2")]:
        subset = [row for row in selected_rows if row["channel"] == channel]
        correlations[name] = float(
            np.corrcoef(
                [row["raw_coefficient"] for row in subset],
                [row["normalized_coefficient"] for row in subset],
            )[0, 1]
        )
    correlations["combined"] = float(
        np.corrcoef(
            [row["raw_coefficient"] for row in selected_rows],
            [row["normalized_coefficient"] for row in selected_rows],
        )[0, 1]
    )

    activations = activation_diagnostics(normalized, metadata)
    summary = {
        "configuration": {
            "samples": 32,
            "source_images": SELECTED_IMAGES,
            "payload_seed": PAYLOAD_SEED,
            "coefficient_seed": COEFFICIENT_SEED,
            "delta": DELTA,
            "wavelet": "haar",
            "dwt_level": 2,
            "subbands": ["LH2", "HL2"],
            "attacks": [],
        },
        "normalization": {
            "scope": "training-set channel-wise (one statistic per LH2/HL2 channel)",
            "formula": "normalized = (coefficient - channel_mean) / channel_std",
            "channel_means": mean.reshape(-1).astype(float).tolist(),
            "channel_stds": std.reshape(-1).astype(float).tolist(),
            "matches_saved_run1_statistics": True,
            "pearson_raw_vs_normalized": correlations,
            "nan_count": int(np.isnan(normalized).sum()),
            "inf_count": int(np.isinf(normalized).sum()),
            "dtype": str(normalized.dtype),
            "clipping_rounding_or_quantization": False,
        },
        "selected_signal": {
            "raw_selected_coefficient_ber": float(np.mean(raw_classical_bits != targets)),
            "normalized_then_affine_inverted_ber": float(np.mean(denormalized_bits != targets)),
            "coefficient_over_delta_target_0": class_stats(raw_zero),
            "coefficient_over_delta_target_1": class_stats(raw_one),
            "normalized_coefficient_target_0": class_stats(norm_zero),
            "normalized_coefficient_target_1": class_stats(norm_one),
            "decision_margin_target_0": class_stats(margin_zero),
            "decision_margin_target_1": class_stats(margin_one),
        },
        "pool1_collisions": pool1,
        "pool2_collisions": pool2,
        "direct_max_survival": survival_summary,
        "payload_dependence": {
            "image_bit_groups_with_both_classes": len(valid_dependence),
            "total_image_bit_groups": len(dependence_rows),
            "mean_absolute_selected_one_minus_zero": float(selected_differences.mean()),
            "median_absolute_selected_one_minus_zero": float(np.median(selected_differences)),
            "mean_absolute_nonselected_one_minus_zero": float(nonselected_differences.mean()),
            "median_absolute_nonselected_one_minus_zero": float(np.median(nonselected_differences)),
            "mean_absolute_difference_ratio_selected_to_nonselected": float(
                selected_differences.mean() / nonselected_differences.mean()
            ) if nonselected_differences.mean() > 0 else None,
        },
        "trained_model_activations": activations,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "selected_coefficients.csv", selected_rows)
    write_csv(OUTPUT_DIR / "pooling_location_map.csv", mapping_rows)
    write_csv(OUTPUT_DIR / "direct_max_survival_samples.csv", survival_rows)
    write_csv(OUTPUT_DIR / "direct_max_survival_summary.csv", survival_summary)
    write_csv(OUTPUT_DIR / "payload_dependence.csv", dependence_rows)
    if activations:
        write_csv(OUTPUT_DIR / "trained_model_activations.csv", activations)
    with (OUTPUT_DIR / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
