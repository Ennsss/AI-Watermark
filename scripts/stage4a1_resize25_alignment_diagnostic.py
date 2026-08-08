"""Stage 4A.1: non-training local alignment diagnostic for Resize25."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import extract_from_image
from watermark.preprocessor import load_image

from diagnose_run1_signal_localization import location_rows
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import write_csv
from stage3a_delta_calibration import features_from_map
from stage4a_zero_shot_resize import CHECKPOINT, resize_scale


OUTPUT_DIR = ROOT / "experiments/stage4a1_resize25_alignment_diagnostic"
STAGE3A_DIR = ROOT / "experiments/stage3a_delta_calibration"
STAGE4A_DIR = ROOT / "experiments/stage4a_zero_shot_resize"
TRAIN_DIR = ROOT / "data/curated/train"
VAL_DIR = ROOT / "data/curated/val"
TRAIN_IMAGES = 500
VAL_IMAGES = 100
PAYLOADS_PER_IMAGE = 2
TRAIN_PAYLOAD_SEED = 20260810
VAL_PAYLOAD_SEED = 20260809
PAYLOAD_BITS = 128
COEFFICIENT_SEED = 42
DELTA = 24.0
PHASE_BINS = 64
CONDITIONS = ("resize25", "resize50")
SCALES = {"resize25": 0.25, "resize50": 0.50}
OFFSETS_3 = tuple((dr, dc) for dr in range(-1, 2) for dc in range(-1, 2))
OFFSETS_5 = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3))


def extract_patch_values(coefficient_map: np.ndarray, locations: list[dict], radius: int) -> np.ndarray:
    size = radius * 2 + 1
    padded = np.pad(coefficient_map, ((radius, radius), (radius, radius), (0, 0)), mode="symmetric")
    patches = np.empty((PAYLOAD_BITS, size, size), dtype=np.float32)
    for location in locations:
        row, col, channel = location["row"], location["column"], location["channel"]
        patches[location["bit_index"]] = padded[
            row : row + size, col : col + size, channel
        ]
    return patches


def phase(coefficients: np.ndarray) -> np.ndarray:
    return np.mod(coefficients / DELTA, 1.0)


def classical_bits(coefficients: np.ndarray) -> np.ndarray:
    values = phase(coefficients)
    d0 = np.minimum(values, 1.0 - values)
    d1 = np.abs(values - 0.5)
    return (d1 < d0).astype(np.uint8)


def circular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.abs(left - right)
    return np.minimum(difference, 1.0 - difference)


def generate_split(paths, metadata_rows, payload_seed: int, split: str):
    sample_count = len(paths) * PAYLOADS_PER_IMAGE
    targets = np.empty((sample_count, PAYLOAD_BITS), dtype=np.uint8)
    clean_centers = np.empty((sample_count, PAYLOAD_BITS), dtype=np.float32)
    patches = {
        condition: np.empty((sample_count, PAYLOAD_BITS, 5, 5), dtype=np.float32)
        for condition in CONDITIONS
    }
    center_classical = {
        condition: np.empty((sample_count, PAYLOAD_BITS), dtype=np.uint8)
        for condition in CONDITIONS
    }
    identities = []
    rng = np.random.default_rng(payload_seed)
    locations = location_rows((128, 128))
    sample = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[{split}] {image_index}/{len(paths)} {path.name}", flush=True)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            fingerprint = hashlib.sha256(bits.tobytes()).hexdigest()
            expected = metadata_rows[sample]
            if path.name != expected["image_filename"] or fingerprint != expected["payload_fingerprint"]:
                raise RuntimeError(f"{split} identity differs from frozen metadata.")
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            clean_map = prepare_cnn_input_from_image(watermarked, wavelet="haar")
            clean_centers[sample] = extract_patch_values(clean_map, locations, 0)[:, 0, 0]
            targets[sample] = bits
            identities.append({
                "split": split, "sample_index": sample, "image_filename": path.name,
                "payload_index": payload_index, "payload_seed": payload_seed,
                "payload_fingerprint": fingerprint,
            })
            for condition in CONDITIONS:
                attacked = resize_scale(watermarked, SCALES[condition]).image
                coefficient_map = prepare_cnn_input_from_image(attacked, wavelet="haar")
                patches[condition][sample] = extract_patch_values(coefficient_map, locations, 2)
                center_classical[condition][sample], _ = extract_from_image(
                    attacked, PAYLOAD_BITS, COEFFICIENT_SEED, DELTA, "haar",
                    target_subbands=("lh2", "hl2"),
                )
            sample += 1
    return {
        "targets": targets, "clean_centers": clean_centers,
        "patches": patches, "center_classical": center_classical,
        "identities": identities,
    }


def values_for_offset(patches: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    dr, dc = offset
    return patches[:, :, dr + 2, dc + 2]


def overlap_metrics(values: np.ndarray, targets: np.ndarray) -> dict:
    phases = phase(values).ravel()
    bits = targets.ravel()
    edges = np.linspace(0.0, 1.0, PHASE_BINS + 1)
    probabilities = []
    for target in (0, 1):
        histogram, _ = np.histogram(phases[bits == target], bins=edges)
        probabilities.append(histogram / histogram.sum())
    p0, p1 = probabilities
    midpoint = 0.5 * (p0 + p1)
    nz0, nz1 = p0 > 0, p1 > 0
    js = 0.5 * float(np.sum(p0[nz0] * np.log2(p0[nz0] / midpoint[nz0])))
    js += 0.5 * float(np.sum(p1[nz1] * np.log2(p1[nz1] / midpoint[nz1])))
    return {
        "overlap_coefficient": float(np.minimum(p0, p1).sum()),
        "total_variation_distance": float(0.5 * np.abs(p0 - p1).sum()),
        "jensen_shannon_divergence_bits": js,
        "target0_phase_mean": float(phases[bits == 0].mean()),
        "target0_phase_std": float(phases[bits == 0].std()),
        "target1_phase_mean": float(phases[bits == 1].mean()),
        "target1_phase_std": float(phases[bits == 1].std()),
    }


def oracle_select(patches: np.ndarray, clean_centers: np.ndarray, radius: int):
    start, stop = 2 - radius, 3 + radius
    local = patches[:, :, start:stop, start:stop]
    flat = local.reshape(local.shape[0], local.shape[1], -1)
    clean_phase = phase(clean_centers)[..., None]
    phase_distance = circular_distance(phase(flat), clean_phase)
    coefficient_distance = np.abs(flat - clean_centers[..., None]) / DELTA
    score = phase_distance + coefficient_distance * 1e-7
    indices = np.argmin(score, axis=-1)
    chosen = np.take_along_axis(flat, indices[..., None], axis=-1)[..., 0]
    offsets = OFFSETS_3 if radius == 1 else OFFSETS_5
    offset_array = np.asarray(offsets, dtype=np.int8)[indices]
    return chosen, offset_array


def offset_rows(train, validation):
    rows = []
    best = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for subband, selection in (("LH2", slice(0, 64)), ("HL2", slice(64, 128)), ("combined", slice(0, 128))):
            candidates = []
            for offset in OFFSETS_3:
                train_values = values_for_offset(train["patches"][condition], offset)[:, selection]
                val_values = values_for_offset(validation["patches"][condition], offset)[:, selection]
                train_targets = train["targets"][:, selection]
                val_targets = validation["targets"][:, selection]
                train_predictions = classical_bits(train_values)
                val_predictions = classical_bits(val_values)
                metrics = overlap_metrics(val_values, val_targets)
                clean_reference = validation["clean_centers"][:, selection]
                coefficient_displacement = np.abs(val_values - clean_reference)
                phase_displacement = circular_distance(phase(val_values), phase(clean_reference))
                row = {
                    "condition": condition, "subband": subband,
                    "offset_row": offset[0], "offset_column": offset[1],
                    "training_ber": float(np.mean(train_predictions != train_targets)),
                    "validation_ber": float(np.mean(val_predictions != val_targets)),
                    "validation_false_zero_count": int(np.sum((val_predictions == 0) & (val_targets == 1))),
                    "validation_false_one_count": int(np.sum((val_predictions == 1) & (val_targets == 0))),
                    "median_absolute_displacement_from_clean_center": float(np.median(coefficient_displacement)),
                    "mean_absolute_displacement_from_clean_center": float(np.mean(coefficient_displacement)),
                    "median_circular_phase_displacement_from_clean_center": float(np.median(phase_displacement)),
                    "mean_circular_phase_displacement_from_clean_center": float(np.mean(phase_displacement)),
                    **metrics,
                }
                rows.append(row)
                candidates.append(row)
            chosen = min(candidates, key=lambda row: (row["training_ber"], abs(row["offset_row"]) + abs(row["offset_column"]), row["offset_row"], row["offset_column"]))
            best[condition][subband] = chosen
    return rows, best


def histogram_rows(condition: str, window: str, offsets: np.ndarray):
    rows = []
    for subband, selection in (("LH2", slice(0, 64)), ("HL2", slice(64, 128))):
        selected = offsets[:, selection].reshape(-1, 2)
        counts = Counter(map(tuple, selected.tolist()))
        total = len(selected)
        for offset in sorted(counts):
            rows.append({
                "condition": condition, "window": window, "subband": subband,
                "offset_row": offset[0], "offset_column": offset[1],
                "count": counts[offset], "percentage": counts[offset] / total,
            })
    return rows


def oracle_summary_rows(validation):
    result_rows, recoverability_rows, histograms = [], [], []
    oracle_values = {}
    for condition in CONDITIONS:
        patches = validation["patches"][condition]
        for radius, window in ((1, "3x3"), (2, "5x5")):
            chosen, offsets = oracle_select(patches, validation["clean_centers"], radius)
            oracle_values[(condition, window)] = chosen
            predictions = classical_bits(chosen)
            start, stop = 2 - radius, 3 + radius
            local = patches[:, :, start:stop, start:stop]
            candidate_predictions = classical_bits(local)
            target_expanded = validation["targets"][..., None, None]
            any_correct = np.any(candidate_predictions == target_expanded, axis=(-1, -2))
            for subband, selection in (("LH2", slice(0, 64)), ("HL2", slice(64, 128)), ("combined", slice(0, 128))):
                errors = predictions[:, selection] != validation["targets"][:, selection]
                selected_offsets = offsets[:, selection].reshape(-1, 2)
                distance = np.sqrt(np.sum(selected_offsets.astype(float) ** 2, axis=1))
                chebyshev = np.max(np.abs(selected_offsets), axis=1)
                axial = np.sum(np.abs(selected_offsets), axis=1) == 1
                diagonal = (np.abs(selected_offsets[:, 0]) == 1) & (np.abs(selected_offsets[:, 1]) == 1)
                result_rows.append({
                    "condition": condition, "window": window, "subband": subband,
                    "oracle_non_deployable_ber": float(errors.mean()),
                    "center_retained_percentage": float(np.mean(chebyshev == 0)),
                    "one_pixel_axial_percentage": float(np.mean(axial)),
                    "one_pixel_diagonal_percentage": float(np.mean(diagonal)),
                    "two_pixel_away_percentage": float(np.mean(chebyshev == 2)),
                    "mean_euclidean_spatial_displacement": float(distance.mean()),
                })
                recoverability_rows.append({
                    "condition": condition, "window": window, "subband": subband,
                    "target_aware_oracle_correct_evidence_rate": float(any_correct[:, selection].mean()),
                    "target_aware_oracle_unrecoverable_rate": float(1.0 - any_correct[:, selection].mean()),
                })
            histograms.extend(histogram_rows(condition, window, offsets))
    return result_rows, recoverability_rows, histograms, oracle_values


def neighborhood_separability_rows(validation):
    rows = []
    for condition in CONDITIONS:
        for window, radius in (("3x3", 1), ("5x5", 2)):
            start, stop = 2 - radius, 3 + radius
            local_phase = phase(validation["patches"][condition][:, :, start:stop, start:stop])
            d0 = np.minimum(local_phase, 1.0 - local_phase)
            d1 = np.abs(local_phase - 0.5)
            summaries = {
                "center_phase": local_phase[:, :, radius, radius],
                "minimum_distance_to_bit0_lattice": d0.min(axis=(-1, -2)),
                "minimum_distance_to_bit1_lattice": d1.min(axis=(-1, -2)),
                "neighborhood_phase_mean": local_phase.mean(axis=(-1, -2)),
                "neighborhood_phase_std": local_phase.std(axis=(-1, -2)),
                "neighborhood_phase_min": local_phase.min(axis=(-1, -2)),
                "neighborhood_phase_max": local_phase.max(axis=(-1, -2)),
            }
            for subband, selection in (("LH2", slice(0, 64)), ("HL2", slice(64, 128)), ("combined", slice(0, 128))):
                bits = validation["targets"][:, selection]
                for statistic, values in summaries.items():
                    selected_values = values[:, selection]
                    rows.append({
                        "condition": condition, "window": window, "subband": subband,
                        "statistic": statistic,
                        "target0_mean": float(selected_values[bits == 0].mean()),
                        "target0_std": float(selected_values[bits == 0].std()),
                        "target1_mean": float(selected_values[bits == 1].mean()),
                        "target1_std": float(selected_values[bits == 1].std()),
                        "standardized_mean_difference": float(
                            (selected_values[bits == 1].mean() - selected_values[bits == 0].mean())
                            / max(np.sqrt(0.5 * (selected_values[bits == 0].var() + selected_values[bits == 1].var())), 1e-12)
                        ),
                    })
    return rows


def main() -> int:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required only to reproduce the frozen CNN result.") from exc
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")

    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGES]
    val_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGES]
    stage3a_payloads = list(csv.DictReader(
        (STAGE3A_DIR / "payload_reproducibility_metadata.csv").open(encoding="utf-8")
    ))
    train_metadata = [row for row in stage3a_payloads if row["split"] == "training"]
    val_metadata = [row for row in stage3a_payloads if row["split"] == "validation"]
    train = generate_split(train_paths, train_metadata, TRAIN_PAYLOAD_SEED, "training")
    validation = generate_split(val_paths, val_metadata, VAL_PAYLOAD_SEED, "validation")

    fixed_rows, best = offset_rows(train, validation)
    oracle_rows, recoverability_rows, histogram, oracle_values = oracle_summary_rows(validation)
    separability_rows = neighborhood_separability_rows(validation)

    model = keras.models.load_model(CHECKPOINT, compile=False)
    locations = location_rows((128, 128))
    resize25_centers = values_for_offset(validation["patches"]["resize25"], (0, 0))
    # Construct the exact production four features directly from selected centers.
    scaled = resize25_centers / DELTA
    subband_ids = np.broadcast_to(np.concatenate([np.zeros(64), np.ones(64)]), scaled.shape)
    center_features = np.stack([scaled, np.sin(np.pi * scaled), np.cos(np.pi * scaled), subband_ids], axis=-1).astype(np.float32)
    probabilities = np.asarray(model.predict(center_features, batch_size=32, verbose=0))
    cnn_ber = float(np.mean((probabilities >= 0.5) != validation["targets"]))
    classical_ber = float(np.mean(validation["center_classical"]["resize25"] != validation["targets"]))
    stage4a = {
        row["condition"]: row for row in csv.DictReader(
            (STAGE4A_DIR / "per_condition_summary.csv").open(encoding="utf-8")
        )
    }
    expected_cnn = float(stage4a["resize25"]["cnn_mean_ber"])
    expected_classical = float(stage4a["resize25"]["classical_mean_ber"])
    reproducibility = {
        "resize25_classical_ber": classical_ber,
        "stage4a_resize25_classical_ber": expected_classical,
        "classical_absolute_difference": classical_ber - expected_classical,
        "resize25_cnn_ber": cnn_ber,
        "stage4a_resize25_cnn_ber": expected_cnn,
        "cnn_absolute_difference": cnn_ber - expected_cnn,
        "reproduced": bool(abs(cnn_ber - expected_cnn) < 1e-12 and abs(classical_ber - expected_classical) < 1e-12),
    }
    if not reproducibility["reproduced"]:
        raise RuntimeError(f"Resize25 reproduction failed: {reproducibility}")

    best_json = {}
    phase_rows = []
    control_rows = []
    for condition in CONDITIONS:
        best_json[condition] = {}
        composite_best = np.empty_like(validation["targets"], dtype=np.float32)
        for subband, selection in (("LH2", slice(0, 64)), ("HL2", slice(64, 128))):
            chosen = best[condition][subband]
            offset = (int(chosen["offset_row"]), int(chosen["offset_column"]))
            composite_best[:, selection] = values_for_offset(validation["patches"][condition], offset)[:, selection]
            best_json[condition][subband] = {
                "selected_using": "training BER only",
                "offset": list(offset),
                "training_ber": chosen["training_ber"],
                "validation_ber": chosen["validation_ber"],
            }
        center = values_for_offset(validation["patches"][condition], (0, 0))
        oracle3 = oracle_values[(condition, "3x3")]
        for label, values in (("expected_center", center), ("best_fixed_offset_by_subband", composite_best), ("oracle_clean_reference_3x3", oracle3)):
            phase_rows.append({
                "condition": condition, "source": label,
                **overlap_metrics(values, validation["targets"]),
            })
        combined_best_ber = float(np.mean(classical_bits(composite_best) != validation["targets"]))
        oracle3_ber = float(np.mean(classical_bits(oracle3) != validation["targets"]))
        control_rows.append({
            "condition": condition,
            "center_classical_ber": float(np.mean(classical_bits(center) != validation["targets"])),
            "best_fixed_3x3_validation_ber": combined_best_ber,
            "oracle_clean_reference_3x3_ber": oracle3_ber,
            "center_phase_overlap": phase_rows[-3]["overlap_coefficient"],
            "best_fixed_phase_overlap": phase_rows[-2]["overlap_coefficient"],
            "oracle_phase_overlap": phase_rows[-1]["overlap_coefficient"],
        })

    resize25_oracle3 = next(row for row in oracle_rows if row["condition"] == "resize25" and row["window"] == "3x3" and row["subband"] == "combined")
    resize25_oracle5 = next(row for row in oracle_rows if row["condition"] == "resize25" and row["window"] == "5x5" and row["subband"] == "combined")
    resize25_fixed = next(row for row in control_rows if row["condition"] == "resize25")
    fixed_gain = classical_ber - resize25_fixed["best_fixed_3x3_validation_ber"]
    oracle_gain = classical_ber - resize25_oracle3["oracle_non_deployable_ber"]
    histogram_3 = [row for row in histogram if row["condition"] == "resize25" and row["window"] == "3x3"]
    dominant_frequency = max(row["percentage"] for row in histogram_3)
    if fixed_gain >= 0.05 and dominant_frequency >= 0.30:
        category = "A. SYSTEMATIC LOCAL SHIFT"
        stage4b = False
        decision = "Test a synchronization/location correction rather than ordinary resize-aware coefficient training."
    elif oracle_gain >= 0.10 and resize25_oracle3["oracle_non_deployable_ber"] < 0.30 and fixed_gain < 0.05:
        category = "B. LOCALLY RECOVERABLE BUT NON-SYSTEMATIC"
        stage4b = True
        decision = "One controlled local-context resize-aware training experiment is justified."
    elif oracle_gain > 0.02 and resize25_oracle3["oracle_non_deployable_ber"] < 0.45:
        category = "C. WEAK LOCAL INFORMATION"
        stage4b = False
        decision = "Local evidence is weak; ordinary resize-aware training has limited expected value."
    else:
        category = "D. LOCAL INFORMATION LOSS"
        stage4b = False
        decision = "Do not train Stage 4B; accept Resize25 as a severe limitation and proceed later to crop evaluation."

    verification = {
        "training_images": len(train_paths), "training_pairs": len(train["targets"]),
        "validation_images": len(val_paths), "validation_pairs": len(validation["targets"]),
        "payload_fingerprints_match_stage3a": True,
        "resize_implementation_matches_stage4a": True,
        "coefficient_seed": COEFFICIENT_SEED, "delta": DELTA,
        "locations": len(locations), "lh2_locations": 64, "hl2_locations": 64,
        "symmetric_padding": True,
        "nonfinite_count": int(sum(np.isnan(array).sum() + np.isinf(array).sum() for split in (train, validation) for array in split["patches"].values())),
        "test_set_used": False, "training_performed": False,
    }
    summary = {
        "resize25_classification": category,
        "stage4b_resize_aware_training_justified": stage4b,
        "decision": decision,
        "center_classical_ber": classical_ber,
        "best_fixed_offset_validation_ber": resize25_fixed["best_fixed_3x3_validation_ber"],
        "oracle_clean_reference_3x3_ber": resize25_oracle3["oracle_non_deployable_ber"],
        "oracle_clean_reference_5x5_ber": resize25_oracle5["oracle_non_deployable_ber"],
        "largest_oracle_3x3_offset_frequency": dominant_frequency,
        "systematic_shift_detected": category.startswith("A."),
        "training_performed": False, "test_set_used": False,
    }
    config = {
        "experiment": "Stage 4A.1 non-training Resize25 local alignment/information diagnostic",
        "delta": DELTA, "coefficient_seed": COEFFICIENT_SEED,
        "payload_bits": PAYLOAD_BITS, "wavelet": "haar", "dwt_level": 2,
        "subbands": ["LH2", "HL2"], "padding": "numpy symmetric",
        "windows": ["3x3", "5x5"], "phase": "(coefficient/delta) mod 1",
        "phase_histogram_bins": PHASE_BINS,
        "best_fixed_offset_selection": "minimum training classical-style BER separately for LH2 and HL2",
        "oracle_selection": "minimum circular phase distance to clean watermarked center; coefficient distance 1e-7 tie-break",
        "resize": "existing resize_scale: 25%/50% INTER_AREA down, INTER_LANCZOS4 restore",
        "training_performed": False, "test_set_used": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "fixed_offset_results.csv", fixed_rows)
    write_csv(OUTPUT_DIR / "oracle_neighbor_results.csv", oracle_rows)
    write_csv(OUTPUT_DIR / "oracle_recoverability_summary.csv", recoverability_rows)
    write_csv(OUTPUT_DIR / "offset_histogram.csv", histogram)
    write_csv(OUTPUT_DIR / "phase_overlap_comparison.csv", phase_rows)
    write_csv(OUTPUT_DIR / "resize50_control_summary.csv", [row for row in control_rows if row["condition"] == "resize50"])
    write_csv(OUTPUT_DIR / "neighborhood_separability.csv", separability_rows)
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "data_identity_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "center_reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "best_global_offset.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"reproducibility": reproducibility, "best": best_json, "control": control_rows, "oracle": oracle_rows, "recoverability": recoverability_rows, "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
