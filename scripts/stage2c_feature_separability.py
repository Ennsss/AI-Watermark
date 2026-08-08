"""Stage 2C: non-training feature separability diagnostic.

This script does not train or modify a CNN.  It regenerates the exact Stage 2B
train/validation identities and attacks, then analyzes coefficient-level
separability using fixed empirical lookup and k-nearest-neighbor rules.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.embedding import dwt2_decompose, qim_extract_bit
from watermark.preprocessor import (
    extract_y_channel,
    load_image,
    pad_to_multiple,
    rgb_to_ycbcr,
)

from diagnose_run1_signal_localization import location_rows
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import (
    COEFFICIENT_SEED,
    DELTA,
    PAYLOAD_BITS,
    PAYLOADS_PER_IMAGE,
    TRAIN_DIR,
    TRAIN_IMAGE_COUNT,
    TRAIN_PAYLOAD_SEED,
    VAL_DIR,
    VAL_IMAGE_COUNT,
    VAL_PAYLOAD_SEED,
    apply_condition,
    write_csv,
)


OUTPUT_DIR = ROOT / "experiments/stage2c_feature_separability"
STAGE2B_DIR = ROOT / "experiments/stage2b_full_factorial_jpeg_reencode"
CONDITIONS = ("clean", "jpeg90", "jpeg70", "jpeg50", "reencode1", "reencode3")
PHASE_BINS = 64
RAW_QUANTILE_BINS = 32
KNN_K = 31
ROUND_DECIMALS = (2, 3, 4, 6)
SUBBANDS = ("LH2", "HL2", "combined")


def selected_coefficients(image: np.ndarray, locations: list[dict]) -> np.ndarray:
    y = extract_y_channel(rgb_to_ycbcr(image))
    y_padded, _ = pad_to_multiple(y, multiple=4)
    lh2, hl2, _hh2 = dwt2_decompose(y_padded, wavelet="haar", level=2)[1]
    coefficient_map = np.stack([lh2, hl2], axis=-1)
    return np.asarray(
        [
            coefficient_map[row["row"], row["column"], row["channel"]]
            for row in locations
        ],
        dtype=np.float64,
    )


def feature_matrix(coefficients: np.ndarray) -> np.ndarray:
    scaled = coefficients / DELTA
    subband = np.broadcast_to(
        np.concatenate([np.zeros(64), np.ones(64)]), scaled.shape
    )
    return np.stack(
        [scaled, np.sin(np.pi * scaled), np.cos(np.pi * scaled), subband], axis=-1
    ).astype(np.float32)


def phase_values(coefficients: np.ndarray) -> np.ndarray:
    scaled = coefficients.astype(np.float32) / np.float32(DELTA)
    return np.mod(scaled, np.float32(1.0))


def grid_distances(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = coefficients / DELTA
    d0 = np.abs(scaled - np.round(scaled))
    d1 = np.abs(scaled - (np.round(scaled - 0.5) + 0.5))
    return d0, d1


def circular_phase_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.abs(left - right)
    return np.minimum(difference, 1.0 - difference)


def load_split(
    paths: list[Path], payload_seed: int, locations: list[dict], split: str
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(payload_seed)
    coefficients: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    clean_coefficients: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    targets: list[np.ndarray] = []
    pair_ids: list[int] = []
    pair_id = 0
    for image_index, path in enumerate(paths, start=1):
        image = resize_square(center_crop_square(load_image(path)), TARGET_SIZE)
        if image_index == 1 or image_index % 25 == 0 or image_index == len(paths):
            print(f"[{split}] {image_index}/{len(paths)} {path.name}", flush=True)
        for _payload_index in range(PAYLOADS_PER_IMAGE):
            bits = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
            watermarked = embed_image(image, bits, DELTA, "haar", COEFFICIENT_SEED)
            clean = selected_coefficients(watermarked, locations)
            for condition in CONDITIONS:
                attacked = apply_condition(watermarked, condition)
                coefficients[condition].append(selected_coefficients(attacked, locations))
                clean_coefficients[condition].append(clean)
            targets.append(bits)
            pair_ids.append(pair_id)
            pair_id += 1
    return {
        "targets": np.stack(targets).astype(np.uint8),
        "pair_ids": np.asarray(pair_ids, dtype=np.int32),
        **{
            f"coeff_{condition}": np.stack(values).astype(np.float64)
            for condition, values in coefficients.items()
        },
        **{
            f"clean_{condition}": np.stack(values).astype(np.float64)
            for condition, values in clean_coefficients.items()
        },
    }


def subband_slice(name: str) -> slice:
    if name == "LH2":
        return slice(0, 64)
    if name == "HL2":
        return slice(64, 128)
    return slice(0, 128)


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


def distribution_rows(split: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    targets = split["targets"]
    for condition in CONDITIONS:
        coefficients = split[f"coeff_{condition}"]
        features = feature_matrix(coefficients)
        phase = phase_values(coefficients)
        d0, d1 = grid_distances(coefficients)
        margin = np.abs(d0 - d1)
        variables = {
            "coefficient_over_delta": features[..., 0],
            "sin_pi_c_over_delta": features[..., 1],
            "cos_pi_c_over_delta": features[..., 2],
            "qim_phase_mod_1": phase,
            "classical_decision_margin_over_delta": margin,
        }
        for subband in ("LH2", "HL2"):
            selection = subband_slice(subband)
            for target_bit in (0, 1):
                target_mask = targets[:, selection] == target_bit
                for variable, values in variables.items():
                    stats = describe(values[:, selection][target_mask])
                    rows.append(
                        {
                            "condition": condition,
                            "subband": subband,
                            "target_bit": target_bit,
                            "variable": variable,
                            **stats,
                        }
                    )
    return rows


def overlap_rows(train: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, PHASE_BINS + 1)
    targets = train["targets"]
    for condition in CONDITIONS:
        phase = phase_values(train[f"coeff_{condition}"])
        for subband in ("LH2", "HL2", "combined"):
            selection = subband_slice(subband)
            selected_phase = phase[:, selection].ravel()
            selected_targets = targets[:, selection].ravel()
            probabilities = []
            counts = []
            for target_bit in (0, 1):
                histogram, _ = np.histogram(
                    selected_phase[selected_targets == target_bit], bins=edges
                )
                counts.append(int(histogram.sum()))
                probabilities.append(histogram / histogram.sum())
            p0, p1 = probabilities
            overlap = float(np.minimum(p0, p1).sum())
            total_variation = float(0.5 * np.abs(p0 - p1).sum())
            bhattacharyya = float(np.sqrt(p0 * p1).sum())
            midpoint = 0.5 * (p0 + p1)
            nonzero0 = p0 > 0
            nonzero1 = p1 > 0
            js = 0.5 * float(np.sum(p0[nonzero0] * np.log2(p0[nonzero0] / midpoint[nonzero0])))
            js += 0.5 * float(np.sum(p1[nonzero1] * np.log2(p1[nonzero1] / midpoint[nonzero1])))
            rows.append(
                {
                    "condition": condition,
                    "subband": subband,
                    "phase_bins": PHASE_BINS,
                    "target_0_count": counts[0],
                    "target_1_count": counts[1],
                    "histogram_overlap_coefficient": overlap,
                    "total_variation_distance": total_variation,
                    "bhattacharyya_coefficient": bhattacharyya,
                    "jensen_shannon_divergence_bits": js,
                }
            )
    return rows


def lookup_predict(
    train_coefficients: np.ndarray,
    train_targets: np.ndarray,
    validation_coefficients: np.ndarray,
    subband: str,
) -> np.ndarray:
    selection = subband_slice(subband)
    train_scaled = (train_coefficients[:, selection] / DELTA).ravel()
    validation_scaled = (validation_coefficients[:, selection] / DELTA).ravel()
    train_bits = train_targets[:, selection].ravel()
    quantiles = np.linspace(0.0, 1.0, RAW_QUANTILE_BINS + 1)[1:-1]
    raw_edges = np.unique(np.quantile(train_scaled, quantiles))
    train_raw_bin = np.searchsorted(raw_edges, train_scaled, side="right")
    validation_raw_bin = np.searchsorted(raw_edges, validation_scaled, side="right")
    train_phase_bin = np.minimum((np.mod(train_scaled, 1.0) * PHASE_BINS).astype(int), PHASE_BINS - 1)
    validation_phase_bin = np.minimum((np.mod(validation_scaled, 1.0) * PHASE_BINS).astype(int), PHASE_BINS - 1)
    raw_count = len(raw_edges) + 1
    train_key = train_raw_bin * PHASE_BINS + train_phase_bin
    validation_key = validation_raw_bin * PHASE_BINS + validation_phase_bin
    if subband == "combined":
        train_sb = np.broadcast_to(np.concatenate([np.zeros(64, dtype=int), np.ones(64, dtype=int)]), train_targets.shape).ravel()
        validation_sb = np.broadcast_to(
            np.concatenate([np.zeros(64, dtype=int), np.ones(64, dtype=int)]),
            validation_coefficients.shape,
        ).ravel()
        train_key += train_sb * raw_count * PHASE_BINS
        validation_key += validation_sb * raw_count * PHASE_BINS
        key_count = raw_count * PHASE_BINS * 2
    else:
        key_count = raw_count * PHASE_BINS
    total = np.bincount(train_key, minlength=key_count)
    ones = np.bincount(train_key, weights=train_bits, minlength=key_count)
    global_majority = int(np.mean(train_bits) >= 0.5)
    majority = np.full(key_count, global_majority, dtype=np.uint8)
    observed = total > 0
    majority[observed] = (ones[observed] * 2 >= total[observed]).astype(np.uint8)
    return majority[validation_key]


def lookup_rows(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for subband in SUBBANDS:
            predictions = lookup_predict(
                train[f"coeff_{condition}"], train["targets"],
                validation[f"coeff_{condition}"], subband,
            )
            targets = validation["targets"][:, subband_slice(subband)].ravel()
            rows.append(
                {
                    "condition": condition,
                    "subband": subband,
                    "phase_bins": PHASE_BINS,
                    "raw_quantile_bins": RAW_QUANTILE_BINS,
                    "validation_rows": int(targets.size),
                    "validation_ber": float(np.mean(predictions != targets)),
                }
            )
    return rows


def knn_rows(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        print(f"[kNN] {condition}", flush=True)
        train_features = feature_matrix(train[f"coeff_{condition}"])
        validation_features = feature_matrix(validation[f"coeff_{condition}"])
        for subband in SUBBANDS:
            selection = subband_slice(subband)
            x_train = train_features[:, selection].reshape(-1, 4).astype(np.float64)
            y_train = train["targets"][:, selection].ravel()
            x_validation = validation_features[:, selection].reshape(-1, 4).astype(np.float64)
            y_validation = validation["targets"][:, selection].ravel()
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0)
            scale[scale < 1e-12] = 1.0
            tree = cKDTree((x_train - mean) / scale)
            _distances, indices = tree.query(
                (x_validation - mean) / scale, k=KNN_K, workers=-1
            )
            predictions = (np.mean(y_train[indices], axis=1) >= 0.5).astype(np.uint8)
            rows.append(
                {
                    "condition": condition,
                    "subband": subband,
                    "classifier": "standardized_k_nearest_neighbors",
                    "k": KNN_K,
                    "training_rows": int(y_train.size),
                    "validation_rows": int(y_validation.size),
                    "validation_ber": float(np.mean(predictions != y_validation)),
                }
            )
    return rows


def ambiguous_rows(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        train_phase = phase_values(train[f"coeff_{condition}"])
        validation_phase = phase_values(validation[f"coeff_{condition}"])
        for subband in ("LH2", "HL2"):
            selection = subband_slice(subband)
            train_bits = train["targets"][:, selection].ravel()
            train_feature_values = feature_matrix(train[f"coeff_{condition}"])[:, selection].reshape(-1, 4)
            validation_feature_values = feature_matrix(validation[f"coeff_{condition}"])[:, selection].reshape(-1, 4)
            for scope in ("phase_only", "current_4_features"):
              for decimals in ROUND_DECIMALS:
                if scope == "phase_only":
                    train_rounded = np.round(train_phase[:, selection].ravel(), decimals)
                    validation_rounded = np.round(validation_phase[:, selection].ravel(), decimals)
                    train_keys, validation_keys = train_rounded, validation_rounded
                else:
                    train_rounded = np.round(train_feature_values, decimals)
                    validation_rounded = np.round(validation_feature_values, decimals)
                    train_keys = [tuple(row) for row in train_rounded]
                    validation_keys = [tuple(row) for row in validation_rounded]
                bit_mask_by_key: dict[int, int] = defaultdict(int)
                for key, bit in zip(train_keys, train_bits):
                    bit_mask_by_key[key] |= 1 << int(bit)
                masks = np.asarray(list(bit_mask_by_key.values()))
                ambiguous_keys = {key for key, mask in bit_mask_by_key.items() if mask == 3}
                validation_ambiguous = np.asarray([key in ambiguous_keys for key in validation_keys])
                rows.append(
                    {
                        "condition": condition,
                        "subband": subband,
                        "representation_scope": scope,
                        "phase_round_decimals": decimals,
                        "rounding_half_width": 0.5 / (10 ** decimals),
                        "training_bins": len(bit_mask_by_key),
                        "only_bit_0_bins": int(np.sum(masks == 1)),
                        "only_bit_1_bins": int(np.sum(masks == 2)),
                        "both_bits_bins": int(np.sum(masks == 3)),
                        "both_bits_bin_proportion": float(np.mean(masks == 3)),
                        "validation_samples_in_ambiguous_bins": int(np.sum(validation_ambiguous)),
                        "validation_ambiguous_bin_proportion": float(np.mean(validation_ambiguous)),
                    }
                )
    return rows


def transition_rows(validation: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    targets = validation["targets"]
    for condition in CONDITIONS:
        clean = validation[f"clean_{condition}"]
        attacked = validation[f"coeff_{condition}"]
        clean_phase = phase_values(clean)
        attacked_phase = phase_values(attacked)
        displacement = np.abs(attacked - clean)
        phase_displacement = circular_phase_distance(clean_phase, attacked_phase)
        clean_bits = np.vectorize(qim_extract_bit)(clean, DELTA).astype(np.uint8)
        attacked_bits = np.vectorize(qim_extract_bit)(attacked, DELTA).astype(np.uint8)
        for subband in ("LH2", "HL2", "combined"):
            selection = subband_slice(subband)
            for target_bit in (0, 1, "all"):
                if target_bit == "all":
                    mask = np.ones_like(targets[:, selection], dtype=bool)
                else:
                    mask = targets[:, selection] == target_bit
                values = displacement[:, selection][mask]
                phases = phase_displacement[:, selection][mask]
                clean_selected = clean_bits[:, selection][mask]
                attacked_selected = attacked_bits[:, selection][mask]
                target_selected = targets[:, selection][mask]
                rows.append(
                    {
                        "condition": condition,
                        "subband": subband,
                        "target_bit": target_bit,
                        "count": int(values.size),
                        "median_absolute_coefficient_displacement": float(np.median(values)),
                        "p90_absolute_coefficient_displacement": float(np.quantile(values, 0.90)),
                        "fraction_displacement_gt_delta_over_4": float(np.mean(values > DELTA / 4)),
                        "fraction_displacement_gt_delta_over_2": float(np.mean(values > DELTA / 2)),
                        "median_circular_phase_displacement": float(np.median(phases)),
                        "p90_circular_phase_displacement": float(np.quantile(phases, 0.90)),
                        "classical_decision_changed_from_clean": float(np.mean(attacked_selected != clean_selected)),
                        "attacked_classical_error_rate": float(np.mean(attacked_selected != target_selected)),
                    }
                )
    return rows


def oracle_rows(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> list[dict]:
    """Analysis-only lookup with attacked phase plus clean watermarked phase."""
    rows = []
    bins = 32
    for condition in CONDITIONS:
        for subband in ("LH2", "HL2", "combined"):
            selection = subband_slice(subband)
            train_attacked = phase_values(train[f"coeff_{condition}"][:, selection]).ravel()
            train_clean = phase_values(train[f"clean_{condition}"][:, selection]).ravel()
            train_bits = train["targets"][:, selection].ravel()
            val_attacked = phase_values(validation[f"coeff_{condition}"][:, selection]).ravel()
            val_clean = phase_values(validation[f"clean_{condition}"][:, selection]).ravel()
            val_bits = validation["targets"][:, selection].ravel()
            train_key = np.minimum((train_attacked * bins).astype(int), bins - 1)
            val_key = np.minimum((val_attacked * bins).astype(int), bins - 1)
            if subband == "combined":
                train_sb = np.broadcast_to(np.concatenate([np.zeros(64, dtype=int), np.ones(64, dtype=int)]), train["targets"].shape).ravel()
                val_sb = np.broadcast_to(np.concatenate([np.zeros(64, dtype=int), np.ones(64, dtype=int)]), validation["targets"].shape).ravel()
                train_key += train_sb * bins
                val_key += val_sb * bins
                attacked_key_count = bins * 2
            else:
                attacked_key_count = bins
            train_clean_bin = np.minimum((train_clean * bins).astype(int), bins - 1)
            val_clean_bin = np.minimum((val_clean * bins).astype(int), bins - 1)
            paired_train_key = train_key + train_clean_bin * attacked_key_count
            paired_val_key = val_key + val_clean_bin * attacked_key_count
            for label, keys_train, keys_val, key_count in (
                ("attacked_phase_only", train_key, val_key, attacked_key_count),
                ("attacked_plus_clean_watermarked_phase", paired_train_key, paired_val_key, attacked_key_count * bins),
            ):
                totals = np.bincount(keys_train, minlength=key_count)
                ones = np.bincount(keys_train, weights=train_bits, minlength=key_count)
                majority = np.full(key_count, int(np.mean(train_bits) >= 0.5), dtype=np.uint8)
                observed = totals > 0
                majority[observed] = (ones[observed] * 2 >= totals[observed]).astype(np.uint8)
                rows.append(
                    {
                        "condition": condition,
                        "subband": subband,
                        "analysis_input": label,
                        "bins_per_phase_axis": bins,
                        "validation_ber": float(np.mean(majority[keys_val] != val_bits)),
                        "deployable_blind_extractor": False if "clean" in label else True,
                    }
                )
    return rows


def keyed(rows: list[dict], key: str) -> dict[tuple[str, str], dict]:
    return {(row["condition"], row["subband"]): row for row in rows if key in row}


def main() -> int:
    train_paths = sorted(TRAIN_DIR.glob("*.png"))[:TRAIN_IMAGE_COUNT]
    validation_paths = sorted(VAL_DIR.glob("*.png"))[:VAL_IMAGE_COUNT]
    if len(train_paths) != TRAIN_IMAGE_COUNT or len(validation_paths) != VAL_IMAGE_COUNT:
        raise SystemExit("Stage 2C requires the same 500 train and 100 validation images as Stage 2B.")
    locations = location_rows((128, 128))
    print("Regenerating exact Stage 2B coefficient data; no model training.", flush=True)
    train = load_split(train_paths, TRAIN_PAYLOAD_SEED, locations, "train")
    validation = load_split(validation_paths, VAL_PAYLOAD_SEED, locations, "validation")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing artifacts in {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distributions = distribution_rows(train)
    overlaps = overlap_rows(train)
    lookups = lookup_rows(train, validation)
    classifiers = knn_rows(train, validation)
    ambiguous = ambiguous_rows(train, validation)
    transitions = transition_rows(validation)
    oracle = oracle_rows(train, validation)

    with (STAGE2B_DIR / "per_condition_summary.csv").open(newline="", encoding="utf-8") as handle:
        cnn_by_condition = {row["condition"]: row for row in csv.DictReader(handle)}
    overlap_map = keyed(overlaps, "histogram_overlap_coefficient")
    lookup_map = keyed(lookups, "validation_ber")
    classifier_map = keyed(classifiers, "validation_ber")
    ambiguity_map = {
        (row["condition"], row["subband"]): row
        for row in ambiguous
        if row["phase_round_decimals"] == 2
        and row["representation_scope"] == "current_4_features"
    }
    transition_map = {
        (row["condition"], row["subband"]): row
        for row in transitions if row["target_bit"] == "all"
    }
    condition_summary = []
    for condition in CONDITIONS:
        for subband in SUBBANDS:
            cnn_key = "stage2b_cnn_ber" if subband == "combined" else f"stage2b_{subband.lower()}_ber"
            transition = transition_map[(condition, subband)]
            if subband == "combined":
                ambiguous_proportion = float(np.mean([
                    ambiguity_map[(condition, "LH2")]["validation_ambiguous_bin_proportion"],
                    ambiguity_map[(condition, "HL2")]["validation_ambiguous_bin_proportion"],
                ]))
            else:
                ambiguous_proportion = ambiguity_map[(condition, subband)]["validation_ambiguous_bin_proportion"]
            condition_summary.append(
                {
                    "condition": condition,
                    "subband": subband,
                    "stage2b_cnn_ber": float(cnn_by_condition[condition][cnn_key]),
                    "classical_ber": transition["attacked_classical_error_rate"],
                    "empirical_lookup_ber": lookup_map[(condition, subband)]["validation_ber"],
                    "knn_ber": classifier_map[(condition, subband)]["validation_ber"],
                    "phase_histogram_overlap": overlap_map[(condition, subband)]["histogram_overlap_coefficient"],
                    "validation_ambiguous_bin_proportion_4_features_rounded_2dp": ambiguous_proportion,
                    "classical_decision_changed_from_clean": transition["classical_decision_changed_from_clean"],
                }
            )

    write_csv(OUTPUT_DIR / "phase_distribution_summary.csv", distributions)
    write_csv(OUTPUT_DIR / "overlap_metrics.csv", overlaps)
    write_csv(OUTPUT_DIR / "empirical_lookup_results.csv", lookups)
    write_csv(OUTPUT_DIR / "lightweight_classifier_results.csv", classifiers)
    write_csv(OUTPUT_DIR / "ambiguous_bins.csv", ambiguous)
    write_csv(OUTPUT_DIR / "coefficient_transition_summary.csv", transitions)
    write_csv(OUTPUT_DIR / "oracle_style_results.csv", oracle)
    write_csv(OUTPUT_DIR / "condition_summary.csv", condition_summary)

    config = {
        "experiment": "Stage 2C non-training feature separability diagnostic",
        "training_performed": False,
        "source": "Regenerated exact Stage 2B train/validation identities in memory",
        "train_images": TRAIN_IMAGE_COUNT,
        "train_image_payload_pairs": TRAIN_IMAGE_COUNT * PAYLOADS_PER_IMAGE,
        "validation_images": VAL_IMAGE_COUNT,
        "validation_image_payload_pairs": VAL_IMAGE_COUNT * PAYLOADS_PER_IMAGE,
        "conditions": list(CONDITIONS),
        "payload_bits": PAYLOAD_BITS,
        "coefficient_seed": COEFFICIENT_SEED,
        "delta": DELTA,
        "train_payload_seed": TRAIN_PAYLOAD_SEED,
        "validation_payload_seed": VAL_PAYLOAD_SEED,
        "qim_phase_definition": "(coefficient / delta) mod 1",
        "qim_period_in_scaled_units": 1.0,
        "lookup": {"phase_bins": PHASE_BINS, "raw_quantile_bins": RAW_QUANTILE_BINS},
        "lightweight_classifier": {"name": "standardized k-nearest neighbors", "k": KNN_K},
        "ambiguous_bin_phase_round_decimals": list(ROUND_DECIMALS),
        "test_set_used": False,
        "cnn_loaded_or_trained": False,
        "oracle_note": "Uses clean watermarked coefficient phase for analysis only; non-deployable.",
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    conclusions = {
        "jpeg70_classification": "C. STRONGLY AMBIGUOUS",
        "jpeg50_classification": "D. NEAR INFORMATION LOSS",
        "practical_single_coefficient_ceiling": True,
        "interpretation": (
            "Fixed empirical lookup and k-nearest-neighbor rules do not materially "
            "outperform the Stage 2B CNN for JPEG70 or JPEG50. Target-conditioned "
            "phase overlap and contradictory rounded feature bins are substantial. "
            "This is practical separability evidence, not a formal information-theoretic proof."
        ),
        "recommended_next_action": (
            "Run one controlled local-spatial-context experiment using a fixed small "
            "neighborhood around each selected coefficient, with embedding, delta, "
            "seed, attacks, train/validation identities, and evaluation grid unchanged."
        ),
    }
    (OUTPUT_DIR / "summary_metrics.json").write_text(
        json.dumps(conclusions, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "condition_summary": condition_summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
