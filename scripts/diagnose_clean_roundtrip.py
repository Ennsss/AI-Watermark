"""Diagnose clean-roundtrip clipping on the frozen Run 1 sample set."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.preprocess import TARGET_SIZE, center_crop_square, resize_square
from evaluation.metrics import compute_psnr, compute_ssim
from watermark.embedding import (
    _get_embedding_locations,
    dwt2_decompose,
    dwt2_reconstruct,
    extract_watermark,
    qim_embed_bit,
    qim_extract_bit,
)
from watermark.extraction import compute_ber, extract_from_image
from watermark.preprocessor import (
    extract_y_channel,
    load_image,
    pad_to_multiple,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from watermark.reconstruction import reconstruct_image


OUTPUT_DIR = ROOT / "experiments/run1_clean_roundtrip_diagnostic"
TRAIN_DIR = ROOT / "data/curated/train"
SOURCE_COUNT = 4
PAYLOADS_PER_IMAGE = 8
PAYLOAD_BITS = 128
PAYLOAD_SEED = 20260808
COEFFICIENT_SEED = 42
DELTAS = (8.0, 16.0, 24.0, 32.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_coefficients(y: np.ndarray, bits: np.ndarray, delta: float) -> tuple[list, np.ndarray]:
    """Apply the production LH2/HL2 QIM rule while retaining DWT coefficients."""
    coeffs = deepcopy(dwt2_decompose(y, wavelet="haar", level=2, mode="symmetric"))
    lh2, hl2, hh2 = coeffs[1]
    half = len(bits) // 2
    lh_locations = _get_embedding_locations(lh2.shape, half, COEFFICIENT_SEED)
    hl_locations = _get_embedding_locations(hl2.shape, len(bits) - half, COEFFICIENT_SEED + 1)
    for index, (row, col) in enumerate(lh_locations):
        lh2[row, col] = qim_embed_bit(lh2[row, col], int(bits[index]), delta)
    for index, (row, col) in enumerate(hl_locations):
        hl2[row, col] = qim_embed_bit(hl2[row, col], int(bits[half + index]), delta)
    coeffs[1] = (lh2, hl2, hh2)
    extracted = np.concatenate(
        [
            np.asarray([qim_extract_bit(lh2[r, c], delta) for r, c in lh_locations]),
            np.asarray([qim_extract_bit(hl2[r, c], delta) for r, c in hl_locations]),
        ]
    ).astype(np.uint8)
    return coeffs, extracted


def extract_from_y(y: np.ndarray, delta: float) -> np.ndarray:
    return extract_watermark(
        y,
        PAYLOAD_BITS,
        COEFFICIENT_SEED,
        delta,
        "haar",
        2,
        "symmetric",
        target_subbands=("lh2", "hl2"),
    )


def float_ycbcr_to_rgb(ycbcr: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """One diagnostic variant: float inverse YCbCr, one final clip/cast."""
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1] - 128.0
    cr = ycbcr[:, :, 2] - 128.0
    rgb_float = np.stack(
        [
            y + 1.403 * cr,
            y - 0.344 * cb - 0.714 * cr,
            y + 1.773 * cb,
        ],
        axis=-1,
    )
    outside = (rgb_float < 0.0) | (rgb_float > 255.0)
    clipped = np.clip(rgb_float, 0.0, 255.0)
    return np.rint(clipped).astype(np.uint8), {
        "out_of_range_rgb_components": int(outside.sum()),
        "out_of_range_rgb_component_percentage": float(100.0 * outside.mean()),
        "mean_rgb_clipping_amount_all_components": float(np.mean(np.abs(rgb_float - clipped))),
        "maximum_rgb_clipping_amount": float(np.max(np.abs(rgb_float - clipped))),
    }


def luminance_stats(y: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}y_min": float(y.min()),
        f"{prefix}y_max": float(y.max()),
        f"{prefix}y_mean": float(y.mean()),
        f"{prefix}y_std": float(y.std()),
        f"{prefix}pct_y_le_8": float(100.0 * np.mean(y <= 8.0)),
        f"{prefix}pct_y_ge_247": float(100.0 * np.mean(y >= 247.0)),
        f"{prefix}pct_y_eq_0": float(100.0 * np.mean(y == 0.0)),
        f"{prefix}pct_y_eq_255": float(100.0 * np.mean(y == 255.0)),
    }


def main() -> int:
    paths = sorted(TRAIN_DIR.glob("*.png"))[:SOURCE_COUNT]
    if len(paths) != SOURCE_COUNT:
        raise SystemExit("The frozen four Run 1 training images are unavailable.")
    rng = np.random.default_rng(PAYLOAD_SEED)
    payloads = [rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8) for _ in range(32)]
    images = [resize_square(center_crop_square(load_image(path)), TARGET_SIZE) for path in paths]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_rows: list[dict] = []
    delta_sample_rows: list[dict] = []
    reconstructed_stats: dict[str, list[dict[str, float]]] = {path.name: [] for path in paths}
    variant_rows: list[dict] = []

    sample_index = 0
    for image_path, original in zip(paths, images):
        original_ycbcr = rgb_to_ycbcr(original)
        original_y = extract_y_channel(original_ycbcr)
        y_padded, pad_sizes = pad_to_multiple(original_y, 4)
        ycbcr_padded, _ = pad_to_multiple(original_ycbcr, 4)
        for payload_index in range(PAYLOADS_PER_IMAGE):
            bits = payloads[sample_index]
            for delta in DELTAS:
                coeffs, stage_a_bits = prepare_coefficients(y_padded, bits, delta)
                reconstructed_y = dwt2_reconstruct(coeffs, "haar", "symmetric")
                stage_b_bits = extract_from_y(reconstructed_y, delta)
                clipped_y = np.clip(reconstructed_y, 0.0, 255.0)
                stage_c_bits = extract_from_y(clipped_y, delta)
                existing_rgb = reconstruct_image(ycbcr_padded, reconstructed_y, pad_sizes)
                rgb_y = extract_y_channel(rgb_to_ycbcr(existing_rgb))
                rgb_y_padded, _ = pad_to_multiple(rgb_y, 4)
                stage_d_bits = extract_from_y(rgb_y_padded, delta)
                stage_e_bits, _ = extract_from_image(
                    existing_rgb,
                    PAYLOAD_BITS,
                    COEFFICIENT_SEED,
                    delta,
                    "haar",
                    2,
                    "symmetric",
                    target_subbands=("lh2", "hl2"),
                )

                below = reconstructed_y < 0.0
                above = reconstructed_y > 255.0
                clipping_amount = np.abs(reconstructed_y - clipped_y)
                clipped_mask = below | above
                base = {
                    "sample_index": sample_index,
                    "source_image": image_path.name,
                    "payload_index": payload_index,
                    "delta": int(delta),
                }
                diagnostic = {
                    **base,
                    "ber_a_modified_dwt": compute_ber(bits, stage_a_bits),
                    "ber_b_idwt_before_clipping": compute_ber(bits, stage_b_bits),
                    "ber_c_after_y_clipping": compute_ber(bits, stage_c_bits),
                    "ber_d_after_rgb_conversion": compute_ber(bits, stage_d_bits),
                    "ber_e_full_rgb_extraction": compute_ber(bits, stage_e_bits),
                    "reconstructed_y_min": float(reconstructed_y.min()),
                    "reconstructed_y_max": float(reconstructed_y.max()),
                    "pixels_y_lt_0": int(below.sum()),
                    "pct_y_lt_0": float(100.0 * below.mean()),
                    "pixels_y_gt_255": int(above.sum()),
                    "pct_y_gt_255": float(100.0 * above.mean()),
                    "total_clipped_pixels": int(clipped_mask.sum()),
                    "pct_clipped_pixels": float(100.0 * clipped_mask.mean()),
                    "mean_abs_clipping_amount_all_pixels": float(clipping_amount.mean()),
                    "mean_abs_clipping_amount_clipped_pixels": (
                        float(clipping_amount[clipped_mask].mean()) if clipped_mask.any() else 0.0
                    ),
                    "maximum_clipping_amount": float(clipping_amount.max()),
                }
                if delta == 16.0:
                    stage_rows.append(diagnostic)
                    reconstructed_stats[image_path.name].append(luminance_stats(reconstructed_y))

                delta_sample_rows.append(
                    {
                        **base,
                        "clean_ber": compute_ber(bits, stage_e_bits),
                        "clipped_y_pixels": int(clipped_mask.sum()),
                        "mean_abs_y_reconstruction_error": float(np.mean(np.abs(reconstructed_y - y_padded))),
                        "psnr_original_vs_watermarked": compute_psnr(original, existing_rgb),
                        "ssim_original_vs_watermarked": compute_ssim(original, existing_rgb),
                    }
                )

                if delta == 16.0:
                    variant_ycbcr = ycbcr_padded.copy()
                    variant_ycbcr[:, :, 0] = reconstructed_y
                    variant_rgb, gamut = float_ycbcr_to_rgb(variant_ycbcr)
                    if pad_sizes != (0, 0):
                        pad_h, pad_w = pad_sizes
                        variant_rgb = variant_rgb[
                            : -pad_h if pad_h else None, : -pad_w if pad_w else None
                        ]
                    variant_bits, _ = extract_from_image(
                        variant_rgb,
                        PAYLOAD_BITS,
                        COEFFICIENT_SEED,
                        delta,
                        "haar",
                        2,
                        "symmetric",
                        target_subbands=("lh2", "hl2"),
                    )
                    variant_recovered_y = extract_y_channel(rgb_to_ycbcr(variant_rgb))
                    variant_rows.append(
                        {
                            **base,
                            "existing_clean_ber": compute_ber(bits, stage_e_bits),
                            "float_variant_clean_ber": compute_ber(bits, variant_bits),
                            "existing_psnr": compute_psnr(original, existing_rgb),
                            "float_variant_psnr": compute_psnr(original, variant_rgb),
                            "existing_ssim": compute_ssim(original, existing_rgb),
                            "float_variant_ssim": compute_ssim(original, variant_rgb),
                            "existing_mean_abs_recovered_y_error": float(np.mean(np.abs(rgb_y - original_y))),
                            "float_variant_mean_abs_recovered_y_error": float(
                                np.mean(np.abs(variant_recovered_y - original_y))
                            ),
                            "valid_uint8_rgb": bool(variant_rgb.dtype == np.uint8),
                            "variant_rgb_min": int(variant_rgb.min()),
                            "variant_rgb_max": int(variant_rgb.max()),
                            **gamut,
                        }
                    )
            sample_index += 1

    delta_rows: list[dict] = []
    for delta in DELTAS:
        rows = [row for row in delta_sample_rows if row["delta"] == int(delta)]
        bers = np.asarray([row["clean_ber"] for row in rows])
        clipped = np.asarray([row["clipped_y_pixels"] for row in rows])
        y_error = np.asarray([row["mean_abs_y_reconstruction_error"] for row in rows])
        psnr = np.asarray([row["psnr_original_vs_watermarked"] for row in rows])
        ssim = np.asarray([row["ssim_original_vs_watermarked"] for row in rows])
        delta_rows.append(
            {
                "delta": int(delta),
                "mean_clean_ber": float(bers.mean()),
                "median_clean_ber": float(np.median(bers)),
                "maximum_clean_ber": float(bers.max()),
                "perfect_recoveries": int(np.sum(bers == 0.0)),
                "samples": len(rows),
                "average_clipped_y_pixels": float(clipped.mean()),
                "maximum_clipped_y_pixels": int(clipped.max()),
                "average_abs_y_reconstruction_error": float(y_error.mean()),
                "mean_psnr": float(psnr.mean()),
                "mean_ssim": float(ssim.mean()),
            }
        )

    image_rows: list[dict] = []
    for path, original in zip(paths, images):
        original_y = extract_y_channel(rgb_to_ycbcr(original))
        aggregate = reconstructed_stats[path.name]
        keys = list(aggregate[0])
        averaged_reconstructed = {
            f"reconstructed_avg_{key}": float(np.mean([row[key] for row in aggregate]))
            for key in keys
        }
        image_rows.append(
            {
                "source_image": path.name,
                **luminance_stats(original_y, "original_"),
                **averaged_reconstructed,
            }
        )

    existing_variant_bers = np.asarray([row["existing_clean_ber"] for row in variant_rows])
    float_variant_bers = np.asarray([row["float_variant_clean_ber"] for row in variant_rows])
    variant_summary = {
        "delta": 16,
        "existing_mean_ber": float(existing_variant_bers.mean()),
        "existing_perfect_recoveries": int(np.sum(existing_variant_bers == 0.0)),
        "float_variant_mean_ber": float(float_variant_bers.mean()),
        "float_variant_median_ber": float(np.median(float_variant_bers)),
        "float_variant_maximum_ber": float(float_variant_bers.max()),
        "float_variant_perfect_recoveries": int(np.sum(float_variant_bers == 0.0)),
        "mean_existing_psnr": float(np.mean([row["existing_psnr"] for row in variant_rows])),
        "mean_float_variant_psnr": float(np.mean([row["float_variant_psnr"] for row in variant_rows])),
        "mean_existing_ssim": float(np.mean([row["existing_ssim"] for row in variant_rows])),
        "mean_float_variant_ssim": float(np.mean([row["float_variant_ssim"] for row in variant_rows])),
        "all_variant_outputs_valid_uint8_rgb": bool(all(row["valid_uint8_rgb"] for row in variant_rows)),
    }
    summary = {
        "configuration": {
            "source_images": [path.name for path in paths],
            "payloads_per_image": PAYLOADS_PER_IMAGE,
            "samples": 32,
            "payload_rng_seed": PAYLOAD_SEED,
            "coefficient_seed": COEFFICIENT_SEED,
            "wavelet": "haar",
            "dwt_level": 2,
            "subbands": ["LH2", "HL2"],
            "payload_bits": PAYLOAD_BITS,
            "deltas": list(DELTAS),
            "attacks": [],
        },
        "first_error_stage_delta16": next(
            (
                name
                for name in [
                    "ber_a_modified_dwt",
                    "ber_b_idwt_before_clipping",
                    "ber_c_after_y_clipping",
                    "ber_d_after_rgb_conversion",
                    "ber_e_full_rgb_extraction",
                ]
                if any(row[name] > 0 for row in stage_rows)
            ),
            None,
        ),
        "delta_comparison": delta_rows,
        "float_variant": variant_summary,
        "configurations_reaching_32_of_32": [
            f"existing_delta_{row['delta']}" for row in delta_rows if row["perfect_recoveries"] == 32
        ]
        + (["float_rgb_variant_delta_16"] if variant_summary["float_variant_perfect_recoveries"] == 32 else []),
    }

    write_csv(OUTPUT_DIR / "stage_by_stage_delta16.csv", stage_rows)
    write_csv(OUTPUT_DIR / "delta_sample_metrics.csv", delta_sample_rows)
    write_csv(OUTPUT_DIR / "delta_comparison.csv", delta_rows)
    write_csv(OUTPUT_DIR / "image_luminance_diagnostics.csv", image_rows)
    write_csv(OUTPUT_DIR / "float_rgb_variant_samples.csv", variant_rows)
    with (OUTPUT_DIR / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
