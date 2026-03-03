"""Multi-configuration evaluation orchestrator.

Drives the full evaluation sweep: for each config × image × attack,
embeds a watermark, applies the attack, extracts, and measures all metrics.
Saves incremental CSV after each config for crash recovery.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from attacks.suite import AttackResult, get_default_attacks
from benchmark.runner import BenchmarkConfig, embed_image, extract_and_measure
from evaluation.configs import EvalConfig
from evaluation.image_corpus import IMAGE_CATEGORIES, ImageCorpus
from evaluation.metrics import compute_nc, compute_psnr, compute_ssim
from watermark.masking import build_delta_map
from watermark.payload import decode_payload_bits, derive_seed, encode_payload
from watermark.preprocessor import (
    extract_y_channel,
    pad_to_multiple,
    rgb_to_ycbcr,
)
from watermark.reconstruction import reconstruct_image
from watermark.tiling import embed_watermark_tiled, extract_watermark_tiled


# ---------------------------------------------------------------------------
# Attack category mapping
# ---------------------------------------------------------------------------

ATTACK_CATEGORIES: dict[str, str] = {}
for _q in [50, 60, 70, 80, 85, 90]:
    ATTACK_CATEGORIES[f"jpeg_q{_q}"] = "compression"
for _d in [1080, 1440, 2048]:
    ATTACK_CATEGORIES[f"resize_{_d}"] = "scaling"
for _r in [10, 20, 30, 40]:
    ATTACK_CATEGORIES[f"crop_{_r}pct"] = "cropping"
ATTACK_CATEGORIES["screenshot"] = "screenshot"
ATTACK_CATEGORIES["format_chain"] = "format_conversion"
for _s in [2, 5, 10]:
    ATTACK_CATEGORIES[f"noise_sigma{_s}"] = "noise"
ATTACK_CATEGORIES["combined_chain"] = "combined"
ATTACK_CATEGORIES["none"] = "none"


@dataclass
class EvalResult:
    """Result of a single evaluation test (one image + one attack + one config)."""

    # Identity
    image_name: str
    image_category: str
    attack_name: str
    attack_category: str
    config_label: str
    # Config params
    wavelet: str
    delta: float
    adaptive: bool
    rs_nsym: int
    repetitions: int
    tiled: bool
    # Embedding quality (no attack)
    psnr_embed: float
    ssim_embed: float
    # Robustness (after attack)
    ber_pre_ecc: float
    nc_score: float
    recovery_success: bool
    confidence: float
    # Attack quality
    psnr_attacked: float
    ssim_attacked: float
    # Capacity
    payload_bits: int
    capacity_bpp: float
    # Timing
    embed_time_s: float
    extract_time_s: float
    # Metadata
    attack_seed: int
    notes: str


# CSV field order
_RESULT_FIELDS = [f.name for f in fields(EvalResult)]


class EvalRun:
    """Collection of evaluation results with CSV persistence."""

    def __init__(self) -> None:
        self.results: list[EvalResult] = []

    def add(self, result: EvalResult) -> None:
        self.results.append(result)

    def extend(self, results: list[EvalResult]) -> None:
        self.results.extend(results)

    def to_csv(self, path: str | Path) -> None:
        """Save all results to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            writer.writeheader()
            for r in self.results:
                row = asdict(r)
                # Format floats for readability
                for key in [
                    "delta", "psnr_embed", "ssim_embed", "ber_pre_ecc",
                    "nc_score", "confidence", "psnr_attacked", "ssim_attacked",
                    "capacity_bpp", "embed_time_s", "extract_time_s",
                ]:
                    if isinstance(row[key], float):
                        row[key] = f"{row[key]:.6f}"
                writer.writerow(row)

    @classmethod
    def from_csv(cls, path: str | Path) -> EvalRun:
        """Load results from CSV."""
        run = cls()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run.results.append(EvalResult(
                    image_name=row["image_name"],
                    image_category=row["image_category"],
                    attack_name=row["attack_name"],
                    attack_category=row["attack_category"],
                    config_label=row["config_label"],
                    wavelet=row["wavelet"],
                    delta=float(row["delta"]),
                    adaptive=row["adaptive"] == "True",
                    rs_nsym=int(row["rs_nsym"]),
                    repetitions=int(row["repetitions"]),
                    tiled=row["tiled"] == "True",
                    psnr_embed=float(row["psnr_embed"]),
                    ssim_embed=float(row["ssim_embed"]),
                    ber_pre_ecc=float(row["ber_pre_ecc"]),
                    nc_score=float(row["nc_score"]),
                    recovery_success=row["recovery_success"] == "True",
                    confidence=float(row["confidence"]),
                    psnr_attacked=float(row["psnr_attacked"]),
                    ssim_attacked=float(row["ssim_attacked"]),
                    payload_bits=int(row["payload_bits"]),
                    capacity_bpp=float(row["capacity_bpp"]),
                    embed_time_s=float(row["embed_time_s"]),
                    extract_time_s=float(row["extract_time_s"]),
                    attack_seed=int(row["attack_seed"]),
                    notes=row["notes"],
                ))
        return run


# ---------------------------------------------------------------------------
# Embedding / extraction wrappers for tiled configs
# ---------------------------------------------------------------------------

def _embed_image_tiled(
    image: np.ndarray,
    config: EvalConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed using tiled approach. Returns (watermarked_rgb, bits, time)."""
    bc = config.to_benchmark_config()
    t0 = time.perf_counter()

    bits = encode_payload(
        bc.artist_id, image, bc.key,
        timestamp=bc.timestamp, rs_nsym=bc.rs_nsym,
        repetitions=bc.repetitions,
    )

    ycbcr = rgb_to_ycbcr(image)
    y = extract_y_channel(ycbcr)
    y_padded, pad_sizes = pad_to_multiple(y, 4)
    ycbcr_padded, _ = pad_to_multiple(ycbcr, 4)

    seed = derive_seed(bc.key)

    wm_y = embed_watermark_tiled(
        y_padded, bits, seed=seed, delta=config.delta,
        wavelet=config.wavelet, tile_size=config.tile_size,
    )

    watermarked = reconstruct_image(ycbcr_padded, wm_y, pad_sizes)
    elapsed = time.perf_counter() - t0
    return watermarked, bits, elapsed


def _extract_and_measure_tiled(
    attacked_image: np.ndarray,
    original_bits: np.ndarray,
    config: EvalConfig,
) -> tuple[float, bool, float, float]:
    """Extract using tiled approach. Returns (ber, success, confidence, time)."""
    bc = config.to_benchmark_config()
    t0 = time.perf_counter()

    seed = derive_seed(bc.key)
    num_bits = len(original_bits)

    ycbcr_att = rgb_to_ycbcr(attacked_image)
    y_att = extract_y_channel(ycbcr_att)
    y_att_padded, _ = pad_to_multiple(y_att, 4)

    extracted_bits = extract_watermark_tiled(
        y_att_padded, num_bits=num_bits, seed=seed,
        delta=config.delta, wavelet=config.wavelet,
        tile_size=config.tile_size, key=bc.key,
        rs_nsym=bc.rs_nsym, repetitions=bc.repetitions,
    )

    from watermark.extraction import compute_ber
    ber = compute_ber(original_bits, extracted_bits)

    # Confidence approximation: 1 - 2*BER (maps 0→1, 0.5→0)
    confidence = max(0.0, 1.0 - 2.0 * ber)

    recovery_success = False
    try:
        decode_payload_bits(
            extracted_bits, bc.key,
            rs_nsym=bc.rs_nsym, repetitions=bc.repetitions,
        )
        recovery_success = True
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    return ber, recovery_success, confidence, elapsed


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def _get_image_category(image_name: str) -> str:
    """Look up the category for an image name."""
    for cat, names in IMAGE_CATEGORIES.items():
        if image_name in names:
            return cat
    return "unknown"


def _get_attack_category(attack_name: str) -> str:
    """Look up the category for an attack name."""
    return ATTACK_CATEGORIES.get(attack_name, "unknown")


def run_single_config(
    images: dict[str, np.ndarray],
    config: EvalConfig,
    attacks: list | None = None,
    stochastic_seeds: list[int] | None = None,
) -> list[EvalResult]:
    """Run evaluation for a single config across all images and attacks.

    Args:
        images: {name: rgb_uint8_array}.
        config: Evaluation configuration.
        attacks: List of (name, fn, kwargs). Defaults to standard suite.
        stochastic_seeds: Seeds for stochastic attacks (crop, noise). If provided,
            stochastic attacks are run once per seed for std dev computation.

    Returns:
        List of EvalResult instances.
    """
    if attacks is None:
        attacks = get_default_attacks()
    if stochastic_seeds is None:
        stochastic_seeds = [0]

    bc = config.to_benchmark_config()
    results: list[EvalResult] = []

    for img_name, image in images.items():
        img_cat = _get_image_category(img_name)

        # Embed
        if config.tiled:
            watermarked, bits, embed_time = _embed_image_tiled(image, config)
        else:
            watermarked, bits, embed_time = embed_image(image, bc)

        # Embedding quality (no attack)
        psnr_embed = compute_psnr(image, watermarked)
        ssim_embed = compute_ssim(image, watermarked)
        payload_bits = len(bits)
        capacity_bpp = payload_bits / (image.shape[0] * image.shape[1])

        # No-attack baseline result
        results.append(EvalResult(
            image_name=img_name,
            image_category=img_cat,
            attack_name="none",
            attack_category="none",
            config_label=config.label,
            wavelet=config.wavelet,
            delta=config.delta,
            adaptive=config.adaptive,
            rs_nsym=config.rs_nsym,
            repetitions=config.repetitions,
            tiled=config.tiled,
            psnr_embed=psnr_embed,
            ssim_embed=ssim_embed,
            ber_pre_ecc=0.0,
            nc_score=1.0,
            recovery_success=True,
            confidence=1.0,
            psnr_attacked=psnr_embed,
            ssim_attacked=ssim_embed,
            payload_bits=payload_bits,
            capacity_bpp=capacity_bpp,
            embed_time_s=embed_time,
            extract_time_s=0.0,
            attack_seed=0,
            notes="",
        ))

        # Apply each attack
        for atk_name, atk_fn, atk_kwargs in attacks:
            # Determine if this attack is stochastic (has 'seed' param)
            is_stochastic = "seed" in atk_kwargs or atk_name.startswith("crop_") or atk_name.startswith("noise_")
            seeds_to_use = stochastic_seeds if is_stochastic else [0]

            for seed_val in seeds_to_use:
                try:
                    # Override seed for stochastic attacks
                    kwargs = dict(atk_kwargs)
                    if is_stochastic and "seed" in kwargs:
                        kwargs["seed"] = seed_val

                    result: AttackResult = atk_fn(watermarked, **kwargs)
                    attacked = result.image

                    # Resize back if dimensions changed (for non-tiled extraction)
                    if attacked.shape[:2] != watermarked.shape[:2]:
                        if not config.tiled:
                            attacked_resized = cv2.resize(
                                cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR),
                                (watermarked.shape[1], watermarked.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4,
                            )
                            attacked_resized = cv2.cvtColor(
                                attacked_resized, cv2.COLOR_BGR2RGB,
                            )
                        else:
                            attacked_resized = attacked
                    else:
                        attacked_resized = attacked

                    # SSIM/PSNR between original and attacked (use min shape)
                    if attacked_resized.shape == image.shape:
                        psnr_att = compute_psnr(image, attacked_resized)
                        ssim_att = compute_ssim(image, attacked_resized)
                    else:
                        # Shapes differ (crop/resize) — compute on resized version
                        att_for_quality = cv2.resize(
                            cv2.cvtColor(attacked_resized, cv2.COLOR_RGB2BGR),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        att_for_quality = cv2.cvtColor(
                            att_for_quality, cv2.COLOR_BGR2RGB,
                        )
                        psnr_att = compute_psnr(image, att_for_quality)
                        ssim_att = compute_ssim(image, att_for_quality)

                    # Extract and measure
                    if config.tiled:
                        ber, success, conf, ext_time = _extract_and_measure_tiled(
                            attacked_resized, bits, config,
                        )
                    else:
                        ber, success, conf, ext_time = extract_and_measure(
                            attacked_resized, bits, bc,
                        )

                    # Normalized correlation
                    from watermark.extraction import extract_from_image
                    nc = compute_nc(
                        bits,
                        _quick_extract_bits(attacked_resized, bits, config),
                    )

                    results.append(EvalResult(
                        image_name=img_name,
                        image_category=img_cat,
                        attack_name=atk_name,
                        attack_category=_get_attack_category(atk_name),
                        config_label=config.label,
                        wavelet=config.wavelet,
                        delta=config.delta,
                        adaptive=config.adaptive,
                        rs_nsym=config.rs_nsym,
                        repetitions=config.repetitions,
                        tiled=config.tiled,
                        psnr_embed=psnr_embed,
                        ssim_embed=ssim_embed,
                        ber_pre_ecc=ber,
                        nc_score=nc,
                        recovery_success=success,
                        confidence=conf,
                        psnr_attacked=psnr_att,
                        ssim_attacked=ssim_att,
                        payload_bits=payload_bits,
                        capacity_bpp=capacity_bpp,
                        embed_time_s=embed_time,
                        extract_time_s=ext_time,
                        attack_seed=seed_val,
                        notes="",
                    ))

                except Exception as e:
                    results.append(EvalResult(
                        image_name=img_name,
                        image_category=img_cat,
                        attack_name=atk_name,
                        attack_category=_get_attack_category(atk_name),
                        config_label=config.label,
                        wavelet=config.wavelet,
                        delta=config.delta,
                        adaptive=config.adaptive,
                        rs_nsym=config.rs_nsym,
                        repetitions=config.repetitions,
                        tiled=config.tiled,
                        psnr_embed=psnr_embed,
                        ssim_embed=ssim_embed,
                        ber_pre_ecc=1.0,
                        nc_score=0.0,
                        recovery_success=False,
                        confidence=0.0,
                        psnr_attacked=0.0,
                        ssim_attacked=0.0,
                        payload_bits=payload_bits,
                        capacity_bpp=capacity_bpp,
                        embed_time_s=embed_time,
                        extract_time_s=0.0,
                        attack_seed=seed_val,
                        notes=str(e),
                    ))

    return results


def _quick_extract_bits(
    image: np.ndarray,
    original_bits: np.ndarray,
    config: EvalConfig,
) -> np.ndarray:
    """Quick bit extraction for NC computation (pre-ECC raw bits)."""
    bc = config.to_benchmark_config()
    seed = derive_seed(bc.key)
    num_bits = len(original_bits)

    if config.tiled:
        ycbcr = rgb_to_ycbcr(image)
        y = extract_y_channel(ycbcr)
        y_padded, _ = pad_to_multiple(y, 4)
        return extract_watermark_tiled(
            y_padded, num_bits=num_bits, seed=seed,
            delta=config.delta, wavelet=config.wavelet,
            tile_size=config.tile_size, key=bc.key,
            rs_nsym=bc.rs_nsym, repetitions=bc.repetitions,
        )
    else:
        from watermark.extraction import extract_from_image
        extracted, _ = extract_from_image(
            image, num_bits=num_bits, seed=seed,
            delta=config.delta, wavelet=config.wavelet,
        )
        return extracted


def run_full_evaluation(
    corpus: ImageCorpus,
    configs: list[EvalConfig],
    output_dir: str | Path,
    attacks: list | None = None,
    stochastic_seeds: list[int] | None = None,
) -> EvalRun:
    """Run the full evaluation sweep across all configs and images.

    Saves incremental CSV after each config for crash recovery.

    Args:
        corpus: Image corpus to evaluate.
        configs: List of configurations to sweep.
        output_dir: Directory for output files.
        attacks: Attack suite. Defaults to standard.
        stochastic_seeds: Seeds for stochastic attacks. Defaults to [0, 1, 2].

    Returns:
        EvalRun with all results.
    """
    if stochastic_seeds is None:
        stochastic_seeds = [0, 1, 2]

    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    images = corpus.get_images_dict()
    run = EvalRun()

    total_configs = len(configs)
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Config {i}/{total_configs}: {config.label}")
        print(f"{'='*60}")

        config_results = run_single_config(
            images, config, attacks=attacks,
            stochastic_seeds=stochastic_seeds,
        )
        run.extend(config_results)

        # Incremental save
        run.to_csv(results_dir / "full_evaluation.csv")
        print(f"  -> {len(config_results)} results saved (total: {len(run.results)})")

    return run
