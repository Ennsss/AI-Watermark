"""Experiment configuration definitions.

The main comparison keeps one fixed DWT-QIM setup for both classical and
CNN-assisted extraction. Delta candidates are retained only for calibration,
not broad optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.runner import BenchmarkConfig


@dataclass
class EvalConfig:
    """A single evaluation configuration."""

    label: str
    wavelet: str = "haar"
    delta: float = 16.0
    payload_bits: int = 128
    seed: int = 42
    adaptive: bool = False
    delta_min: float = 20.0
    delta_max: float = 80.0
    rs_nsym: int = 128
    repetitions: int = 1
    tiled: bool = False
    tile_size: int = 256
    use_legacy_secure_payload: bool = False

    def to_benchmark_config(self) -> BenchmarkConfig:
        """Convert to BenchmarkConfig for use with embed_image/extract_and_measure."""
        return BenchmarkConfig(
            wavelet=self.wavelet,
            delta=self.delta,
            payload_bits=self.payload_bits,
            payload_seed=self.seed,
            coefficient_seed=self.seed,
            adaptive=self.adaptive,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            rs_nsym=self.rs_nsym,
            repetitions=self.repetitions,
            use_legacy_secure_payload=self.use_legacy_secure_payload,
        )


# ---------------------------------------------------------------------------
# Sweep factory functions
# ---------------------------------------------------------------------------

def get_baseline_config() -> EvalConfig:
    """Paper-aligned main configuration."""
    return EvalConfig(label="main_haar_d16_raw128")


def get_delta_sweep() -> list[EvalConfig]:
    """Delta calibration candidates: 8, 16, 24, 32."""
    return [
        EvalConfig(label=f"delta_{int(d)}", delta=d)
        for d in [8.0, 16.0, 24.0, 32.0]
    ]


def get_wavelet_comparison() -> list[EvalConfig]:
    """Optional wavelet comparison outside the main experiment."""
    return [
        EvalConfig(label="wavelet_haar", wavelet="haar"),
        EvalConfig(label="wavelet_db4", wavelet="db4"),
    ]


def get_repetition_comparison() -> list[EvalConfig]:
    """Legacy repetition comparison outside the main experiment."""
    return [
        EvalConfig(label="rep_1", repetitions=1),
        EvalConfig(label="rep_3", repetitions=3, use_legacy_secure_payload=True),
    ]


def get_tiling_comparison() -> list[EvalConfig]:
    """Legacy tiling comparison outside the main experiment."""
    return [
        EvalConfig(label="no_tiling", tiled=False),
        EvalConfig(label="tiled_256", tiled=True, tile_size=256),
    ]


def get_adaptive_comparison() -> list[EvalConfig]:
    """Optional adaptive masking comparison outside the main experiment."""
    return [
        EvalConfig(label="uniform", adaptive=False),
        EvalConfig(label="adaptive", adaptive=True),
    ]


def get_full_sweep() -> list[EvalConfig]:
    """Legacy broad sweep retained for optional exploratory work."""
    configs: dict[str, EvalConfig] = {}

    for config_list in [
        get_delta_sweep(),
        get_wavelet_comparison(),
        get_repetition_comparison(),
        get_tiling_comparison(),
        get_adaptive_comparison(),
    ]:
        for cfg in config_list:
            if cfg.label not in configs:
                configs[cfg.label] = cfg

    return list(configs.values())


def get_configs_by_name(name: str) -> list[EvalConfig]:
    """Get a configuration sweep by name.

    Args:
        name: One of 'baseline', 'delta', 'wavelet', 'repetition',
              'tiling', 'adaptive', 'full'.

    Returns:
        List of EvalConfig instances.
    """
    dispatch = {
        "baseline": lambda: [get_baseline_config()],
        "delta": get_delta_sweep,
        "wavelet": get_wavelet_comparison,
        "repetition": get_repetition_comparison,
        "tiling": get_tiling_comparison,
        "adaptive": get_adaptive_comparison,
        "full": get_full_sweep,
    }
    factory = dispatch.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown config sweep: {name}. Options: {list(dispatch.keys())}"
        )
    return factory()
