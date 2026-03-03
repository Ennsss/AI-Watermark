"""Configuration sweep definitions for systematic evaluation.

Provides EvalConfig dataclass and factory functions for one-at-a-time
parameter sweeps used in thesis evaluation tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.runner import BenchmarkConfig


@dataclass
class EvalConfig:
    """A single evaluation configuration."""

    label: str
    wavelet: str = "haar"
    delta: float = 60.0
    adaptive: bool = False
    delta_min: float = 20.0
    delta_max: float = 80.0
    rs_nsym: int = 128
    repetitions: int = 1
    tiled: bool = False
    tile_size: int = 256

    def to_benchmark_config(self) -> BenchmarkConfig:
        """Convert to BenchmarkConfig for use with embed_image/extract_and_measure."""
        return BenchmarkConfig(
            wavelet=self.wavelet,
            delta=self.delta,
            adaptive=self.adaptive,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            rs_nsym=self.rs_nsym,
            repetitions=self.repetitions,
        )


# ---------------------------------------------------------------------------
# Sweep factory functions
# ---------------------------------------------------------------------------

def get_baseline_config() -> EvalConfig:
    """Default baseline configuration."""
    return EvalConfig(label="baseline_haar_d60")


def get_delta_sweep() -> list[EvalConfig]:
    """Delta parameter sweep: 30, 40, 50, 60, 70, 80."""
    return [
        EvalConfig(label=f"delta_{int(d)}", delta=d)
        for d in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    ]


def get_wavelet_comparison() -> list[EvalConfig]:
    """Haar vs db4 wavelet comparison."""
    return [
        EvalConfig(label="wavelet_haar", wavelet="haar"),
        EvalConfig(label="wavelet_db4", wavelet="db4"),
    ]


def get_repetition_comparison() -> list[EvalConfig]:
    """Repetition coding R=1 vs R=3."""
    return [
        EvalConfig(label="rep_1", repetitions=1),
        EvalConfig(label="rep_3", repetitions=3),
    ]


def get_tiling_comparison() -> list[EvalConfig]:
    """Tiled vs non-tiled embedding."""
    return [
        EvalConfig(label="no_tiling", tiled=False),
        EvalConfig(label="tiled_256", tiled=True, tile_size=256),
    ]


def get_adaptive_comparison() -> list[EvalConfig]:
    """Uniform vs adaptive masking."""
    return [
        EvalConfig(label="uniform", adaptive=False),
        EvalConfig(label="adaptive", adaptive=True),
    ]


def get_full_sweep() -> list[EvalConfig]:
    """All configurations deduplicated (~15 configs).

    Combines delta sweep + wavelet + repetition + tiling + adaptive
    with duplicates removed by label.
    """
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
