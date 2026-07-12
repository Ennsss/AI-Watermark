"""Statistical aggregation for evaluation results.

Provides grouping, mean/std/95% CI computation, and pivot table generation
for thesis-grade statistical reporting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist

from evaluation.runner import EvalResult, EvalRun


@dataclass
class AggregatedMetric:
    """A metric with statistical summary."""

    mean: float
    std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.std:.4f} (n={self.n})"

    @classmethod
    def from_values(cls, values: list[float]) -> AggregatedMetric:
        """Compute aggregated metric from a list of values."""
        if not values:
            return cls(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n=0)

        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if n > 1 and std > 0:
            se = std / np.sqrt(n)
            ci_lo, ci_hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=se)
            ci_lower = float(ci_lo)
            ci_upper = float(ci_hi)
        else:
            ci_lower = mean
            ci_upper = mean

        return cls(mean=mean, std=std, ci_lower=ci_lower, ci_upper=ci_upper, n=n)


@dataclass
class AggregatedRow:
    """A row of aggregated metrics for a group."""

    group_key: str
    ber: AggregatedMetric
    nc: AggregatedMetric
    psnr: AggregatedMetric
    ssim: AggregatedMetric
    recovery_rate: float
    n_total: int


def aggregate_by(
    results: list[EvalResult],
    group_by: str,
) -> list[AggregatedRow]:
    """Group results and compute aggregate statistics.

    Args:
        results: List of EvalResult instances.
        group_by: Field name to group by. Common values:
            "attack_name", "config_label", "image_category",
            "attack_category", "image_name".

    Returns:
        List of AggregatedRow, one per group, sorted by group_key.
    """
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        key = getattr(r, group_by)
        groups[str(key)].append(r)

    rows = []
    for key in sorted(groups.keys()):
        group = groups[key]
        rows.append(AggregatedRow(
            group_key=key,
            ber=AggregatedMetric.from_values([r.ber_pre_ecc for r in group]),
            nc=AggregatedMetric.from_values([r.nc_score for r in group]),
            psnr=AggregatedMetric.from_values([r.psnr_attacked for r in group]),
            ssim=AggregatedMetric.from_values([r.ssim_attacked for r in group]),
            recovery_rate=_compute_recovery_rate(group),
            n_total=len(group),
        ))

    return rows


def aggregate_embedding_quality(
    results: list[EvalResult],
    group_by: str = "config_label",
) -> list[AggregatedRow]:
    """Aggregate embedding quality metrics (no-attack results only).

    Args:
        results: Full result list (will be filtered to attack_name="none").
        group_by: Field to group by.

    Returns:
        List of AggregatedRow with PSNR/SSIM from embedding (no attack).
    """
    no_attack = [r for r in results if r.attack_name == "none"]
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in no_attack:
        key = getattr(r, group_by)
        groups[str(key)].append(r)

    rows = []
    for key in sorted(groups.keys()):
        group = groups[key]
        rows.append(AggregatedRow(
            group_key=key,
            ber=AggregatedMetric.from_values([r.ber_pre_ecc for r in group]),
            nc=AggregatedMetric.from_values([r.nc_score for r in group]),
            psnr=AggregatedMetric.from_values([r.psnr_embed for r in group]),
            ssim=AggregatedMetric.from_values([r.ssim_embed for r in group]),
            recovery_rate=_compute_recovery_rate(group),
            n_total=len(group),
        ))

    return rows


def pivot_table(
    results: list[EvalResult],
    row_key: str,
    col_key: str,
    metric: str,
) -> tuple[list[str], list[str], list[list[AggregatedMetric]]]:
    """Build a 2D pivot table of aggregated metrics.

    Args:
        results: List of EvalResult instances.
        row_key: Field for rows (e.g., "attack_name").
        col_key: Field for columns (e.g., "config_label").
        metric: Metric to aggregate. One of:
            "ber_pre_ecc", "nc_score", "psnr_attacked", "ssim_attacked",
            "psnr_embed", "ssim_embed", "recovery_rate".

    Returns:
        Tuple of (row_labels, col_labels, 2D list of AggregatedMetric).
    """
    # Collect all row/col keys
    row_keys: set[str] = set()
    col_keys: set[str] = set()
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in results:
        rk = str(getattr(r, row_key))
        ck = str(getattr(r, col_key))
        row_keys.add(rk)
        col_keys.add(ck)

        if metric == "recovery_rate":
            cells[(rk, ck)].append(1.0 if r.recovery_success else 0.0)
        else:
            cells[(rk, ck)].append(getattr(r, metric))

    row_labels = sorted(row_keys)
    col_labels = sorted(col_keys)

    grid = []
    for rk in row_labels:
        row = []
        for ck in col_labels:
            values = cells.get((rk, ck), [])
            row.append(AggregatedMetric.from_values(values))
        grid.append(row)

    return row_labels, col_labels, grid


def filter_results(
    results: list[EvalResult],
    *,
    attack_name: str | None = None,
    attack_category: str | None = None,
    config_label: str | None = None,
    image_category: str | None = None,
    exclude_no_attack: bool = False,
) -> list[EvalResult]:
    """Filter results by various criteria.

    Args:
        results: Full result list.
        attack_name: Filter to specific attack.
        attack_category: Filter to attack category.
        config_label: Filter to specific config.
        image_category: Filter to image category.
        exclude_no_attack: If True, exclude attack_name="none" results.

    Returns:
        Filtered list.
    """
    filtered = results
    if exclude_no_attack:
        filtered = [r for r in filtered if r.attack_name != "none"]
    if attack_name is not None:
        filtered = [r for r in filtered if r.attack_name == attack_name]
    if attack_category is not None:
        filtered = [r for r in filtered if r.attack_category == attack_category]
    if config_label is not None:
        filtered = [r for r in filtered if r.config_label == config_label]
    if image_category is not None:
        filtered = [r for r in filtered if r.image_category == image_category]
    return filtered


def _compute_recovery_rate(results: list[EvalResult]) -> float:
    """Compute recovery rate from results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.recovery_success) / len(results)
