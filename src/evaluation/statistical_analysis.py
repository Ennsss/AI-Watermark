"""Paired statistics for classical-vs-CNN BER comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedComparison:
    """Summary of paired BER differences for one degradation condition."""

    n: int
    classical_mean: float
    cnn_mean: float
    classical_median: float
    cnn_median: float
    mean_difference: float
    p_value: float
    rank_biserial: float


def rank_biserial_from_pairs(classical_ber: np.ndarray, cnn_ber: np.ndarray) -> float:
    """Compute rank-biserial correlation from paired BER differences."""
    diff = np.asarray(classical_ber, dtype=float) - np.asarray(cnn_ber, dtype=float)
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0

    abs_diff = np.abs(nonzero)
    order = np.argsort(abs_diff)
    ranks = np.empty_like(abs_diff, dtype=float)
    ranks[order] = np.arange(1, len(abs_diff) + 1, dtype=float)
    pos = float(np.sum(ranks[nonzero > 0]))
    neg = float(np.sum(ranks[nonzero < 0]))
    denom = pos + neg
    return 0.0 if denom == 0 else (pos - neg) / denom


def wilcoxon_paired_ber(classical_ber: np.ndarray, cnn_ber: np.ndarray) -> PairedComparison:
    """Run a Wilcoxon signed-rank test for paired BER arrays."""
    classical = np.asarray(classical_ber, dtype=float)
    cnn = np.asarray(cnn_ber, dtype=float)
    if classical.shape != cnn.shape:
        raise ValueError(f"Shape mismatch: {classical.shape} vs {cnn.shape}")

    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise ImportError(
            "Wilcoxon analysis requires scipy. Install the optional stats dependencies."
        ) from exc

    result = wilcoxon(classical, cnn, zero_method="wilcox", alternative="two-sided")
    return PairedComparison(
        n=int(classical.size),
        classical_mean=float(np.mean(classical)),
        cnn_mean=float(np.mean(cnn)),
        classical_median=float(np.median(classical)),
        cnn_median=float(np.median(cnn)),
        mean_difference=float(np.mean(classical - cnn)),
        p_value=float(result.pvalue),
        rank_biserial=rank_biserial_from_pairs(classical, cnn),
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return reject/not-reject decisions using Holm-Bonferroni correction."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    decisions = [False] * len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        threshold = alpha / (len(p_values) - rank)
        if p_value <= threshold:
            decisions[idx] = True
        else:
            break
    return decisions

