"""Thesis-ready table generation in LaTeX and Markdown.

Generates 10 standard tables for watermarking thesis evaluation:
1. Embedding Quality (PSNR/SSIM per config)
2. Attack Category x Config -> Recovery Rate
3. Per-Attack Detailed (BER/NC/PSNR/SSIM)
4. Image Category x Attack Category -> mean BER
5. Delta Sweep (PSNR vs BER trade-off)
6. Capacity Analysis
7. False Positive Rate
8. Wavelet Comparison (haar vs db4)
9. Repetition Coding Effect
10. Tiled vs Non-Tiled crop results
"""

from __future__ import annotations

from pathlib import Path

from evaluation.aggregator import (
    AggregatedMetric,
    AggregatedRow,
    aggregate_by,
    aggregate_embedding_quality,
    filter_results,
    pivot_table,
)
from evaluation.runner import EvalResult, EvalRun


class ReportGenerator:
    """Generate all 10 thesis tables from evaluation results."""

    def __init__(self, eval_run: EvalRun, output_dir: str | Path) -> None:
        self.results = eval_run.results
        self.output_dir = Path(output_dir)
        self.latex_dir = self.output_dir / "reports" / "latex"
        self.markdown_dir = self.output_dir / "reports" / "markdown"
        self.plot_data_dir = self.output_dir / "figures" / "plot_data"

    def generate_all(self) -> None:
        """Generate all 10 tables in both LaTeX and Markdown formats."""
        self.latex_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.plot_data_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            ("table_01_embedding_quality", self.table_embedding_quality),
            ("table_02_robustness_overview", self.table_robustness_overview),
            ("table_03_per_attack_detailed", self.table_per_attack_detailed),
            ("table_04_content_sensitivity", self.table_content_sensitivity),
            ("table_05_delta_sweep", self.table_delta_sweep),
            ("table_06_capacity", self.table_capacity),
            ("table_07_false_positive", self.table_false_positive),
            ("table_08_wavelet_comparison", self.table_wavelet_comparison),
            ("table_09_repetition_coding", self.table_repetition_coding),
            ("table_10_tiled_vs_nontiled", self.table_tiled_vs_nontiled),
        ]

        all_latex = []
        all_markdown = []

        for filename, gen_fn in tables:
            latex, markdown = gen_fn()
            _write_file(self.latex_dir / f"{filename}.tex", latex)
            all_latex.append(f"% {filename}\n{latex}")
            all_markdown.append(f"## {filename.replace('_', ' ').title()}\n\n{markdown}")

        # Combined files
        _write_file(self.latex_dir / "all_tables.tex", "\n\n".join(all_latex))
        _write_file(
            self.markdown_dir / "full_report.md",
            "# Watermark Evaluation Report\n\n" + "\n\n---\n\n".join(all_markdown),
        )

        # Export plot data CSVs for pgfplots
        self._export_plot_data()

        print(f"Report generated in {self.output_dir}")

    # ------------------------------------------------------------------
    # Table 1: Embedding Quality
    # ------------------------------------------------------------------

    def table_embedding_quality(self) -> tuple[str, str]:
        """PSNR and SSIM per config (no attack). Thesis section: Imperceptibility."""
        rows = aggregate_embedding_quality(self.results, group_by="config_label")

        headers = ["Configuration", "PSNR (dB)", "SSIM", "N"]
        data = []
        for r in rows:
            data.append([
                _escape_latex(r.group_key),
                _fmt_metric(r.psnr),
                _fmt_metric(r.ssim, precision=4),
                str(r.n_total),
            ])

        latex = format_latex_table(
            headers, data,
            caption="Embedding quality: PSNR and SSIM between original and watermarked images (no attack).",
            label="tab:embedding_quality",
        )
        md = format_markdown_table(headers, data)
        return latex, md

    # ------------------------------------------------------------------
    # Table 2: Robustness Overview
    # ------------------------------------------------------------------

    def table_robustness_overview(self) -> tuple[str, str]:
        """Attack Category x Config -> Recovery Rate (%). Thesis section: Robustness Overview."""
        attacked = filter_results(self.results, exclude_no_attack=True)
        row_labels, col_labels, grid = pivot_table(
            attacked, row_key="attack_category", col_key="config_label",
            metric="recovery_rate",
        )

        headers = ["Attack Category"] + col_labels
        data = []
        for i, rk in enumerate(row_labels):
            row = [_escape_latex(rk)]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(f"{m.mean * 100:.1f}\\%" if m.n > 0 else "---")
            data.append(row)

        latex = format_latex_table(
            headers, data,
            caption="Recovery rate (\\%) per attack category and configuration.",
            label="tab:robustness_overview",
        )
        # Markdown version without LaTeX escaping
        md_data = []
        for i, rk in enumerate(row_labels):
            row = [rk]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(f"{m.mean * 100:.1f}%" if m.n > 0 else "---")
            md_data.append(row)
        md = format_markdown_table(["Attack Category"] + col_labels, md_data)
        return latex, md

    # ------------------------------------------------------------------
    # Table 3: Per-Attack Detailed
    # ------------------------------------------------------------------

    def table_per_attack_detailed(self) -> tuple[str, str]:
        """Per-attack BER, NC, PSNR, SSIM (mean +/- std). Thesis section: Detailed Robustness."""
        attacked = filter_results(self.results, exclude_no_attack=True)
        rows = aggregate_by(attacked, group_by="attack_name")

        headers = ["Attack", "BER", "NC", "PSNR (dB)", "SSIM", "Recovery %", "N"]
        data = []
        for r in rows:
            data.append([
                _escape_latex(r.group_key),
                _fmt_metric(r.ber, precision=4),
                _fmt_metric(r.nc, precision=3),
                _fmt_metric(r.psnr, precision=2),
                _fmt_metric(r.ssim, precision=4),
                f"{r.recovery_rate * 100:.1f}\\%",
                str(r.n_total),
            ])

        latex = format_latex_table(
            headers, data,
            caption="Detailed robustness metrics per attack (mean $\\pm$ std across all images and configs).",
            label="tab:per_attack_detailed",
        )
        md_data = []
        for r in rows:
            md_data.append([
                r.group_key,
                _fmt_metric_md(r.ber, precision=4),
                _fmt_metric_md(r.nc, precision=3),
                _fmt_metric_md(r.psnr, precision=2),
                _fmt_metric_md(r.ssim, precision=4),
                f"{r.recovery_rate * 100:.1f}%",
                str(r.n_total),
            ])
        md = format_markdown_table(
            ["Attack", "BER", "NC", "PSNR (dB)", "SSIM", "Recovery %", "N"],
            md_data,
        )
        return latex, md

    # ------------------------------------------------------------------
    # Table 4: Content Sensitivity
    # ------------------------------------------------------------------

    def table_content_sensitivity(self) -> tuple[str, str]:
        """Image Category x Attack Category -> mean BER. Thesis section: Content Sensitivity."""
        attacked = filter_results(self.results, exclude_no_attack=True)
        row_labels, col_labels, grid = pivot_table(
            attacked, row_key="image_category", col_key="attack_category",
            metric="ber_pre_ecc",
        )

        headers = ["Image Category"] + col_labels
        data = []
        for i, rk in enumerate(row_labels):
            row = [_escape_latex(rk)]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(_fmt_metric(m, precision=4) if m.n > 0 else "---")
            data.append(row)

        latex = format_latex_table(
            headers, data,
            caption="Mean BER by image category and attack category.",
            label="tab:content_sensitivity",
        )
        md_data = []
        for i, rk in enumerate(row_labels):
            row = [rk]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(_fmt_metric_md(m, precision=4) if m.n > 0 else "---")
            md_data.append(row)
        md = format_markdown_table(["Image Category"] + col_labels, md_data)
        return latex, md

    # ------------------------------------------------------------------
    # Table 5: Delta Sweep
    # ------------------------------------------------------------------

    def table_delta_sweep(self) -> tuple[str, str]:
        """Delta parameter sweep: PSNR vs BER trade-off. Thesis section: Parameter Optimization."""
        delta_configs = [r for r in self.results if r.config_label.startswith("delta_")]

        if not delta_configs:
            return _empty_table("No delta sweep results found.", "tab:delta_sweep")

        # Group by config_label for embedding quality
        embed_rows = aggregate_embedding_quality(delta_configs, group_by="config_label")

        # Group by config_label for robustness (attacked only)
        attacked = filter_results(delta_configs, exclude_no_attack=True)
        attack_rows = aggregate_by(attacked, group_by="config_label")

        # Merge by config_label
        attack_map = {r.group_key: r for r in attack_rows}

        headers = ["Delta", "PSNR (dB)", "SSIM", "Mean BER", "Recovery %"]
        data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            ber_str = _fmt_metric(ar.ber, precision=4) if ar else "---"
            rec_str = f"{ar.recovery_rate * 100:.1f}\\%" if ar else "---"
            data.append([
                _escape_latex(er.group_key),
                _fmt_metric(er.psnr, precision=2),
                _fmt_metric(er.ssim, precision=4),
                ber_str,
                rec_str,
            ])

        latex = format_latex_table(
            headers, data,
            caption="PSNR/SSIM vs. BER trade-off across delta values.",
            label="tab:delta_sweep",
        )
        md_data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            ber_str = _fmt_metric_md(ar.ber, precision=4) if ar else "---"
            rec_str = f"{ar.recovery_rate * 100:.1f}%" if ar else "---"
            md_data.append([
                er.group_key,
                _fmt_metric_md(er.psnr, precision=2),
                _fmt_metric_md(er.ssim, precision=4),
                ber_str,
                rec_str,
            ])
        md = format_markdown_table(
            ["Delta", "PSNR (dB)", "SSIM", "Mean BER", "Recovery %"],
            md_data,
        )
        return latex, md

    # ------------------------------------------------------------------
    # Table 6: Capacity Analysis
    # ------------------------------------------------------------------

    def table_capacity(self) -> tuple[str, str]:
        """Embedding capacity metrics. Thesis section: Embedding Capacity."""
        no_attack = [r for r in self.results if r.attack_name == "none"]

        # Group by config
        from collections import defaultdict
        groups: dict[str, list[EvalResult]] = defaultdict(list)
        for r in no_attack:
            groups[r.config_label].append(r)

        headers = ["Configuration", "Payload (bits)", "BPP", "Subband Util."]
        data = []
        for key in sorted(groups.keys()):
            group = groups[key]
            avg_bits = sum(r.payload_bits for r in group) / len(group)
            avg_bpp = sum(r.capacity_bpp for r in group) / len(group)
            # Compute subband utilization from first image's shape
            from evaluation.metrics import compute_subband_utilization
            sample = group[0]
            util = compute_subband_utilization(
                sample.payload_bits, (512, 512),  # corpus target size
            )
            data.append([
                _escape_latex(key),
                f"{avg_bits:.0f}",
                f"{avg_bpp:.6f}",
                f"{util:.4f}",
            ])

        latex = format_latex_table(
            headers, data,
            caption="Embedding capacity: payload size, bits per pixel, and subband utilization.",
            label="tab:capacity",
        )
        md = format_markdown_table(headers, data)
        return latex, md

    # ------------------------------------------------------------------
    # Table 7: False Positive Rate
    # ------------------------------------------------------------------

    def table_false_positive(self) -> tuple[str, str]:
        """False positive rate table. Thesis section: Security Analysis.

        Note: FPR is computed separately via `python -m evaluation fpr`.
        This generates a placeholder table that can be populated from fpr_analysis.csv.
        """
        headers = ["Image Size", "Trials", "False Positives", "FPR"]
        data = [
            ["512x512", "1000", "---", "---"],
        ]

        latex = format_latex_table(
            headers, data,
            caption="False positive rate: probability of successful RS decoding on unwatermarked images.",
            label="tab:false_positive",
        )
        md = format_markdown_table(headers, data)
        return latex, md

    def update_fpr_table(
        self,
        image_size: str,
        trials: int,
        false_positives: int,
        fpr: float,
    ) -> tuple[str, str]:
        """Generate FPR table with actual data."""
        headers = ["Image Size", "Trials", "False Positives", "FPR"]
        data = [[image_size, str(trials), str(false_positives), f"{fpr:.6f}"]]

        latex = format_latex_table(
            headers, data,
            caption="False positive rate: probability of successful RS decoding on unwatermarked images.",
            label="tab:false_positive",
        )
        md = format_markdown_table(headers, data)
        return latex, md

    # ------------------------------------------------------------------
    # Table 8: Wavelet Comparison
    # ------------------------------------------------------------------

    def table_wavelet_comparison(self) -> tuple[str, str]:
        """Haar vs db4 comparison. Thesis section: Algorithm Variants."""
        wavelet_results = [
            r for r in self.results
            if r.config_label.startswith("wavelet_")
        ]

        if not wavelet_results:
            return _empty_table("No wavelet comparison results.", "tab:wavelet_comparison")

        # Embedding quality
        embed_rows = aggregate_embedding_quality(wavelet_results, group_by="config_label")

        # Robustness
        attacked = filter_results(wavelet_results, exclude_no_attack=True)
        attack_rows = aggregate_by(attacked, group_by="config_label")
        attack_map = {r.group_key: r for r in attack_rows}

        headers = ["Wavelet", "PSNR (dB)", "SSIM", "Mean BER", "NC", "Recovery %"]
        data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            data.append([
                _escape_latex(er.group_key),
                _fmt_metric(er.psnr, precision=2),
                _fmt_metric(er.ssim, precision=4),
                _fmt_metric(ar.ber, precision=4) if ar else "---",
                _fmt_metric(ar.nc, precision=3) if ar else "---",
                f"{ar.recovery_rate * 100:.1f}\\%" if ar else "---",
            ])

        latex = format_latex_table(
            headers, data,
            caption="Wavelet basis comparison: Haar vs. Daubechies-4.",
            label="tab:wavelet_comparison",
        )
        md_data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            md_data.append([
                er.group_key,
                _fmt_metric_md(er.psnr, precision=2),
                _fmt_metric_md(er.ssim, precision=4),
                _fmt_metric_md(ar.ber, precision=4) if ar else "---",
                _fmt_metric_md(ar.nc, precision=3) if ar else "---",
                f"{ar.recovery_rate * 100:.1f}%" if ar else "---",
            ])
        md = format_markdown_table(
            ["Wavelet", "PSNR (dB)", "SSIM", "Mean BER", "NC", "Recovery %"],
            md_data,
        )
        return latex, md

    # ------------------------------------------------------------------
    # Table 9: Repetition Coding
    # ------------------------------------------------------------------

    def table_repetition_coding(self) -> tuple[str, str]:
        """R=1 vs R=3 comparison. Thesis section: Error Correction."""
        rep_results = [
            r for r in self.results
            if r.config_label.startswith("rep_")
        ]

        if not rep_results:
            return _empty_table("No repetition coding results.", "tab:repetition_coding")

        embed_rows = aggregate_embedding_quality(rep_results, group_by="config_label")
        attacked = filter_results(rep_results, exclude_no_attack=True)
        attack_rows = aggregate_by(attacked, group_by="config_label")
        attack_map = {r.group_key: r for r in attack_rows}

        headers = ["Repetitions", "Payload (bits)", "PSNR (dB)", "Mean BER", "Recovery %"]
        data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            # Get average payload bits
            no_attack = [r for r in rep_results if r.config_label == er.group_key and r.attack_name == "none"]
            avg_bits = sum(r.payload_bits for r in no_attack) / len(no_attack) if no_attack else 0
            data.append([
                _escape_latex(er.group_key),
                f"{avg_bits:.0f}",
                _fmt_metric(er.psnr, precision=2),
                _fmt_metric(ar.ber, precision=4) if ar else "---",
                f"{ar.recovery_rate * 100:.1f}\\%" if ar else "---",
            ])

        latex = format_latex_table(
            headers, data,
            caption="Effect of repetition coding on capacity and robustness.",
            label="tab:repetition_coding",
        )
        md_data = []
        for er in embed_rows:
            ar = attack_map.get(er.group_key)
            no_attack = [r for r in rep_results if r.config_label == er.group_key and r.attack_name == "none"]
            avg_bits = sum(r.payload_bits for r in no_attack) / len(no_attack) if no_attack else 0
            md_data.append([
                er.group_key,
                f"{avg_bits:.0f}",
                _fmt_metric_md(er.psnr, precision=2),
                _fmt_metric_md(ar.ber, precision=4) if ar else "---",
                f"{ar.recovery_rate * 100:.1f}%" if ar else "---",
            ])
        md = format_markdown_table(
            ["Repetitions", "Payload (bits)", "PSNR (dB)", "Mean BER", "Recovery %"],
            md_data,
        )
        return latex, md

    # ------------------------------------------------------------------
    # Table 10: Tiled vs Non-Tiled
    # ------------------------------------------------------------------

    def table_tiled_vs_nontiled(self) -> tuple[str, str]:
        """Tiled vs non-tiled for crop attacks. Thesis section: Geometric Resistance."""
        tiling_results = [
            r for r in self.results
            if r.config_label in ("no_tiling", "tiled_256")
        ]

        if not tiling_results:
            return _empty_table("No tiling comparison results.", "tab:tiled_vs_nontiled")

        # Focus on crop attacks
        crop_results = [r for r in tiling_results if r.attack_category == "cropping"]

        if not crop_results:
            # Fall back to all attacks
            crop_results = filter_results(tiling_results, exclude_no_attack=True)

        row_labels, col_labels, grid = pivot_table(
            crop_results, row_key="attack_name", col_key="config_label",
            metric="recovery_rate",
        )

        headers = ["Attack"] + col_labels
        data = []
        for i, rk in enumerate(row_labels):
            row = [_escape_latex(rk)]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(f"{m.mean * 100:.1f}\\%" if m.n > 0 else "---")
            data.append(row)

        latex = format_latex_table(
            headers, data,
            caption="Recovery rate comparison: tiled vs. non-tiled embedding under crop attacks.",
            label="tab:tiled_vs_nontiled",
        )
        md_data = []
        for i, rk in enumerate(row_labels):
            row = [rk]
            for j in range(len(col_labels)):
                m = grid[i][j]
                row.append(f"{m.mean * 100:.1f}%" if m.n > 0 else "---")
            md_data.append(row)
        md = format_markdown_table(["Attack"] + col_labels, md_data)
        return latex, md

    # ------------------------------------------------------------------
    # Plot data export
    # ------------------------------------------------------------------

    def _export_plot_data(self) -> None:
        """Export CSV data suitable for pgfplots."""
        # Delta vs PSNR/BER
        delta_results = [r for r in self.results if r.config_label.startswith("delta_")]
        if delta_results:
            embed_rows = aggregate_embedding_quality(delta_results, group_by="config_label")
            attacked = filter_results(delta_results, exclude_no_attack=True)
            attack_rows = aggregate_by(attacked, group_by="config_label")
            attack_map = {r.group_key: r for r in attack_rows}

            lines = ["delta,psnr_mean,psnr_std,ber_mean,ber_std"]
            for er in embed_rows:
                # Extract delta value from label
                delta_val = er.group_key.replace("delta_", "")
                ar = attack_map.get(er.group_key)
                ber_mean = ar.ber.mean if ar else 0.0
                ber_std = ar.ber.std if ar else 0.0
                lines.append(
                    f"{delta_val},{er.psnr.mean:.4f},{er.psnr.std:.4f},"
                    f"{ber_mean:.6f},{ber_std:.6f}"
                )
            _write_file(self.plot_data_dir / "delta_tradeoff.csv", "\n".join(lines))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_metric(m: AggregatedMetric, precision: int = 3) -> str:
    """Format metric as LaTeX: $mean \\pm std$."""
    if m.n == 0:
        return "---"
    if m.n == 1 or m.std == 0:
        return f"${m.mean:.{precision}f}$"
    return f"${m.mean:.{precision}f} \\pm {m.std:.{precision}f}$"


def _fmt_metric_md(m: AggregatedMetric, precision: int = 3) -> str:
    """Format metric for Markdown: mean +/- std."""
    if m.n == 0:
        return "---"
    if m.n == 1 or m.std == 0:
        return f"{m.mean:.{precision}f}"
    return f"{m.mean:.{precision}f} +/- {m.std:.{precision}f}"


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_latex_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
) -> str:
    """Generate a LaTeX table with booktabs formatting.

    Args:
        headers: Column headers.
        rows: 2D list of cell values (already formatted).
        caption: Table caption.
        label: LaTeX label for cross-referencing.

    Returns:
        Complete LaTeX table environment string.
    """
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def format_markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    """Generate a GitHub-flavored Markdown pipe table.

    Args:
        headers: Column headers.
        rows: 2D list of cell values.

    Returns:
        Markdown table string.
    """
    # Compute column widths
    all_rows = [headers] + rows
    widths = [
        max(len(str(row[i])) for row in all_rows)
        for i in range(len(headers))
    ]

    def fmt_row(row: list[str]) -> str:
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines = [fmt_row(headers)]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def _write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _empty_table(message: str, label: str) -> tuple[str, str]:
    """Return placeholder table for missing data."""
    latex = format_latex_table(
        ["Note"], [[message]], caption=message, label=label,
    )
    md = format_markdown_table(["Note"], [[message]])
    return latex, md
