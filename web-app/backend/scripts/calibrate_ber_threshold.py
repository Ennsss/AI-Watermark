"""Evaluate BER operating points from explicitly labeled calibration evidence."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def fmt(value):
    return "N/A" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1] / "calibration"
    parser.add_argument("--input", type=Path, default=base / "verification_threshold_audit.csv")
    parser.add_argument("--report", type=Path, default=base / "ber_threshold_calibration_report.csv")
    parser.add_argument("--summary", type=Path, default=base / "ber_threshold_calibration_summary.md")
    parser.add_argument("--minimum-specificity", type=float, default=0.95)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row.get("ground_truth") in {"positive", "negative"} and (row.get("recomputed_ber") or row.get("ber")) not in {"", None}]
    if not rows:
        raise SystemExit("No labeled positive/negative rows. Label evidence before selecting a threshold.")

    positives = [float(row.get("recomputed_ber") or row["ber"]) for row in rows if row["ground_truth"] == "positive"]
    negatives = [float(row.get("recomputed_ber") or row["ber"]) for row in rows if row["ground_truth"] == "negative"]
    condition_counts = Counter((row["ground_truth"], row.get("condition") or "unspecified") for row in rows)
    condition_values = defaultdict(list)
    for row in rows:
        condition_values[(row["ground_truth"], row.get("condition") or "unspecified")].append(float(row.get("recomputed_ber") or row["ber"]))
    metrics = []
    for bits in range(65):
        threshold = bits / 128
        tp = sum(value <= threshold for value in positives)
        fn = len(positives) - tp
        fp = sum(value <= threshold for value in negatives)
        tn = len(negatives) - fp
        recall = ratio(tp, tp + fn)
        specificity = ratio(tn, tn + fp)
        precision = ratio(tp, tp + fp)
        accuracy = ratio(tp + tn, len(rows))
        balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        metrics.append({
            "threshold_bits": bits, "threshold": threshold, "true_positives": tp,
            "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
            "sensitivity_recall": recall, "specificity": specificity, "precision": precision,
            "false_positive_rate": None if specificity is None else 1 - specificity,
            "false_negative_rate": None if recall is None else 1 - recall,
            "accuracy": accuracy, "balanced_accuracy": balanced, "f1": f1,
            "youden_j": None if recall is None or specificity is None else recall + specificity - 1,
        })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    eligible = [m for m in metrics if m["specificity"] is not None and m["specificity"] >= args.minimum_specificity]
    candidate = max(eligible, key=lambda m: (m["sensitivity_recall"] or -1, m["balanced_accuracy"] or -1, -m["threshold"])) if eligible else None
    overlap_low = max(min(positives), min(negatives)) if positives and negatives else None
    overlap_high = min(max(positives), max(negatives)) if positives and negatives else None
    overlap = overlap_low <= overlap_high if overlap_low is not None else False
    lines = [
        "# BER Threshold Calibration Summary", "",
        "> Candidate analysis only. A human-approved labeled dataset is required before changing the committed policy.", "",
        f"- Labeled positives: {len(positives)}", f"- Labeled negatives: {len(negatives)}",
        f"- Positive BER range: {fmt(min(positives) if positives else None)} to {fmt(max(positives) if positives else None)}",
        f"- Negative BER range: {fmt(min(negatives) if negatives else None)} to {fmt(max(negatives) if negatives else None)}",
        f"- Distribution overlap: {'yes' if overlap else 'no'}", "",
        "## Counts by condition", "",
    ]
    for (truth, condition), count in sorted(condition_counts.items()):
        values = condition_values[(truth, condition)]
        lines.append(f"- {truth} / {condition}: n={count}, min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
    lines.extend(["", f"## Candidate at minimum specificity {args.minimum_specificity:.2f}", ""])
    if candidate:
        lines.extend([
            f"- Threshold: {candidate['threshold_bits']}/128 = {candidate['threshold']:.6f}",
            f"- Sensitivity: {fmt(candidate['sensitivity_recall'])}",
            f"- Specificity: {fmt(candidate['specificity'])}",
            f"- False positives: {candidate['false_positives']}",
            f"- False negatives: {candidate['false_negatives']}",
        ])
    else:
        lines.append("No candidate meets the requested specificity with the supplied labels.")
    lines.extend(["", "This script does not modify application policy or historical rows."])
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluated {len(metrics)} thresholds from {len(rows)} labeled rows")
    print(f"Wrote {args.report.resolve()} and {args.summary.resolve()}")


if __name__ == "__main__":
    main()
