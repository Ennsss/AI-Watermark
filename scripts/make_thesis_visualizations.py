"""Generate thesis/defense SVG visuals from the frozen held-out benchmark.

Run from the repository root:
    python scripts/make_thesis_visualizations.py

Only Python's standard library is required. Outputs are editable SVG files in
visualizations/. The plotted values are read from experiments/final_heldout_benchmark.
"""

from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "final_heldout_benchmark"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

NAVY = "#17324D"
BLUE = "#3B82F6"
ORANGE = "#F59E0B"
TEAL = "#0F766E"
GRAY = "#64748B"
LIGHT = "#E8EEF4"
RED = "#B91C1C"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def svg_document(title: str, body: str, width: int = 1200, height: int = 700) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>{html.escape(title)}</title><rect width="100%" height="100%" fill="#FFFFFF"/>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:{NAVY}}}.title{{font-size:30px;font-weight:700}}.sub{{font-size:16px;fill:{GRAY}}}.axis{{font-size:14px;fill:{GRAY}}}.label{{font-size:16px}}.small{{font-size:13px;fill:{GRAY}}}.value{{font-size:14px;font-weight:700}}</style>
{body}</svg>'''


def save(name: str, title: str, body: str, width: int = 1200, height: int = 700) -> None:
    (OUT / name).write_text(svg_document(title, body, width, height), encoding="utf-8")


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start", extra: str = "") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{html.escape(value)}</text>'


def display_condition(condition: str) -> str:
    labels = {
        "jpeg90": "JPEG90", "jpeg70": "JPEG70", "jpeg50": "JPEG50",
        "resize75": "Resize75", "resize50": "Resize50", "resize25": "Resize25",
        "crop_mild": "Crop mild", "crop_moderate": "Crop mod.", "crop_severe": "Crop severe",
        "reencode1": "JPEG Q85 ×1", "reencode2": "JPEG Q85 ×2", "reencode3": "JPEG Q85 ×3",
        "clean": "Clean",
    }
    return labels.get(condition, condition.replace("_", " "))


def format_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.2e}".split("e")
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa} × 10{str(int(exponent)).translate(superscripts)}"


def framework_diagram() -> None:
    boxes = [
        (45, 190, 190, 120, "Illustration", "512 × 512 RGB"),
        (285, 190, 190, 120, "Embed payload", "Y channel → 2-level Haar DWT\nLH2 / HL2 → QIM, Δ=24"),
        (525, 190, 190, 120, "Degradation", "JPEG · resize · crop\nRepeated JPEG Q85"),
        (765, 100, 190, 120, "Classical path", "DWT + analytical\nQIM bit decision"),
        (765, 280, 190, 120, "CNN-assisted path", "Same selected coefficients\nCNN bit decision · 369 params"),
        (1005, 190, 160, 120, "Compare", "BER · perfect recovery\nFidelity · compute"),
    ]
    body = [text(50, 55, "Experimental framework", "title"), text(50, 84, "The extractor decision is the comparison; embedding and attacked inputs are shared.", "sub")]
    for x, y, w, h, title, desc in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#F4F8FB" stroke="{BLUE if "CNN" in title else NAVY}" stroke-width="2"/>')
        body.append(text(x + w / 2, y + 37, title, "label", "middle", 'font-weight="700"'))
        for j, line in enumerate(desc.split("\n")):
            body.append(text(x + w / 2, y + 69 + 22*j, line, "small", "middle"))
    arrows = [(235, 250, 285, 250), (475,250,525,250), (715,235,765,160), (715,265,765,340), (955,160,1005,225), (955,340,1005,275)]
    for x1,y1,x2,y2 in arrows:
        body.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{GRAY}" stroke-width="3" marker-end="url(#arrow)"/>')
    body.insert(0, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748B"/></marker></defs>')
    body.append(text(50, 520, "Held-out comparison: 500 test images × 13 conditions = 6,500 image-condition pairs", "label"))
    body.append(text(50, 555, "Degradations are evaluated separately; clean plus 12 attack conditions.", "small"))
    save("01_framework.svg", "Experimental framework", "\n".join(body))


def dataset_diagram() -> None:
    body = [text(55, 55, "Dataset and evaluation split", "title"), text(55, 84, "Corpus partitions and the smaller CNN development subsets", "sub")]
    cols = [(100, "Training pool", "10,000 images", "CNN subset: 500 images\n2 payloads/image → 1,000 pairs", BLUE),
            (450, "Validation pool", "1,000 images", "CNN subset: 100 images\n2 payloads/image → 200 pairs", TEAL),
            (800, "Held-out test", "500 images", "Final benchmark only\n500 × 13 = 6,500 pairs", ORANGE)]
    for x, title, n, detail, color in cols:
        body.append(f'<rect x="{x}" y="170" width="260" height="250" rx="18" fill="#F8FAFC" stroke="{color}" stroke-width="3"/>')
        body.append(text(x+130, 225, title, "label", "middle", 'font-weight="700"'))
        body.append(text(x+130, 285, n, "title", "middle"))
        for i, line in enumerate(detail.split("\n")):
            body.append(text(x+130, 340+i*25, line, "small", "middle"))
    body.append(text(55, 485, "The development subsets were selected by sorted filename; they were not random or style-stratified.", "label"))
    body.append(text(55, 520, "The 500-image test partition was held out from CNN training, validation, and Δ calibration.", "small"))
    save("02_dataset_split.svg", "Dataset and evaluation split", "\n".join(body))


def bar_chart(rows: list[dict[str, str]], field_a: str, field_b: str, title: str, subtitle: str,
              filename: str, y_max: float, percent: bool = True, significant: set[str] | None = None,
              ber_axis: bool = False, value_labels: bool = False, note: str | None = None) -> None:
    labels = [display_condition(r["condition"]) for r in rows]
    left, right, top, bottom = 105, 1165, 145, 555
    chart_h = bottom-top
    body = [text(55, 82, subtitle, "sub")]
    # grid and y labels
    tick_values = [i / 10 for i in range(6)] if ber_axis else [y_max * i / 5 for i in range(6)]
    for val in tick_values:
        y = bottom - chart_h * val / y_max
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{LIGHT}"/>')
        lab = f"{val*100:.0f}%" if percent else f"{val:.1f}"
        body.append(text(left-15, y+5, lab, "axis", "end"))
    if ber_axis:
        random_y = bottom - chart_h * .5 / y_max
        body.append(f'<line x1="{left}" y1="{random_y:.1f}" x2="{right}" y2="{random_y:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="9 7"/>')
        body.append(text(right-8, random_y-9, "50% random guessing", "small", "end", f'fill="{RED}"'))
        body.append(text(28, (top+bottom)/2, "Mean BER", "axis", "middle", f'transform="rotate(-90 28 {(top+bottom)/2})"'))
    group_w = (right-left)/len(rows)
    bw = min(29, group_w*.30)
    for i, (row, label) in enumerate(zip(rows, labels)):
        cx = left + group_w*(i+.5)
        for offset, key, color in [(-bw*.65, field_a, BLUE), (bw*.65, field_b, ORANGE)]:
            val=float(row[key]); h=chart_h*val/y_max; x=cx+offset-bw/2; y=bottom-h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        body.append(text(cx, bottom+25, label, "small", "middle", 'font-size="12px"'))
        if value_labels:
            for offset, key in [(-bw*.65, field_a), (bw*.65, field_b)]:
                val=float(row[key]); h=chart_h*val/y_max
                body.append(text(cx+offset, max(top+3, bottom-h-7), f"{val*100:.1f}%", "small", "middle", 'font-size="12px" fill="#17324D"'))
        if significant and row["condition"] in significant:
            body.append(text(cx, top-10, "*", "value", "middle", f'fill="{TEAL}"'))
    body.append(f'<rect x="435" y="620" width="18" height="18" fill="{BLUE}"/>')
    body.append(text(460, 635, "Classical QIM", "small"))
    body.append(f'<rect x="590" y="620" width="18" height="18" fill="{ORANGE}"/>')
    body.append(text(615, 635, "CNN-assisted", "small"))
    if significant is not None:
        body.append(text(790, 635, "* CNN advantage significant after Holm correction", "small"))
    if note:
        for j, line in enumerate(note.split("\n")):
            body.append(text(55, 665+20*j, line, "small"))
    save(filename, title, "\n".join(body), height=720 if note and "\n" in note else 700)


def results_charts() -> None:
    rows = read_csv("per_condition_summary.csv")
    attacks = [r for r in rows if r["condition"] != "clean"]
    tests = {r["condition"] for r in read_csv("statistical_tests.csv") if r["reject_at_0.05"].lower() == "true"}
    classical_macro = statistics.mean(float(r["classical_mean"]) for r in attacks)
    cnn_macro = statistics.mean(float(r["cnn_mean"]) for r in attacks)
    relative = (classical_macro-cnn_macro)/classical_macro*100
    bar_chart(attacks, "classical_mean", "cnn_mean", "BER by degradation condition",
              f"Attack-only macro BER: {classical_macro:.4f} → {cnn_macro:.4f} ({relative:.2f}% relative reduction); n=500/condition.",
              "03_ber_by_attack.svg", .55, significant=tests, ber_axis=True)
    selected = [r for r in rows if r["condition"] in {"clean", "jpeg90", "reencode1", "reencode2", "reencode3"}]
    bar_chart(selected, "classical_perfect_rate", "cnn_perfect_rate", "Perfect 128-bit payload recovery",
              "Share of 500 test images with BER = 0; selected conditions.",
              "04_perfect_payload_recovery.svg", 1.0, value_labels=True,
              note="Clean: Classical 284/500 (56.8%) vs CNN 488/500 (97.6%).\nOther 8 attack conditions: 0% perfect recovery for both methods.")
    statistical_summary()


def statistical_summary() -> None:
    rows = read_csv("statistical_tests.csv")
    overall = next(r for r in rows if r["condition"] == "attack_macro_by_image")
    conditions = [r for r in rows if r["condition"] != "attack_macro_by_image"]
    body = [text(55, 84, "Wilcoxon signed-rank tests on paired image-level BER; Holm correction across attack conditions.", "sub")]
    body.append(f'<rect x="60" y="120" width="1080" height="95" rx="14" fill="#F4F8FB" stroke="{TEAL}" stroke-width="2"/>')
    body.append(text(90, 158, "Aggregate attack-only comparison", "label", extra='font-weight="700"'))
    body.append(text(90, 190, f"n = {overall['n']} images · p = {format_p(float(overall['raw_p_value']))} · rank-biserial r = {float(overall['rank_biserial']):.3f} · mean BER improvement = {float(overall['mean_classical_minus_cnn_ber']):.4f}", "small"))
    headers = [(90,"Condition"),(390,"Raw p"),(610,"Holm-adjusted p"),(870,"Effect size r"),(1000,"Result")]
    for x, label in headers: body.append(text(x, 265, label, "label", "start", 'font-weight="700"'))
    y=305
    for r in conditions:
        is_sig=r["reject_at_0.05"].lower()=="true"
        if r["condition"] == "jpeg70":
            body.append(f'<rect x="75" y="{y-20}" width="1060" height="28" rx="5" fill="#FFF4D6"/>')
        weight = 'font-weight="700"' if r["condition"] == "jpeg70" else ''
        body.append(f'<line x1="75" y1="{y+17}" x2="1135" y2="{y+17}" stroke="{LIGHT}"/>')
        body.append(text(90,y,display_condition(r["condition"]),"small",extra=weight))
        body.append(text(390,y,format_p(float(r["raw_p_value"])),"small",extra=weight))
        body.append(text(610,y,format_p(float(r["holm_adjusted_p_value"])),"small",extra=weight))
        body.append(text(870,y,f"{float(r['rank_biserial']):+.3f}","small",extra=weight))
        body.append(text(1000,y,"Significant" if is_sig else "Not significant","small",extra=f'fill="{TEAL if is_sig else GRAY}" {weight}'))
        y+=37
    body.append(text(75, 770, "Negative r = leans toward classical.", "small"))
    save("06_statistical_summary.svg", "Paired statistical comparison", "\n".join(body), height=800)


def compute_summary() -> dict[str, dict[str, float]]:
    rows = read_csv("computational_metrics.csv")
    summary: dict[str, dict[str, float]] = {}
    for decoder in ("classical", "cnn"):
        group = [r for r in rows if r["decoder"] == decoder and r["boundary"] == "end_to_end"]
        summary[decoder] = {
            "time_ms": statistics.mean(float(r["wall_time_ns_median"]) for r in group)/1e6,
            "memory_mb": statistics.mean(float(r["peak_python_bytes_median"]) for r in group)/1e6,
            "cpu": statistics.mean(float(r["cpu_utilization_percent"]) for r in group),
        }
    return summary


def fidelity_compute() -> None:
    fidelity = json.loads((DATA / "fidelity_summary.json").read_text(encoding="utf-8"))["summary"]
    comp = compute_summary()
    body = [text(55, 86, "Held-out benchmark summary; fidelity measures compare original and clean watermarked images.", "sub")]
    cards = [
        (70, "Mean PSNR", f"{fidelity['mean_psnr_rgb_db']:.2f} dB", "Higher indicates lower pixel error", BLUE),
        (350, "Mean SSIM", f"{fidelity['mean_ssim_rgb']:.6f}", "Closer to 1 indicates higher similarity", TEAL),
        (630, "Latency", f"{comp['cnn']['time_ms']/comp['classical']['time_ms']:.2f}× slower", "Per-image Keras calls; our implementation", ORANGE),
        (910, "Model size", "369 parameters", "CNN-assisted decoder", NAVY),
    ]
    for x, title, value, note, color in cards:
        body.append(f'<rect x="{x}" y="155" width="245" height="155" rx="16" fill="#F8FAFC" stroke="{color}" stroke-width="3"/>')
        body.append(text(x+122, 195, title, "small", "middle"))
        body.append(text(x+122, 245, value, "label", "middle", 'font-size="21" font-weight="700"'))
        body.append(text(x+122, 280, note, "small", "middle"))
    max_time=max(comp["classical"]["time_ms"],comp["cnn"]["time_ms"])*1.15
    for i, decoder in enumerate(("classical", "cnn")):
        y=400+i*70; color=BLUE if decoder=="classical" else ORANGE
        body.append(text(100,y+22,"CNN" if decoder=="cnn" else "Classical", "label"))
        w=650*comp[decoder]["time_ms"]/max_time
        body.append(f'<rect x="260" y="{y}" width="{w:.1f}" height="32" rx="6" fill="{color}"/>')
        body.append(text(260+w+12,y+22,f"{comp[decoder]['time_ms']:.2f} ms", "value"))
    body.append(text(100, 570, f"Mean peak Python allocation: {comp['classical']['memory_mb']:.2f} MB classical; {comp['cnn']['memory_mb']:.2f} MB CNN.", "small"))
    body.append(text(100, 605, f"Mean recorded CPU utilization: {comp['classical']['cpu']:.1f}% classical; {comp['cnn']['cpu']:.1f}% CNN. Interpret cautiously for short operations.", "small"))
    save("05_fidelity_and_computation.svg", "Fidelity and computational measurements", "\n".join(body))


def main() -> None:
    framework_diagram()
    dataset_diagram()
    results_charts()
    fidelity_compute()
    print(f"Wrote six SVG visuals to {OUT}")


if __name__ == "__main__":
    main()
