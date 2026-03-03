"""CLI entry point for the evaluation framework.

Usage:
    python -m evaluation prepare   # Build and verify image corpus
    python -m evaluation run       # Run full evaluation sweep
    python -m evaluation report    # Generate tables from CSV
    python -m evaluation fpr       # Run false positive rate analysis
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_OUTPUT_DIR = "evaluation_output"
DEFAULT_FIXTURES_DIR = "tests/fixtures"


def cmd_prepare(args: argparse.Namespace) -> None:
    """Build and verify the image corpus."""
    from evaluation.image_corpus import ImageCorpus

    print("Building image corpus...")
    corpus = ImageCorpus.build(
        fixtures_dir=args.fixtures_dir,
        target_size=args.size,
    )

    print(f"\nCorpus: {len(corpus)} images")
    for cat in corpus.get_categories():
        images = corpus.filter_by_category(cat)
        names = ", ".join(sorted(images.keys()))
        print(f"  {cat}: {len(images)} images ({names})")

    print(f"\nAll images: {args.size}x{args.size} RGB uint8")
    print("Corpus ready.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation sweep."""
    from evaluation.configs import get_configs_by_name
    from evaluation.image_corpus import ImageCorpus
    from evaluation.runner import run_full_evaluation

    output_dir = Path(args.output_dir)

    print("Building image corpus...")
    corpus = ImageCorpus.build(
        fixtures_dir=args.fixtures_dir,
        target_size=args.size,
    )
    print(f"Corpus: {len(corpus)} images")

    configs = get_configs_by_name(args.configs)
    print(f"Configs: {len(configs)} ({args.configs} sweep)")

    # Parse stochastic seeds
    seeds = [int(s) for s in args.seeds.split(",")]

    t0 = time.perf_counter()
    run = run_full_evaluation(
        corpus, configs, output_dir,
        stochastic_seeds=seeds,
    )
    elapsed = time.perf_counter() - t0

    csv_path = output_dir / "results" / "full_evaluation.csv"
    print(f"\nDone: {len(run.results)} results in {elapsed:.1f}s")
    print(f"CSV: {csv_path}")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate tables from existing CSV."""
    from evaluation.report import ReportGenerator
    from evaluation.runner import EvalRun

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {csv_path}...")
    run = EvalRun.from_csv(csv_path)
    print(f"Loaded {len(run.results)} results")

    output_dir = Path(args.output_dir)
    gen = ReportGenerator(run, output_dir)
    gen.generate_all()


def cmd_fpr(args: argparse.Namespace) -> None:
    """Run false positive rate analysis."""
    from evaluation.metrics import compute_false_positive_rate
    from evaluation.report import ReportGenerator, _write_file
    from evaluation.runner import EvalRun

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = b"benchmark_secret_key_32bytes!!"
    from watermark.payload import derive_seed, encode_payload
    import numpy as np

    seed = derive_seed(key)

    # Compute expected payload size
    dummy_img = np.full((args.size, args.size, 3), 128, dtype=np.uint8)
    bits = encode_payload(
        "benchmark_artist", dummy_img, key,
        timestamp=1709337600, rs_nsym=args.rs_nsym,
        repetitions=args.repetitions,
    )
    num_bits = len(bits)

    print(f"Running FPR analysis: {args.trials} trials on {args.size}x{args.size} images")
    print(f"Payload: {num_bits} bits, delta={args.delta}, wavelet={args.wavelet}")

    t0 = time.perf_counter()
    fpr, n_fp = compute_false_positive_rate(
        num_trials=args.trials,
        image_shape=(args.size, args.size),
        num_bits=num_bits,
        delta=args.delta,
        seed=seed,
        key=key,
        rs_nsym=args.rs_nsym,
        repetitions=args.repetitions,
        wavelet=args.wavelet,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nResults ({elapsed:.1f}s):")
    print(f"  Trials:          {args.trials}")
    print(f"  False positives: {n_fp}")
    print(f"  FPR:             {fpr:.6f}")

    # Save CSV
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fpr_path = results_dir / "fpr_analysis.csv"
    _write_file(fpr_path, f"image_size,trials,false_positives,fpr\n{args.size}x{args.size},{args.trials},{n_fp},{fpr:.6f}\n")
    print(f"  Saved: {fpr_path}")

    # Generate updated table if main CSV exists
    main_csv = results_dir / "full_evaluation.csv"
    if main_csv.exists():
        run = EvalRun.from_csv(main_csv)
        gen = ReportGenerator(run, output_dir)
        latex, md = gen.update_fpr_table(
            f"{args.size}x{args.size}", args.trials, n_fp, fpr,
        )
        latex_dir = output_dir / "reports" / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)
        _write_file(latex_dir / "table_07_false_positive.tex", latex)
        print(f"  Updated: {latex_dir / 'table_07_false_positive.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Thesis-grade watermark evaluation framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Build and verify image corpus")
    p_prepare.add_argument("--fixtures-dir", default=DEFAULT_FIXTURES_DIR)
    p_prepare.add_argument("--size", type=int, default=512)

    # run
    p_run = subparsers.add_parser("run", help="Run evaluation sweep")
    p_run.add_argument("--configs", default="full",
                       choices=["baseline", "delta", "wavelet", "repetition",
                                "tiling", "adaptive", "full"])
    p_run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_run.add_argument("--fixtures-dir", default=DEFAULT_FIXTURES_DIR)
    p_run.add_argument("--size", type=int, default=512)
    p_run.add_argument("--seeds", default="0,1,2",
                       help="Comma-separated seeds for stochastic attacks")

    # report
    p_report = subparsers.add_parser("report", help="Generate tables from CSV")
    p_report.add_argument("--csv", default=f"{DEFAULT_OUTPUT_DIR}/results/full_evaluation.csv")
    p_report.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # fpr
    p_fpr = subparsers.add_parser("fpr", help="Run false positive rate analysis")
    p_fpr.add_argument("--trials", type=int, default=1000)
    p_fpr.add_argument("--size", type=int, default=512)
    p_fpr.add_argument("--delta", type=float, default=60.0)
    p_fpr.add_argument("--wavelet", default="haar")
    p_fpr.add_argument("--rs-nsym", type=int, default=128)
    p_fpr.add_argument("--repetitions", type=int, default=1)
    p_fpr.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    dispatch = {
        "prepare": cmd_prepare,
        "run": cmd_run,
        "report": cmd_report,
        "fpr": cmd_fpr,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
