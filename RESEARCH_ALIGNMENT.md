# Research Alignment Notes

This file tracks how the workspace now maps to the current study context:

**DWT-Based Classical-Neural Watermarking for Digital Art Provenance in Social Media-Degraded Images**

## Current Status

The repository is now aligned enough to begin **classical baseline testing and
delta calibration**. The full final research pipeline is not complete yet
because the final dataset/preprocessing handoff, CNN training data generation,
CNN model training, paired classical-vs-CNN evaluation, and final statistical
tables are still pending.

Local verification reported by Canard:

```text
117 tests passed
0 failed
0 errors
```

## Main Defaults Now Aligned

- Main payload is a deterministic raw 128-bit bitstream.
- AES-256, Reed-Solomon ECC, repetition coding, adaptive masking, and tiling are optional/legacy rather than default experiment behavior.
- Main degradation suite is limited to JPEG, resize, crop severity bands, and repeated re-encoding.
- DWT boundary handling is explicitly set to symmetric mode in the core DWT helpers.
- Benchmark defaults use raw BER rather than payload decode success.
- CNN extraction utilities now prepare LH2/HL2 transform-domain inputs shaped `128 x 128 x 2`.
- A shallow optional CNN decoder builder and training helper have been added.
- Paired statistical utilities have been added for Wilcoxon signed-rank, rank-biserial effect size, and Holm-Bonferroni correction.
- README now describes the current research scope and optional dependency sets.
- `docs/experiment_log.md` now exists for recording Chapter 4 benchmark, calibration, training, and evaluation notes.

## What Is Ready To Test Now

The following are ready for local testing:

- Unit/integration tests:

```powershell
python -m pytest tests/ -q
```

- Classical baseline benchmark on fixture images:

```powershell
python -m cli.main benchmark tests/fixtures --delta 16 --wavelet haar -o baseline_results.csv
```

- Delta calibration on fixture images:

```powershell
python -m cli.main benchmark tests/fixtures --delta 24 --wavelet haar -o baseline_delta24.csv
python -m cli.main benchmark tests/fixtures --delta 32 --wavelet haar -o baseline_delta32.csv
```

- Structured classical evaluation runner:

```powershell
python -m evaluation run --configs baseline --output-dir evaluation_output
```

Fixture benchmarks are only sanity checks. They should not be treated as final
paper evidence.

## What To Watch During Baseline Testing

Record each meaningful run in `docs/experiment_log.md`.

Watch for:

- No-attack BER should be `0.0000`. If not, the clean embed/extract path is broken.
- JPEG Q90 should usually be stronger than Q70/Q50.
- Crop attacks may be near random BER because the main baseline is not using tiled/synchronized extraction.
- Line-art or sparse images may perform poorly because the main experiment forces LH2/HL2 and does not use the old LL2 fallback.
- Higher delta should reduce BER but may reduce visual quality.
- SSIM/PSNR should be reported as embedding/degradation fidelity, not as something improved by CNN extraction.
- `[FAIL]` in the CLI currently means strict perfect raw payload recovery. Small BER values such as `0.0078` are still useful research observations.

## Dependency Setup By Role

Most members:

```powershell
pip install -r requirements.txt
pip install -e ".[dev]"
```

CNN member:

```powershell
pip install -e ".[dev,ml]"
```

Evaluation/statistics member:

```powershell
pip install -e ".[dev,stats]"
```

CNN plus final evaluation lead:

```powershell
pip install -e ".[dev,ml,stats]"
```

TensorFlow is only needed for CNN training/extraction. SciPy is only needed for
statistical analysis.

## Current Baseline Sanity Result

A fixture benchmark was run with:

```powershell
python -m cli.main benchmark tests/fixtures --delta 16 --wavelet haar -o baseline_results.csv
```

Summary:

```text
Mean BER: 0.2564
Mean SSIM: 0.8871
Strict perfect-recovery rate: 17.6%
```

Interpretation:

- Clean/no-attack extraction worked for all fixture images.
- Delta 16 appears weak for JPEG Q70/Q50, severe resizing, crop, and sparse line art.
- Crop weakness is expected for the non-tiled main baseline.
- Next step is delta calibration with 24 and 32 before selecting a fixed delta.

## Intentionally Not Changed

- The preprocessing/data acquisition module was not refactored because dataset preprocessing is assigned separately.
- Danbooru/Safebooru API access and the final `10,000 / 1,000 / 500` split are still not implemented.
- Legacy tests and modules for AES/ECC, masking, tiling, and extra attacks remain available for optional or future work.

## Still Needed For The Full Final Pipeline

- Dataset/preprocessing member completes the paper-aligned dataset pipeline.
- Final corpus is normalized to the agreed `512 x 512` contract.
- Delta candidates `8`, `16`, `24`, and `32` are calibrated, then one selected delta is frozen.
- CNN training examples are generated from degraded LH2/HL2 maps and original 128-bit payload labels.
- CNN decoder is trained with validation monitoring and early stopping.
- Classical and CNN extraction are evaluated on the same held-out images, payloads, coefficient locations, and degradation instances.
- Final CSV includes method, BER, SSIM/PSNR, timing, hardware/config, and resource measurements if collected.
- Wilcoxon signed-rank and effect-size analysis are run on paired BER results.
- Chapter 4 tables/figures are produced from final held-out test results.

## Chapter 4 Notes To Keep

For every calibration, training, or evaluation run, log:

- Date and commit/version.
- Dataset and split.
- Delta and payload/coefficient seeds.
- Attack settings.
- Command run.
- Output CSV/model/checkpoint path.
- Mean/median BER by attack and method.
- SSIM/PSNR.
- Timing and hardware.
- Observed weak cases.
- Decision made from the run.

## Main Config

See `configs/main_experiment.yaml` for the current paper-aligned experiment settings.
