# DWT-Based Classical-Neural Watermarking for Digital Art Provenance in Social Media-Degraded Images

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

## Overview

This repository implements a research pipeline for comparing classical
DWT-QIM extraction with CNN-assisted bit prediction for invisible watermark
recovery in non-photorealistic digital illustrations.

The main experiment embeds a fixed 128-bit ownership payload into the LH2 and
HL2 subbands of the Y luminance channel using two-level DWT and binary QIM.
The embedding pipeline remains classical. The CNN, when enabled, operates only
during extraction and predicts the embedded bitstream from degraded LH2/HL2
coefficient maps.

This is not a deployed copyright enforcement tool, anti-scraping system,
platform DRM mechanism, or full anti-AI-training protection system. It is a
controlled experiment for measuring raw watermark bit recovery under selected
social-media-like degradations.

## Main Experiment

Classical embedding:

```text
RGB image
-> YCbCr
-> Y channel
-> two-level DWT, Haar, symmetric mode
-> deterministic LH2/HL2 coefficient selection
-> binary QIM embedding of fixed 128-bit payload
-> inverse DWT
-> recombine Y with Cb/Cr
-> RGB watermarked image
```

Classical extraction:

```text
Degraded watermarked RGB image
-> YCbCr
-> Y channel
-> two-level DWT
-> same LH2/HL2 coefficient locations
-> QIM grid decision
-> recovered 128-bit payload
-> BER
```

CNN-assisted extraction:

```text
Degraded watermarked RGB image
-> YCbCr
-> Y channel
-> two-level DWT
-> stack LH2 and HL2 as 128 x 128 x 2 tensor
-> shallow CNN decoder
-> predicted 128-bit payload
-> BER
```

Only the extraction bit-decision stage changes between the classical and CNN
branches.

## Paper-Aligned Defaults

- Image size: `512 x 512` expected by the main experiment data pipeline
- Wavelet: Haar
- DWT level: 2
- DWT boundary mode: symmetric
- Target subbands: LH2 and HL2
- Payload: fixed raw 128-bit ownership payload
- Delta calibration candidates: `8`, `16`, `24`, `32`
- Main degradation suite:
  - JPEG: QF `90`, `70`, `50`
  - Resize: `75%`, `50%`, `25%`
  - Crop: mild `5-10%`, moderate `20-30%`, severe `40-50%`
  - Re-encoding: `1x`, `2x`, `3x`

The current main experiment config is in
[`configs/main_experiment.yaml`](configs/main_experiment.yaml).

## Optional Or Legacy Modules

The repo still contains useful older modules, but they are not part of the
main controlled comparison unless explicitly enabled:

- AES-256 payload encryption
- Reed-Solomon ECC
- repetition coding
- adaptive perceptual masking
- tiled embedding and synchronization
- Gaussian noise attacks
- screenshot simulation
- combined attack chains
- false positive analysis
- broad parameter sweeps

These features can support future work or ablation studies, but main reported
BER should be raw 128-bit recovery without ECC, encryption, or checksum
correction.

## Project Structure

```text
src/
  watermark/
    preprocessor.py       RGB/YCbCr helpers, Y-channel extraction, padding
    payload.py            Main raw payload helper plus optional legacy payload tools
    embedding.py          DWT decomposition, QIM embedding, classical extraction
    extraction.py         Classical extraction wrapper and BER
    cnn_extraction.py     LH2/HL2 CNN input prep and bit prediction
    train_cnn.py          Optional CNN training helper
    models/
      cnn_decoder.py      Shallow baseline CNN decoder
    masking.py            Optional adaptive masking
    tiling.py             Optional tiled embedding
    sync.py               Optional crop synchronization
  attacks/
    suite.py              Main degradations plus optional stress attacks
  benchmark/
    runner.py             Classical raw-BER benchmark
  evaluation/
    configs.py            Main and optional evaluation configs
    runner.py             Classical extraction evaluation runner
    statistical_analysis.py Wilcoxon and Holm-Bonferroni helpers
```

## Usage

Install the classical pipeline:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Embed a raw 128-bit payload:

```bash
python -m cli.main embed artwork.png \
  --delta 16 \
  --payload-bits 128 \
  --payload-seed 42 \
  --coefficient-seed 42 \
  -o artwork_watermarked.png
```

Extract raw bits with the classical branch:

```bash
python -m cli.main extract artwork_watermarked.png \
  --num-bits 128 \
  --delta 16 \
  --coefficient-seed 42
```

Run the classical baseline benchmark:

```bash
python -m cli.main benchmark tests/fixtures \
  --delta 16 \
  -o results.csv
```

Run the evaluation CLI:

```bash
python -m evaluation run \
  --configs baseline \
  --output-dir evaluation_output
```

## CNN Branch

CNN support is intentionally optional. Install TensorFlow only when training or
running the CNN-assisted extractor:

```bash
pip install -e ".[ml]"
```

The baseline decoder is defined in
[`src/watermark/models/cnn_decoder.py`](src/watermark/models/cnn_decoder.py).
CNN inputs are prepared by
[`src/watermark/cnn_extraction.py`](src/watermark/cnn_extraction.py) as
`128 x 128 x 2` LH2/HL2 tensors for `512 x 512` images.

## Metrics

Primary metric:

- BER: Hamming distance between recovered bits and original embedded bits,
  divided by payload length.

Supporting metrics:

- SSIM
- PSNR
- inference time
- memory usage and CPU utilization when collected by the evaluation harness

SSIM/PSNR evaluate visual fidelity of the fixed classical embedding. The CNN
does not improve SSIM directly because it does not change the embedder.

## Optional Dependency Sets

Not every team member needs every dependency:

- `.[dev]` is for running tests and normal development.
- `.[ml]` is only needed for CNN training or CNN-assisted extraction.
- `.[stats]` is only needed for Wilcoxon/effect-size analysis.

Examples:

```bash
pip install -e ".[dev]"
pip install -e ".[dev,ml]"
pip install -e ".[dev,stats]"
pip install -e ".[dev,ml,stats]"
```

The CNN/evaluation lead can install all extras. Teammates working only on the
classical baseline, preprocessing handoff, or documentation do not need the ML
stack.

## Experiment Logging

Use [`docs/experiment_log.md`](docs/experiment_log.md) to record benchmark,
calibration, CNN training, and final evaluation runs. These notes should feed
Chapter 4 results and discussion.

## Current Thesis Status

- Clean seed-aware decoder validated
- Stage 2 JPEG/re-encoding robustness: partial
- Stage 2B full-factorial exposure: only marginal additional improvement
- Next: Stage 2C separability diagnostic
- Held-out test set remains untouched

See [`EXPERIMENT_MILESTONE.md`](EXPERIMENT_MILESTONE.md) for the controlled
experimental record and current interpretation.

## Codebase Guide

Use [`docs/codebase_guide.md`](docs/codebase_guide.md) for a module-by-module
map of the repository, important functions, testing commands, and what each
team member needs to understand first.

## Dataset Status

Danbooru/Safebooru acquisition and dataset split construction are not yet
implemented in this repository. The current preprocessing module is intentionally
left mostly untouched while the dataset member completes that part.

Target split for the paper-aligned dataset:

- Training: 10,000 images
- Validation: 1,000 images
- Held-out test: 500 images
- Optional small curated qualitative inspection set

## Testing

```bash
pytest tests/
```

The tests cover classical DWT/QIM behavior, BER, attack suite behavior, and
CNN input/output utility shape checks. TensorFlow is not required for the
default test suite.
