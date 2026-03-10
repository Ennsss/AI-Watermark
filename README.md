# Frequency-Domain Image Watermarking

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a blind, non-interactive steganographic watermarking
system that embeds invisible provenance signatures into digital artwork using
frequency-domain signal processing. The watermark encodes artist identity,
a timestamp, and a perceptual hash of the original image, enabling artists to
prove ownership of their work after it has been reposted, cropped, compressed,
or used without authorization to train generative AI models.

The core technique operates in the YCbCr color space, embedding data into
the luminance (Y) channel via a two-level Discrete Wavelet Transform (DWT).
Quantization Index Modulation (QIM) is applied to mid-frequency detail
subbands (LH2/HL2), which are chosen for their resilience to JPEG compression.
An adaptive perceptual masking scheme modulates embedding strength based on
local variance, increasing robustness in textured regions while preserving
visual quality in flat areas.

The payload is protected by AES-256 encryption and Reed-Solomon error
correction coding. A seeded PRNG spread-spectrum mapping distributes embedded
bits across wavelet coefficients, providing resistance to detection and
localized removal. The entire pipeline is deterministic, runs on CPU only,
and requires no internet connectivity.

## Features

- **Invisible watermark embedding** in the frequency domain (DWT + QIM)
- **Provenance payload** containing artist ID, UTC timestamp, and perceptual hash
- **AES-256 encryption** of the embedded payload
- **Reed-Solomon error correction** for recovery under degradation
- **Spread-spectrum PRNG mapping** for coefficient selection
- **Adaptive perceptual masking** via local variance heuristic
- **Tiled embedding** for crop resistance across spatial regions
- **Social media attack simulation** (JPEG, resize, crop, noise, format conversion, combined chains)
- **Robustness benchmarking** with BER and SSIM measurement and CSV export
- **Thesis evaluation framework** with configurable parameter sweeps and LaTeX table generation
- **False positive rate analysis** for verifying detector specificity
- **Fully offline, CPU-only** operation with deterministic reproducibility

## Architecture

### Embedding Pipeline

```
Input Image (RGB)
      |
      v
  RGB -> YCbCr conversion
      |
      v
  Extract Y (luminance) channel
      |
      v
  Pad dimensions to multiple of 4
      |
      v
  Build provenance payload
  (artist ID + timestamp + pHash)
      |
      v
  Reed-Solomon ECC encoding
      |
      v
  AES-256 encryption
      |
      v
  Spread-spectrum PRNG coefficient mapping
      |
      v
  2-level DWT (Haar or db4)
      |
      v
  QIM embedding into LH2/HL2 subbands
  (with optional adaptive delta from local variance)
      |
      v
  Inverse DWT reconstruction
      |
      v
  Recombine Y' + Cb + Cr -> YCbCr -> RGB
      |
      v
  Output watermarked PNG (lossless)
```

### Extraction Pipeline

```
Watermarked Image (possibly degraded)
      |
      v
  RGB -> YCbCr -> Extract Y channel -> Pad
      |
      v
  2-level DWT decomposition
      |
      v
  PRNG-guided coefficient lookup
      |
      v
  QIM bit reading (with adaptive delta if enabled)
      |
      v
  AES-256 decryption
      |
      v
  Reed-Solomon error correction decoding
      |
      v
  Payload validation + confidence score
      |
      v
  Recovered provenance (artist ID, timestamp, pHash)
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-watermark

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e ".[dev]"
```

### Embed a Watermark

```bash
python -m cli.main embed artwork.png \
    --artist-id "Jane Doe" \
    --key "my_secret_key_32bytes!!!!!!!!" \
    --wavelet haar \
    --delta 60.0 \
    -o artwork_watermarked.png
```

With adaptive perceptual masking:

```bash
python -m cli.main embed artwork.png \
    --artist-id "Jane Doe" \
    --key "my_secret_key_32bytes!!!!!!!!" \
    --adaptive \
    --delta-min 20.0 \
    --delta-max 80.0 \
    -o artwork_watermarked.png
```

### Extract a Watermark

```bash
python -m cli.main extract artwork_watermarked.png \
    --key "my_secret_key_32bytes!!!!!!!!" \
    --num-bits 2216 \
    --wavelet haar \
    --delta 60.0 \
    --verify-artist "Jane Doe"
```

### Run Robustness Benchmark

```bash
python -m cli.main benchmark test_images/ \
    --delta 60.0 \
    --wavelet haar \
    -o results.csv
```

## Evaluation Framework

The project includes a thesis-grade evaluation framework that runs configurable
parameter sweeps across the full attack suite and generates LaTeX-formatted
result tables.

### Prepare the Image Corpus

```bash
python -m evaluation prepare --size 512
```

### Run the Full Evaluation

```bash
python -m evaluation run \
    --configs full \
    --output-dir evaluation_output \
    --seeds 0,1,2
```

Available config sweeps: `baseline`, `delta`, `wavelet`, `repetition`,
`tiling`, `adaptive`, `full`.

### Generate Report Tables

```bash
python -m evaluation report \
    --csv evaluation_output/results/full_evaluation.csv \
    --output-dir evaluation_output
```

This produces LaTeX and Markdown tables in `evaluation_output/reports/`.

### False Positive Rate Analysis

```bash
python -m evaluation fpr \
    --trials 1000 \
    --size 512 \
    --delta 60.0 \
    --output-dir evaluation_output
```

## Project Structure

```
src/
  watermark/
    preprocessor.py       RGB/YCbCr conversion, Y channel extraction, padding
    payload.py            Payload construction, Reed-Solomon coding, AES-256 encryption
    embedding.py          DWT decomposition, QIM embedding, spread-spectrum mapping
    extraction.py         DWT decomposition, QIM bit reading, decryption, RS decoding
    masking.py            Local variance computation, adaptive delta map
    reconstruction.py     Inverse DWT, channel recombination, RGB output
    tiling.py             Tiled embedding/extraction for crop resistance
    sync.py               Synchronization support for tiled extraction
  attacks/
    suite.py              JPEG, resize, crop, noise, format conversion, combined chains
  benchmark/
    runner.py             Batch BER/SSIM measurement, CSV export
  evaluation/
    __main__.py           Evaluation CLI (prepare, run, report, fpr)
    configs.py            Parameter sweep configurations
    image_corpus.py       Test image corpus builder
    runner.py             Full evaluation sweep executor
    metrics.py            BER, SSIM, false positive rate computation
    aggregator.py         Result aggregation across seeds and images
    report.py             LaTeX and Markdown table generation
  cli/
    main.py               CLI with embed, extract, benchmark subcommands
tests/
  test_preprocessor.py    Unit tests for image preprocessing
  test_payload.py         Unit tests for payload encoding and decoding
  test_embedding.py       Unit tests for DWT embedding and QIM
  test_extraction.py      Unit tests for watermark extraction
  test_masking.py         Unit tests for adaptive perceptual masking
  test_attacks.py         Unit tests for attack simulation suite
  test_benchmark.py       Unit tests for benchmark runner
  test_roundtrip.py       End-to-end embed -> attack -> extract integration tests
  test_sync.py            Unit tests for synchronization module
  conftest.py             Shared pytest fixtures
```

## Technical Details

### Discrete Wavelet Transform (DWT)

The image luminance channel is decomposed using a two-level DWT with either
the Haar or Daubechies-4 (db4) wavelet basis. This produces four subbands at
each level: LL (approximation), LH (horizontal detail), HL (vertical detail),
and HH (diagonal detail). Embedding targets the LH2 and HL2 subbands at
level 2, which capture mid-frequency content that survives JPEG quantization
better than high-frequency subbands (HH) while remaining less perceptually
significant than the low-frequency approximation (LL2).

### Quantization Index Modulation (QIM)

Each bit is embedded by quantizing a selected wavelet coefficient to one of
two interleaved uniform grids separated by a step size (delta). To embed a 0,
the coefficient is rounded to the nearest multiple of delta; to embed a 1,
it is rounded to the nearest multiple of delta offset by delta/2. Extraction
determines which grid a coefficient is closest to, recovering the embedded
bit. The delta parameter controls the trade-off between robustness (higher
delta) and visual quality (lower delta).

### Adaptive Perceptual Masking

Local variance of wavelet coefficients drives spatially varying embedding
strength. In high-variance (textured) regions, a larger delta is used for
greater robustness. In low-variance (flat or smooth) regions, a smaller
delta avoids visible banding artifacts. The delta map is computed from the
level-2 detail subbands and interpolated to match coefficient positions.

### Reed-Solomon Error Correction

The payload is encoded with Reed-Solomon codes that add configurable
redundancy symbols (default: 128). This allows recovery of the original
payload even when a significant fraction of embedded bits are corrupted by
image processing operations. The number of correctable symbol errors is
nsym / 2.

### Spread-Spectrum PRNG Mapping

A seeded Mersenne Twister PRNG generates a pseudorandom permutation that maps
payload bits to wavelet coefficient positions. This distributes the watermark
across the spatial extent of the image, preventing localized removal and
making the watermark statistically difficult to detect without the secret key.

### Tiled Embedding

For resistance to cropping attacks, the payload can be redundantly embedded
into independent spatial tiles of configurable size. Each tile contains a
complete copy of the watermark. During extraction, tiles are independently
decoded, and the best result (by confidence score) is selected.

## Success Metrics

| Metric                   | Target                                    |
|--------------------------|-------------------------------------------|
| Watermark BER (pre-ECC)  | < 25% after JPEG Q70 + resize             |
| Perfect Recovery Rate    | >= 90% after RS decoding across attack suite |
| Visual Quality (SSIM)    | >= 0.97 original vs. watermarked           |
| Perceptual Transparency  | Zero visible artifacts in blind A/B test   |
| Embedding Speed          | < 2 s for 4K image on mid-range CPU        |
| Extraction Speed         | < 3 s including ECC decoding               |
| Payload Capacity         | >= 64 bits actual provenance data           |
| Compression Resilience   | Survives 3x sequential JPEG Q75            |

## Running Tests

```bash
# Run the full test suite
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test module
pytest tests/test_roundtrip.py -v

# Run tests with coverage (requires pytest-cov)
pytest tests/ --cov=src --cov-report=term-missing
```

## Dependencies

| Package          | Purpose                                |
|------------------|----------------------------------------|
| numpy >= 1.24    | Array operations and seeded PRNG       |
| opencv-python >= 4.8 | Image I/O and color space conversion |
| Pillow >= 10.0   | Image format handling                  |
| PyWavelets >= 1.4 | Discrete Wavelet Transform            |
| imagehash >= 4.3 | Perceptual hashing (pHash)             |
| reedsolo >= 1.7  | Reed-Solomon error correction          |
| pycryptodome >= 3.19 | AES-256 encryption                 |
| scikit-image >= 0.21 | SSIM computation                   |
| pytest >= 7.4    | Test framework (dev dependency)        |

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this work in academic research, please cite:

```bibtex
@thesis{frequency_watermark_2026,
    title     = {Frequency-Domain Steganographic Watermarking for
                 Digital Art Provenance},
    author    = {TODO: Author Name},
    year      = {2026},
    school    = {TODO: Institution},
    type      = {TODO: Thesis Type},
    note      = {Software available at TODO: repository URL}
}
```
