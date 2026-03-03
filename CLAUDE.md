# Frequency-Domain Watermarking Framework

## Project Overview

A lightweight, client-side steganographic watermarking tool that embeds invisible provenance signatures into digital artwork using frequency-domain signal processing (DWT + QIM). Designed to survive JPEG compression, resizing, and cropping from social media platforms. Runs on CPU only — no GPU or internet required.

**Goal:** Enable artists to prove ownership of their work after it has been scraped, reposted, or used to train AI models.

## Architecture

### Core Pipelines

1. **Embedding Pipeline:** RGB → YCbCr → Extract Y channel → Pad → Build payload (artist ID + timestamp + pHash) → Reed-Solomon ECC → AES-256 encrypt → Spread-spectrum PRNG mapping → 2-level DWT (Haar/db4) → QIM into LH2/HL2 subbands → IDWT → Recombine → RGB → PNG output
2. **Extraction Pipeline:** Input image → YCbCr → Y channel → 2-level DWT → PRNG coefficient lookup → QIM bit reading → Decrypt → RS decode → Validate payload → Output provenance + confidence score
3. **Benchmark Pipeline:** Embed → Apply attack suite → Extract → Measure BER + SSIM → CSV export

### Key Technical Decisions

- **Embed in Y (luminance) channel only** — JPEG operates in YCbCr; human eye tolerates luminance changes better than chrominance
- **Target LH2/HL2 subbands** — Mid-frequency detail survives JPEG compression; LL2 is destroyed, HH subbands are stripped first
- **QIM with adaptive delta** — Local variance drives embedding strength: higher Δ in textured regions (more robust), lower Δ in flat regions (avoids banding)
- **No ML required for core** — DWT, QIM, Reed-Solomon, AES are all deterministic algorithms

### Technology Stack

| Component | Library |
|---|---|
| Language | Python 3.10+ |
| Wavelet Transform | PyWavelets (`pywt`) |
| Image Processing | OpenCV + Pillow |
| Perceptual Hashing | `imagehash` (pHash) |
| Error Correction | `reedsolo` |
| Encryption | PyCryptodome (AES-256) |
| PRNG | NumPy (seeded MT19937) |
| Benchmarking | scikit-image (SSIM) |
| GUI (post-MVP) | PyQt6 or Tkinter |
| Packaging | PyInstaller |

## Project Structure (Target)

```
src/
  watermark/
    __init__.py
    preprocessor.py      # F1: RGB↔YCbCr, Y extraction, padding
    payload.py           # F2: Payload construction, RS coding, AES encryption
    embedding.py         # F3: DWT decomposition, QIM embedding, spread-spectrum
    extraction.py        # F5: DWT decomposition, QIM reading, decryption, RS decoding
    masking.py           # F10: Local variance, adaptive delta computation
    reconstruction.py    # F4: IDWT, channel recombination, RGB output
  attacks/
    __init__.py
    suite.py             # F6: All 7 attack types, parameterized pipeline
  benchmark/
    __init__.py
    runner.py            # F7: Batch BER/SSIM measurement, CSV export
  cli/
    __init__.py
    main.py              # F8: CLI with embed/extract/benchmark subcommands
tests/
  test_preprocessor.py
  test_payload.py
  test_embedding.py
  test_extraction.py
  test_attacks.py
  test_benchmark.py
  test_roundtrip.py      # End-to-end embed → attack → extract
```

## MVP Features & Priorities

### P0 (Must Have)
- **F1** Image Pre-Processor — YCbCr conversion, Y channel extraction, dimension padding
- **F2** Provenance Payload Builder — Artist ID + timestamp + pHash, Reed-Solomon ECC, AES-256
- **F3** DWT Embedding Engine — 2-level DWT, QIM into LH2/HL2, adaptive strength via local variance
- **F4** Inverse DWT & Output — IDWT reconstruction, channel recombination, lossless PNG export
- **F5** Watermark Extractor — DWT decomposition, QIM bit reading, decrypt, RS decode, confidence score
- **F6** Social Media Attack Suite — JPEG compression, resize, crop, screenshot sim, format conversion, noise, combined chains
- **F7** Robustness Benchmark Runner — Batch BER/SSIM measurement, CSV export

### P1 (Should Have)
- **F8** CLI Interface — `embed`, `extract`, `benchmark` subcommands with config flags
- **F9** Simple Desktop GUI — Drag-and-drop for non-technical artists
- **F10** Perceptual Masking (Adaptive) — Local variance heuristic for spatially-varying QIM delta

### P2 (Nice to Have)
- **F11** Batch Processing — Folder-level embedding with CSV manifest
- **F12** Verification API — REST endpoint for platform integration

## Success Metrics

| Metric | Target |
|---|---|
| Watermark BER (pre-ECC) | < 25% after JPEG Q70 + resize |
| Perfect Recovery Rate | ≥ 90% after RS decoding across attack suite |
| Visual Quality (SSIM) | ≥ 0.97 original vs watermarked |
| Perceptual Transparency | Zero visible artifacts in blind A/B |
| Embedding Speed | < 2s for 4K on mid-range CPU |
| Extraction Speed | < 3s including ECC |
| Payload Capacity | ≥ 64 bits actual provenance data |
| Compression Resilience | Survives 3x sequential JPEG Q75 |

## Social Media Attack Parameters

| Attack | Parameters |
|---|---|
| JPEG Compression | Q: 50, 60, 70, 80, 85, 90 |
| Resize (Downscale) | Max dim: 1080, 1440, 2048px |
| Random Crop | 10%, 20%, 30%, 40% removal |
| Screenshot Sim | sRGB → Display P3 → sRGB + JPEG Q85 |
| Format Conversion | PNG → JPEG → WebP → JPEG |
| Gaussian Noise | σ = 2, 5, 10 |
| Combined Chain | Resize → JPEG Q70 → Crop 20% → JPEG Q80 |

## Development Phases

1. **Core Embedding Engine** (2 wk) — YCbCr, DWT, QIM, IDWT, basic roundtrip
2. **Payload Security Layer** (2 wk) — Reed-Solomon, AES-256, spread-spectrum PRNG
3. **Perceptual Masking** (1.5 wk) — Local variance, adaptive delta, SSIM validation
4. **Social Media Attack Suite** (1.5 wk) — All 7 attacks, parameterized pipeline
5. **Benchmark Framework** (1.5 wk) — Batch processing, BER/SSIM, CSV export
6. **CLI Tool & Packaging** (1 wk) — Argparse, subcommands, PyInstaller
7. **Evaluation & Documentation** (1.5 wk) — Full benchmark, results analysis, README

## Coding Conventions

- Pure Python 3.10+, type hints on all public functions
- NumPy arrays for all image/signal data — avoid Python loops over pixels
- All randomness must be seeded and deterministic (reproducible results)
- Separate concerns: preprocessing, payload, embedding, extraction, attacks, benchmarking
- Unit tests for each module; integration test for full embed→attack→extract roundtrip
- Keep core algorithms free of I/O — accept/return arrays, handle files at CLI/GUI layer

## Key Constraints

- **No GPU dependency** — All algorithms must run on CPU via NumPy/SciPy
- **No internet required** — Fully offline, client-side tool
- **No ML in core pipeline** — Deterministic signal processing only (ML is post-MVP optional enhancement)
- **Lossless output** — Watermarked images exported as PNG to preserve embedded data before platform upload
- **Minimum image size** — Dimensions must support 2-level DWT (minimum multiple of 4)
- **Alpha channel** — Open question; transparent regions have zero-value coefficients problematic for QIM

## Out of Scope (Post-MVP)

- ML-based perceptual masking (CNN for adaptive delta)
- Learned extraction decoder
- Public key infrastructure (PKI) for third-party verification
- Video watermarking
- Real-time plugin for Photoshop/Clip Studio
- Blockchain provenance registry
- Adversarial removal attack resistance
- Mobile app

## Glossary

- **DWT** — Discrete Wavelet Transform: decomposes image into frequency subbands
- **QIM** — Quantization Index Modulation: embeds bits by quantizing coefficients to interleaved grids
- **BER** — Bit Error Rate: fraction of bits incorrectly recovered after degradation
- **SSIM** — Structural Similarity Index: perceptual visual similarity (0–1, 1 = identical)
- **Reed-Solomon** — Error correction code allowing recovery from partial corruption
- **Spread Spectrum** — Distributes data across locations via PRNG to resist detection/removal
- **YCbCr** — Color space separating luminance (Y) from chrominance (Cb, Cr)
- **pHash** — Perceptual hash producing similar outputs for visually similar images
