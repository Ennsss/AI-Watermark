# Codebase Guide

This guide explains the repository at the level needed to run tests, debug
baseline experiments, and understand where each research pipeline step lives.
You do not need to memorize every function before working. Start with the main
path, then learn optional/legacy modules only when you touch them.

## What To Learn First

Learn these concepts first:

1. Image arrays are usually `numpy.ndarray` values shaped `(H, W, 3)` for RGB
   or `(H, W)` for the Y luminance channel.
2. The main experiment embeds a raw 128-bit payload into DWT LH2/HL2
   coefficients using QIM.
3. Classical extraction reads bits using the QIM grid decision.
4. CNN extraction does not change embedding. It only predicts bits from
   degraded LH2/HL2 maps.
5. BER is the main recovery metric.

You do **not** need to understand every optional module immediately. AES/ECC,
tiling, sync, adaptive masking, and false-positive analysis are legacy or
future-work paths unless a task specifically uses them.

## Main Research Flow

```text
RGB image
-> preprocessing: RGB to YCbCr, extract Y
-> embedding: two-level DWT, LH2/HL2, QIM, inverse DWT
-> attacks: JPEG / resize / crop / re-encode
-> extraction:
   - classical branch: QIM bit decision
   - CNN branch: LH2/HL2 tensor -> CNN -> predicted bits
-> metrics: BER, SSIM, PSNR, timing
-> evaluation/statistics: paired classical-vs-CNN comparison
```

## Important Commands

Run all tests:

```powershell
python -m pytest tests/ -q
```

Run the classical sanity benchmark:

```powershell
python -m cli.main benchmark tests/fixtures --delta 16 --wavelet haar -o baseline_results.csv
```

Run delta sanity checks:

```powershell
python -m cli.main benchmark tests/fixtures --delta 24 --wavelet haar -o baseline_delta24.csv
python -m cli.main benchmark tests/fixtures --delta 32 --wavelet haar -o baseline_delta32.csv
```

Run structured evaluation:

```powershell
python -m evaluation run --configs baseline --output-dir evaluation_output
```

## Module Map

### `src/watermark/preprocessor.py`

Image loading and color-space utilities.

Important functions:

- `load_image(path)`: loads an image as RGB `uint8`.
- `save_image(path, image)`: saves RGB image output.
- `rgb_to_ycbcr(image)`: converts RGB to YCbCr.
- `ycbcr_to_rgb(ycbcr)`: converts YCbCr back to RGB.
- `extract_y_channel(ycbcr)`: returns the Y luminance channel.
- `replace_y_channel(ycbcr, y)`: puts a modified Y channel back.
- `pad_to_multiple(image, multiple=4)`: pads for two-level DWT.
- `unpad(image, pad_sizes)`: removes padding.

Research note:

This module exists, but final dataset normalization and alpha-flattening are
assigned separately. Do not treat it as the final dataset pipeline yet.

### `src/watermark/payload.py`

Payload generation and optional legacy payload security.

Main experiment function:

- `generate_fixed_payload(num_bits=128, seed=42)`: creates the deterministic
  raw ownership payload used for BER comparison.

Optional/legacy functions:

- `encode_payload(...)`: builds provenance payload, applies RS, AES, and bit conversion.
- `decode_payload_bits(...)`: reverses the legacy secure payload path.
- `rs_encode`, `rs_decode`: Reed-Solomon helpers.
- `aes_encrypt`, `aes_decrypt`: AES-CTR helpers.
- `apply_repetition_coding`, `decode_repetition_coding`: optional redundancy.
- `bytes_to_bits`, `bits_to_bytes`: bit/byte conversion.
- `derive_seed(key)`: derives a PRNG seed from a key.

Research note:

The main experiment should use `generate_fixed_payload`, not AES/ECC, so raw
BER is not hidden by correction or decoding.

### `src/watermark/embedding.py`

Core DWT-QIM logic. This is one of the most important files.

Important functions:

- `dwt2_decompose(y_channel, wavelet="haar", level=2, mode="symmetric")`:
  computes DWT coefficients.
- `dwt2_reconstruct(coeffs, wavelet="haar", mode="symmetric")`: inverse DWT.
- `qim_embed_bit(coefficient, bit, delta)`: embeds one bit into one coefficient.
- `qim_extract_bit(coefficient, delta)`: reads one bit using the QIM grid decision.
- `_get_embedding_locations(...)`: deterministic PRNG coefficient selection.
- `embed_watermark(y_channel, bits, seed, delta, ...)`: embeds payload bits.
- `extract_watermark(y_channel, num_bits, seed, delta, ...)`: low-level classical extraction.

Research note:

The main path uses Haar, level 2, symmetric mode, and LH2/HL2. The old LL2
fallback is still available but should not be part of the main comparison.

### `src/watermark/extraction.py`

Classical extraction wrapper for RGB images.

Important functions:

- `extract_from_image(image, num_bits, seed, delta, ...)`: RGB image -> Y -> DWT -> extracted bits.
- `compute_ber(original_bits, extracted_bits)`: computes raw bit error rate.

Research note:

This is the classical branch. CNN extraction should be compared against this
using the same images, payloads, coefficient seeds, and degradations.

### `src/watermark/cnn_extraction.py`

CNN-assisted extraction utilities.

Important functions:

- `prepare_cnn_input_from_y(y_channel, ...)`: returns stacked LH2/HL2 tensor.
- `prepare_cnn_input_from_image(image, ...)`: RGB image -> `128 x 128 x 2` tensor for 512 images.
- `predict_payload_bits(model, cnn_input, threshold=0.5)`: CNN probabilities -> raw bits.

Research note:

The CNN receives transform-domain coefficients, not raw RGB images.

### `src/watermark/models/cnn_decoder.py`

CNN model builder.

Important function:

- `build_cnn_decoder(input_shape=(128, 128, 2), output_bits=128, learning_rate=0.001)`.

Architecture:

- Conv2D 32, ReLU, MaxPool
- Conv2D 64, ReLU, MaxPool
- Flatten
- Dense 128, ReLU
- Dropout 0.30
- Dense 128, sigmoid

Research note:

TensorFlow is optional and imported lazily. Install `.[ml]` only when training
or running the CNN branch.

### `src/watermark/train_cnn.py`

CNN training helper.

Important function:

- `train_cnn_decoder(train_inputs, train_targets, val_inputs, val_targets, ...)`

Expected data:

- Inputs: arrays shaped `(N, 128, 128, 2)`.
- Targets: arrays shaped `(N, 128)`.

Research note:

This helper exists, but the full training-data generator still needs to be
built around the final dataset and degradation suite.

### `src/watermark/reconstruction.py`

Image reconstruction after embedding.

Important function:

- `reconstruct_image(ycbcr, watermarked_y, pad_sizes)`: modified Y + original Cb/Cr -> RGB.

### `src/attacks/suite.py`

Degradation simulation.

Main paper-aligned functions:

- `jpeg_compression(image, quality)`: JPEG QF attack.
- `resize_scale(image, scale)`: downscale/restore at 75%, 50%, or 25%.
- `crop_severity(image, severity, seed)`: mild/moderate/severe crop bands.
- `reencode_jpeg(image, passes, quality=85)`: repeated re-encoding.
- `get_default_attacks()`: returns the main experiment attack list.

Optional stress-test functions:

- `screenshot_simulation(...)`
- `format_conversion(...)`
- `gaussian_noise(...)`
- `combined_chain(...)`
- `get_optional_attacks()`
- `get_all_attacks()`

Research note:

Use `get_default_attacks()` for main results.

### `src/benchmark/runner.py`

Quick classical baseline benchmark.

Important classes/functions:

- `BenchmarkConfig`: config for quick benchmark runs.
- `BenchmarkResult`: one image/attack result row.
- `BenchmarkSummary`: aggregate results and CSV export.
- `embed_image(image, config)`: creates watermarked image and payload bits.
- `extract_and_measure(attacked_image, original_bits, config)`: BER/confidence/timing.
- `run_benchmark(images, config, attacks)`: full quick benchmark loop.

Research note:

Good for sanity checks and delta calibration on fixtures. Do not use fixture
results as final paper evidence.

### `src/cli/main.py`

Command-line interface.

Subcommands:

- `embed`: embed a raw 128-bit payload or optional legacy payload.
- `extract`: extract raw bits or optional legacy payload.
- `benchmark`: run quick classical benchmark.

Use this when you want direct manual testing from PowerShell.

### `src/evaluation/configs.py`

Evaluation configuration factories.

Important functions:

- `get_baseline_config()`: current paper-aligned baseline config.
- `get_delta_sweep()`: delta calibration candidates.
- `get_configs_by_name(name)`: used by the evaluation CLI.

Research note:

Only `baseline` and controlled delta calibration are main-experiment aligned.
Wavelet/repetition/tiling/adaptive/full sweeps are optional or legacy.

### `src/evaluation/runner.py`

Structured evaluation runner.

Important classes/functions:

- `EvalResult`: one structured result row.
- `EvalRun`: collection of result rows with CSV load/save.
- `get_evaluation_attacks()`: returns paper-aligned degradation suite.
- `run_single_config(...)`: evaluates one config.
- `run_full_evaluation(...)`: evaluates one or more configs and writes CSV.

Research note:

Currently focused on the classical branch. The paired CNN-vs-classical runner
still needs to be completed.

### `src/evaluation/metrics.py`

Metric helpers.

Important functions:

- `compute_psnr(original, modified)`.
- `compute_ssim(original, modified)`.
- `compute_nc(original_bits, extracted_bits)`.
- `compute_capacity_bpp(num_bits, image_shape)`.
- `compute_subband_utilization(num_bits, image_shape, level=2)`.
- `compute_false_positive_rate(...)`: optional legacy analysis.

Research note:

Main paper metric is BER. SSIM/PSNR support visual fidelity discussion.

### `src/evaluation/statistical_analysis.py`

Paired statistical testing helpers.

Important functions:

- `wilcoxon_paired_ber(classical_ber, cnn_ber)`.
- `rank_biserial_from_pairs(classical_ber, cnn_ber)`.
- `holm_bonferroni(p_values, alpha=0.05)`.

Research note:

Use after you have paired held-out BER results for classical and CNN extraction.
Requires optional `.[stats]`.

### `src/evaluation/aggregator.py`

Aggregates evaluation rows into means, standard deviations, medians, and
tables.

Important functions:

- `aggregate_by(...)`
- `aggregate_embedding_quality(...)`
- `pivot_table(...)`
- `filter_results(...)`

### `src/evaluation/report.py`

Generates Markdown and LaTeX tables from evaluation results.

Important class:

- `ReportGenerator`

Research note:

This was built around older report tables and may need cleanup once final
classical-vs-CNN CSV output is finalized.

### `src/evaluation/image_corpus.py`

Small local/synthetic image corpus builder.

Important classes/functions:

- `CorpusImage`
- `ImageCorpus`
- `_resize_to_square(...)`
- synthetic image generators

Research note:

This is not the final Danbooru/Safebooru dataset pipeline.

### `src/evaluation/__main__.py`

Evaluation CLI entry point.

Subcommands:

- `prepare`
- `run`
- `report`
- `fpr`

Example:

```powershell
python -m evaluation run --configs baseline --output-dir evaluation_output
```

### `src/watermark/masking.py`

Optional adaptive masking.

Important functions:

- `compute_local_variance(...)`
- `compute_adaptive_delta(...)`
- `detect_sparse_subbands(...)`
- `build_delta_map(...)`

Research note:

Adaptive masking is outside the main controlled comparison unless explicitly
introduced as future work or an ablation.

### `src/watermark/tiling.py`

Optional tiled embedding/extraction for crop resistance.

Important functions:

- `embed_watermark_tiled(...)`
- `extract_watermark_tiled(...)`
- `compute_tile_grid(...)`
- `_majority_vote(...)`

Research note:

Tiling is useful but outside the main baseline. It would change the crop story,
so keep it separate from main results.

### `src/watermark/sync.py`

Optional synchronization helpers for tiled extraction.

Important functions:

- `generate_sync_sequences(...)`
- `embed_sync_pattern(...)`
- `detect_crop_offset(...)`

Research note:

This is also outside the main controlled comparison.

## Tests Map

- `tests/test_preprocessor.py`: color conversion, Y channel, padding.
- `tests/test_payload.py`: raw payload helper and legacy AES/RS payload helpers.
- `tests/test_embedding.py`: DWT, QIM, coefficient locations, embed/extract roundtrip.
- `tests/test_extraction.py`: RGB extraction wrapper and BER.
- `tests/test_attacks.py`: degradation functions and default attack list.
- `tests/test_benchmark.py`: quick benchmark runner and CSV export.
- `tests/test_cnn_extraction.py`: CNN input shape and bit thresholding.
- `tests/test_roundtrip.py`: end-to-end and legacy/optional integration tests.
- `tests/test_masking.py`: optional adaptive masking.
- `tests/test_sync.py`: optional sync support.
- `tests/conftest.py`: shared fixtures.

When a test fails, read the test name first. It usually tells you which module
or behavior broke.

## How To Become Independent With Testing

1. Run the full tests after every meaningful change:

   ```powershell
   python -m pytest tests/ -q
   ```

2. If a test fails, run only that file:

   ```powershell
   python -m pytest tests/test_embedding.py -q
   ```

3. If needed, run one test class or test:

   ```powershell
   python -m pytest tests/test_embedding.py::TestEmbedExtract -q
   ```

4. For baseline behavior, run benchmarks and open the CSV:

   ```powershell
   python -m cli.main benchmark tests/fixtures --delta 24 --wavelet haar -o baseline_delta24.csv
   ```

5. Record meaningful benchmark/training/evaluation runs in
   `docs/experiment_log.md`.

## What You Should Understand Deeply

For your likely role, understand these deeply:

- `payload.generate_fixed_payload`
- `embedding.embed_watermark`
- `embedding.extract_watermark`
- `extraction.extract_from_image`
- `extraction.compute_ber`
- `attacks.suite.get_default_attacks`
- `cnn_extraction.prepare_cnn_input_from_image`
- `models.cnn_decoder.build_cnn_decoder`
- `train_cnn.train_cnn_decoder`
- `evaluation.statistical_analysis.wilcoxon_paired_ber`

Everything else can be learned as needed.

