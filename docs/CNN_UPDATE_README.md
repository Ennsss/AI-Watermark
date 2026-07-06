# CNN Update README

This note documents the recent CNN-assisted extraction updates, how to run them,
and how to interpret the current results.

## What Changed

The project now has a runnable CNN benchmark script:

```text
scripts/run_cnn_benchmark.py
```

The script trains a CNN decoder from generated watermark examples and evaluates
it with the same attack suite used by the classical benchmark.

The comparison output includes both branches:

```text
classical_ber
classical_success
cnn_ber
cnn_success
```

The CNN input is not raw RGB. It uses the same extraction-side preprocessing
defined in:

```text
src/watermark/cnn_extraction.py
```

The CNN input flow is:

```text
RGB image
-> YCbCr
-> Y luminance channel
-> two-level DWT
-> LH2 and HL2 subband maps
-> 128 x 128 x 2 tensor
```

The dataset image preprocessing is separate and is defined in:

```text
src/dataset/preprocess.py
```

That flow is:

```text
flatten alpha
-> center-crop square
-> resize to 512 x 512
-> save RGB PNG
```

## Dataset Status

The curated dataset is now present locally:

```text
data/curated/train  10000 PNG files
data/curated/val     1000 PNG files
data/curated/test     500 PNG files
```

The manifest files describe the same split:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
data/manifests/dataset_manifest.csv
```

The manifests are the index/catalog. The actual images are the PNG files under
`data/curated/`.

## How To Verify The Dataset

From Git Bash:

```bash
cd "/c/Users/My PC/Desktop/watermark/AI-Watermark"
source .venv/Scripts/activate
python scripts/verify_dataset.py
```

Expected counts:

```text
train: 10000
val:    1000
test:    500
```

## How To Run The Classical Benchmark

```bash
python -m cli.main benchmark tests/fixtures --delta 16 -o results.csv
```

This uses the classical DWT-QIM extractor only.

## How To Run The CNN Benchmark On Fixtures

This is only a smoke test because it uses seven fixture images:

```bash
python scripts/run_cnn_benchmark.py tests/fixtures \
  --samples-per-image 4 \
  --epochs 5 \
  --batch-size 4 \
  -o cnn_results.csv \
  --model-output cnn_decoder_fixture.keras
```

The fixture run proves the CNN path works, but it is not a research-valid
training setup.

## How To Run The CNN Benchmark On The Dataset

Use the curated train split for training and the curated test split for final
evaluation:

```bash
python scripts/run_cnn_benchmark.py data/curated/test \
  --train-images data/curated/train \
  --max-train-images 1000 \
  --samples-per-image 4 \
  --epochs 20 \
  --batch-size 16 \
  -o cnn_results_dataset.csv \
  --model-output cnn_decoder_dataset.keras
```

For a larger experiment, increase `--max-train-images`, `--samples-per-image`,
and/or `--epochs`.

## Current Dataset Result

The current dataset benchmark file is:

```text
cnn_results_dataset.csv
```

It contains:

```text
6500 rows
500 test images x 13 attack conditions
```

Current aggregate result:

```text
Classical mean BER: 0.2956
CNN mean BER:       0.5099
```

Current success counts:

```text
Classical perfect recoveries:
  none:       279/500
  jpeg_q90:    10/500
  all other attacks: 0/500

CNN perfect recoveries:
  0/6500
```

Interpretation:

```text
The CNN pipeline runs, but the current trained CNN is still near random.
```

A BER near `0.50` means about half of the 128 payload bits are wrong, which is
effectively random guessing.

## Why The CNN Is Not Improving Yet

The current CNN branch is functional, but the training strategy is still very
basic. It uses a shallow CNN and generated attacked examples, but it does not yet
outperform the classical extractor.

Likely reasons:

- The CNN architecture may be too small for robust bit recovery.
- The training examples may need better balancing across attacks.
- The model may need more training data, epochs, or a stronger input
  normalization strategy.
- A fixed 128-bit payload may encourage memorization or weak supervision unless
  payload seeds and coefficient seeds are varied carefully.
- Cropping and resizing break coefficient alignment, which is difficult for the
  current CNN input alone.

## Safe Conclusion For Reporting

Use this wording:

```text
A CNN-assisted extraction branch was implemented and evaluated using the same
benchmark suite as the classical DWT-QIM extractor. The CNN pipeline is
operational and accepts LH2/HL2 transform-domain tensors as input. However, the
current CNN model achieved a mean BER of approximately 0.5099 on the curated test
benchmark, with no perfect 128-bit recoveries. This indicates near-random bit
prediction and shows that the current CNN configuration does not yet improve
watermark recovery over the classical extractor.
```

## Next Steps

Recommended next work:

```text
1. Keep the classical benchmark as the baseline.
2. Improve the CNN training data generator.
3. Train with more images and more varied payload/coefficient seeds.
4. Track per-attack CNN BER during validation.
5. Consider a stronger CNN or residual architecture.
6. Add a paired classical-vs-CNN evaluation report.
```

The most important thing is not just "more epochs". The current results suggest
the CNN needs better training design and possibly a stronger architecture.
