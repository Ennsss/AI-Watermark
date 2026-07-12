# Experiment Log

Use this file to record implementation and evaluation runs that may become
Chapter 4 evidence. Keep entries factual: config, command, output, observation,
and decision. Final claims should wait for the held-out test evaluation.

## Run Template

```text
Run ID:
Date:
Commit/version:
Owner:

Purpose:

Dataset:
- Source:
- Split:
- Image count:
- Preprocessing notes:

Configuration:
- Payload bits:
- Payload seed:
- Coefficient seed:
- Wavelet:
- DWT level:
- DWT mode:
- Subbands:
- Delta:
- Extraction method:

Attacks/degradations:

Command:

Output files:

Key results:
- BER:
- SSIM:
- PSNR:
- Inference time:
- Memory/CPU, if collected:

Observations:

Decision / next step:
```

## Run 001 - Classical Sanity Benchmark

```text
Run ID: 001
Date:
Commit/version:
Owner:

Purpose:
Sanity-check the classical DWT-QIM baseline on the fixture images.

Dataset:
- Source: tests/fixtures
- Split: sanity-only, not final train/validation/test
- Image count: 7
- Preprocessing notes: existing fixture sizes; not the final dataset pipeline

Configuration:
- Payload bits: 128
- Payload seed: 42
- Coefficient seed: 42
- Wavelet: haar
- DWT level: 2
- DWT mode: symmetric
- Subbands: LH2, HL2
- Delta: 16
- Extraction method: classical QIM bit decision

Attacks/degradations:
Default paper-aligned suite: JPEG 90/70/50, resize 75/50/25%, crop
mild/moderate/severe, re-encode 1x/2x/3x.

Command:
python -m cli.main benchmark tests/fixtures --delta 16 --wavelet haar -o baseline_results.csv

Output files:
baseline_results.csv

Key results:
- Mean BER: 0.2564
- Mean SSIM: 0.8871
- Recovery rate: 17.6% using strict perfect-recovery labeling

Observations:
No-attack BER was 0.0000 for all fixture images, so the clean embed/extract
roundtrip works. JPEG Q70/Q50, crop, severe resize, and line-art cases were
weak at delta 16. Crop performance was close to random, which is expected for
the non-tiled coefficient-aligned main baseline.

Decision / next step:
Run delta calibration for 24 and 32 on the same sanity fixtures before choosing
a provisional baseline delta. Do not treat this fixture benchmark as final
paper evidence.
```

