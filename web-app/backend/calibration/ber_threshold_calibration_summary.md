# BER Threshold Calibration Summary

> Candidate analysis only. A human-approved labeled dataset is required before changing the committed policy.

- Labeled positives: 35
- Labeled negatives: 8
- Positive BER range: 0.000000 to 0.648438
- Negative BER range: 0.476562 to 0.562500
- Distribution overlap: yes

## Counts by condition

- negative / non_watermarked: n=3, min=0.507812, max=0.562500, mean=0.528646
- negative / wrong_artwork_record: n=5, min=0.476562, max=0.546875, mean=0.503125
- positive / clean_watermarked: n=5, min=0.000000, max=0.648438, mean=0.325000
- positive / jpeg_50: n=5, min=0.289062, max=0.593750, mean=0.478125
- positive / jpeg_70: n=5, min=0.164062, max=0.593750, mean=0.435937
- positive / jpeg_90: n=5, min=0.015625, max=0.640625, mean=0.326562
- positive / png_resave: n=5, min=0.000000, max=0.648438, mean=0.325000
- positive / resize_50: n=5, min=0.132812, max=0.562500, mean=0.390625
- positive / resize_75: n=5, min=0.101562, max=0.609375, mean=0.389062

## Candidate at minimum specificity 0.95

- Threshold: 57/128 = 0.445312
- Sensitivity: 0.542857
- Specificity: 1.000000
- False positives: 0
- False negatives: 16

This script does not modify application policy or historical rows.
