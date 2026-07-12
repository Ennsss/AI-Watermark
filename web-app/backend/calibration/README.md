# BER calibration evidence

This directory contains read-only audit and controlled calibration outputs. Historical verification rows are never rewritten by these tools.

- `verification_threshold_audit.csv` recomputes payload differences, BER, and stored-rule consistency from the active database.
- `verification_ground_truth.csv` is the human-label input for historical events. Keep uncertain records as `unknown`.
- `controlled_calibration_observations.csv` contains explicitly labeled runs generated from locally available originals and registered watermarked copies.
- `ber_threshold_calibration_report.csv` contains metrics for every boundary from `0/128` through `64/128`.
- `ber_threshold_calibration_summary.md` reports distributions and a conservative candidate operating point for review.

The generated candidate does not automatically become application policy. Review source-file provenance, dataset representativeness, condition coverage, and false-positive/false-negative tradeoffs first.
