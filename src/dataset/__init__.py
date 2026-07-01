"""Dataset construction for the ARTIFACT watermarking framework.

Builds a curated Safebooru (SFW Danbooru mirror) corpus of non-photorealistic
digital illustrations for training/validating the CNN extraction decoder and for
the held-out classical-vs-hybrid comparison.

Pipeline stages (see docs/superpowers/specs/2026-06-22-safebooru-dataset-build-design.md):

    acquire  -> filter -> dedup -> split -> preprocess -> manifest

This package keeps core logic free of I/O side effects where practical: functions
accept and return arrays / plain data structures; orchestration and file handling
live in scripts/build_dataset.py.
"""
