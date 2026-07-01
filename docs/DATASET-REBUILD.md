# Dataset — for teammates

The curated corpus is **10,000 / 1,000 / 500** non-photorealistic illustrations
(512×512 PNG) drawn from Safebooru, artist-disjoint, MD5 + pHash deduplicated.
Total ~3.7 GB. Built deterministically (`seed=0`).

You have two ways to get it.

## Option A — Just download the files (no setup)

Get the zip link from the team Google Drive, download, and unzip so you end up with:

```
data/curated/{train,val,test}/*.png
data/manifests/{train,val,test}.csv
```

That's it. Do **not** repost the zip publicly — the study does not redistribute
source images; team-internal sharing only.

## Option B — Rebuild it yourself (reproducible, from the repo)

This regenerates the identical images from the committed manifest, so nothing
large needs to be shared.

### 1. One-time HuggingFace access
The images come from the gated dataset `deepghs/safebooru_full`:

1. Make a free HuggingFace account.
2. Visit <https://huggingface.co/settings/content-preferences> and enable
   **"Show Not-For-All-Audiences content"**.
3. Visit <https://huggingface.co/datasets/deepghs/safebooru_full> and click
   **"Agree and access repository"**.
4. Log the CLI in: `huggingface-cli login` (paste a token from
   <https://huggingface.co/settings/tokens>).

### 2. Install deps
```
pip install cheesechaser imagehash opencv-python pillow numpy pandas pyarrow
```

### 3. Rebuild
From the repo root (the manifest in `data/manifests/` is committed):
```
python scripts/download_from_ids.py
```
This downloads + preprocesses each image into `data/curated/{split}/`. It is
resumable — re-run it to fetch anything that failed. Expect ~30–45 min.

### Verify (either option)
```
python scripts/verify_dataset.py
```
Confirms counts, 512×512 RGB, manifest integrity, and artist-disjointness.

---

**Reproducibility record:** `data/candidates/build_meta.json` (seed, quotas,
filter config, per-stage drop/dedup counts) and `candidate_ids.json` (the exact
11,500 IDs). The full pipeline lives in `src/dataset/` and `scripts/build_dataset.py`.
