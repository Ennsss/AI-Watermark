# Safebooru Illustration Dataset Build — Design Spec

**Date:** 2026-06-22
**Project:** ARTIFACT (Hawak Mo Ang Bits) — hybrid DWT-QIM + CNN watermarking
**Objective served:** Specific Objective #1 — *Curate* a specialized dataset of
non-photorealistic digital illustrations, used to train/validate the CNN decoder
and to hold out a test partition for the classical-vs-hybrid comparison.

Source of truth: the live paper (Google Doc) — Safebooru-only, artist-disjoint,
MD5 + pHash dedup. The older `CONTEXT-HANDOFF.md §6` COCO/DIV2K control set is
**superseded** and explicitly out of scope.

---

## 1. Success Criteria

- `data/curated/{train,val,test}/` containing **10,000 / 1,000 / 500** images.
- Every image **512×512**, lossless **PNG**, sRGB, 8-bit RGB (alpha flattened).
- **Artist-disjoint** split: no artist appears in more than one partition.
- **MD5** exact-duplicate removal + **pHash** near-duplicate removal (Hamming ≤ 6 / 64).
- Per-split CSV manifests with provenance: `id, artist, tags, orig_w, orig_h,
  rating, md5, phash, split, source_url`.
- **Deterministic & reproducible**: fixed seed; the curated candidate ID list is
  saved so the exact dataset can be regenerated.

## 2. Data Source & Access

- **Source:** Safebooru (SFW mirror of Danbooru2021) via the deepghs HuggingFace
  ecosystem — `deepghs/safebooru_full` (images) + its metadata table.
- **Access mechanism:** download **metadata first** (small), filter to a candidate
  ID list, then fetch **only those images by ID** using `cheesechaser`
  (deepghs's by-ID downloader). No multi-TB torrent, no full-repo download.
- **MD5 for exact-dup** comes from the **metadata** (Danbooru's canonical per-file
  md5), so full-resolution originals never need to be retained.

## 3. Filter Parameters (from dataset spec — RRL-anchored where stated)

- **Rating:** `rating:safe` only.
- **Tag whitelist:** `1girl, 1boy, scenery, original, illustration, line_art,
  monochrome, cel_shading, flat_color` (any-match).
- **Tag blacklist:** `photo, 3d, screenshot, comic, multiple_views, text_focus,
  realistic` (any-match → exclude).
- **Min resolution:** ≥ 512×512 (supports clean 2-level DWT into LH2/HL2).
- **Aspect ratio:** 0.5 ≤ w/h ≤ 2.0 (center-crop integrity).
- **pHash near-dup:** Hamming ≤ 6 / 64.

These are confirmed against the current dataset spec; any value lacking RRL
backing stays implementation-defined and is recorded as such, not asserted as
literature-derived.

## 4. Architecture — `src/dataset/` package

| Module | Responsibility |
|---|---|
| `acquire.py` | Fetch + filter metadata → candidate ID list; download images by ID; downscale-on-save to ≤768px working copies into `data/raw/`. |
| `filter.py` | Metadata-stage inclusion/exclusion (tags, rating, dims, aspect); post-decode checks (corrupt, true dims, alpha presence). |
| `dedup.py` | MD5 exact-dup (from metadata) → pHash near-dup (Hamming ≤ 6) on working copies. |
| `split.py` | Artist-disjoint deterministic split → 10k/1k/500: assign whole artists to a partition by `hash(seed, artist)` until each quota fills. |
| `preprocess.py` | Resize 512×512 (Lanczos) → RGB → alpha-flatten on neutral bg → PNG sRGB 8-bit into `data/curated/{split}/`. |
| `manifest.py` | Write per-split + master manifests; record seed and candidate ID list. |

Orchestrated by `scripts/build_dataset.py` with flags: `--sample N` (smoke),
`--full`, per-stage toggles, and **resumable** stages (skip work already done).

```
data/
  raw/        ≤768px working copies + raw_manifest.csv   (deleted after preprocess)
  curated/    train/ val/ test/   (512×512 PNG)
  manifests/  train.csv val.csv test.csv dataset_manifest.csv
  candidates/ candidate_ids.json + seed   (for exact reproduction)
```

## 5. Disk Strategy (tight, < 20 GB)

- Download working copies at ≤768px (not full-res).
- Use metadata MD5 for exact-dup so originals are unnecessary.
- Delete `data/raw/` working copies immediately after `preprocess.py` emits the
  512×512 PNGs.
- Estimated peak ≈ 9 GB (transient working copies + final), final ≈ 4.6 GB.

## 6. Determinism

- Single top-level `SEED`. Used for: artist→split hashing, any sampling, and the
  download order. The candidate ID list + seed are persisted so the dataset is
  byte-reproducible.

## 7. Execution Plan (smoke-first)

1. **Stage 0 — de-risk access:** install `cheesechaser`, pull a slice of metadata,
   download ~10–50 *filtered* images. Proves the entire approach is viable before
   any module is written around it.
2. **Build modules**, then run end-to-end on a **~200-image sample**; verify
   counts, artist-disjointness, dedup behavior, 512×512 output, manifest integrity.
3. **Full run** → 11,500 images.

## 8. Out of Scope

- COCO / DIV2K photorealistic control set (superseded).
- The 22-image qualitative set (skipped; qualitative examples will be drawn from
  the 500-image test partition).
- Watermark embedding, attack application, CNN training — separate downstream work.

## 9. Open Risks

- deepghs metadata may not expose every whitelist/blacklist tag uniformly →
  resolve empirically in Stage 0.
- Artist tags can be missing/multiple per image → define artist key explicitly
  (primary `artist:` tag; images with no artist tag handled by a documented rule).
- Yield: ~11,500 survivors may require a larger candidate pool after dedup/filter
  → pull with headroom (e.g., 1.5–2× candidates) and log drop counts per stage.
