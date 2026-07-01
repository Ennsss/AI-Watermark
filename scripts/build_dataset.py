"""Build the curated Safebooru illustration dataset.

Pipeline: select candidates (metadata filter) -> download + preprocess to 512 PNG
-> MD5 exact-dedup -> pHash near-dedup -> artist-disjoint split -> finalize into
data/curated/{train,val,test}/ with manifests.

Usage:
    python scripts/build_dataset.py --sample 40        # smoke test (tiny)
    python scripts/build_dataset.py --full             # full 10000/1000/500
    python scripts/build_dataset.py --full --keep-pool # keep 512 working copies

Run from the project root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import acquire, dedup, split as dsplit
from src.dataset.filter import FilterConfig

FULL_QUOTAS = {"train": 10000, "val": 1000, "test": 500}
MANIFEST_FIELDS = ["id", "artist", "md5", "phash", "width", "height",
                   "mimetype", "rating", "split", "path"]


def sample_quotas(n: int) -> dict[str, int]:
    """Scale the 20:2:1 train:val:test ratio down to ~n total (min 1 each)."""
    test = max(1, round(n / 23))
    val = max(1, test * 2)
    train = max(1, n - val - test)
    return {"train": train, "val": val, "test": test}


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the curated Safebooru dataset.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="smoke build of ~N images total")
    g.add_argument("--full", action="store_true", help="full 10000/1000/500 build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headroom", type=float, default=1.6,
                    help="candidate over-pull factor to absorb dedup/download loss")
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--keep-pool", action="store_true",
                    help="keep 512 working copies in data/pool after finalize")
    args = ap.parse_args()

    quotas = FULL_QUOTAS if args.full else sample_quotas(args.sample)
    total = sum(quotas.values())
    limit = math.ceil(total * args.headroom)

    data = Path(args.data_dir)
    pool_dir = data / "pool"
    curated = data / "curated"
    manifests = data / "manifests"
    meta_dir = data / "candidates"
    for d in (curated, manifests, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = FilterConfig()
    t0 = time.time()
    print(f"[build] quotas={quotas} total={total} candidate_limit={limit} seed={args.seed}")

    # 1. artist tags + candidate selection
    print("[1/5] loading artist tags + selecting candidates ...")
    artist_tags = acquire.load_artist_tags()
    candidates, drops, scanned = acquire.select_candidates(artist_tags, cfg, limit=limit)
    print(f"      scanned={scanned} selected={len(candidates)} drops={drops}")
    if not candidates:
        sys.exit("No candidates selected; aborting.")

    # 2. download + preprocess to 512 PNG, compute phash
    print(f"[2/5] downloading + preprocessing {len(candidates)} candidates "
          f"(chunk={args.chunk_size}) ...")
    kept, dl_stats = acquire.download_and_preprocess(
        candidates, pool_dir, chunk_size=args.chunk_size)
    print(f"      {dl_stats}")

    # 3. dedup: exact (md5) then near (phash)
    print("[3/5] dedup ...")
    kept, n_exact = dedup.remove_exact_duplicates(kept, key="md5")
    kept, n_near = dedup.remove_near_duplicates(kept, phash_key="phash",
                                                threshold=dedup.NEAR_DUP_THRESHOLD)
    print(f"      removed exact={n_exact} near={n_near} -> survivors={len(kept)}")

    # 4. artist-disjoint split
    print("[4/5] artist-disjoint split ...")
    splits, counts, unused = dsplit.split_artist_disjoint(
        kept, quotas=quotas, seed=args.seed)
    dsplit.assert_disjoint(splits)
    print(f"      counts={counts} unused={unused}")

    # 5. finalize: move into curated/{split}/, write manifests
    print("[5/5] finalizing ...")
    all_rows = []
    for name, recs in splits.items():
        split_dir = curated / name
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in recs:
            src = Path(r["path"])
            dst = split_dir / f"{r['id']}.png"
            if src.exists():
                shutil.move(str(src), str(dst))
            row = {k: r.get(k) for k in MANIFEST_FIELDS}
            row["split"] = name
            row["path"] = str(dst.relative_to(data))
            rows.append(row)
        write_manifest(manifests / f"{name}.csv", rows)
        all_rows.extend(rows)
    write_manifest(manifests / "dataset_manifest.csv", all_rows)

    # reproducibility record
    with (meta_dir / "build_meta.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "seed": args.seed, "quotas": quotas, "counts": counts,
            "candidate_limit": limit, "scanned": scanned,
            "drops": drops, "download_stats": dl_stats,
            "removed_exact": n_exact, "removed_near": n_near,
            "filter": {
                "min_width": cfg.min_width, "min_height": cfg.min_height,
                "aspect": [cfg.aspect_lo, cfg.aspect_hi],
                "ratings": sorted(cfg.ratings), "mimetypes": sorted(cfg.mimetypes),
                "require_artist": cfg.require_artist,
                "whitelist": sorted(cfg.whitelist), "blacklist": sorted(cfg.blacklist),
            },
        }, fh, indent=2)
    with (meta_dir / "candidate_ids.json").open("w", encoding="utf-8") as fh:
        json.dump([r["id"] for r in all_rows], fh)

    if not args.keep_pool:
        shutil.rmtree(pool_dir, ignore_errors=True)

    print(f"[done] {sum(counts.values())} images in {curated} "
          f"({time.time() - t0:.1f}s). Manifests in {manifests}.")


if __name__ == "__main__":
    main()
