# -*- coding: utf-8 -*-
"""Verify a built dataset: image specs, manifests, artist-disjointness, cleanup."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
curated = DATA / "curated"
manifests = DATA / "manifests"

ok = True


def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg)
    ok = ok and cond


print("=== files on disk ===")
for name in ("train", "val", "test"):
    d = curated / name
    pngs = list(d.glob("*.png")) if d.exists() else []
    print(f"  {name}: {len(pngs)} png")

print("\n=== image specs (sample up to 5 per split) ===")
bad = 0
for name in ("train", "val", "test"):
    d = curated / name
    for p in list(d.glob("*.png"))[:5]:
        with Image.open(p) as im:
            if im.size != (512, 512) or im.mode != "RGB":
                bad += 1
                print(f"   BAD {p.name}: {im.size} {im.mode}")
check(bad == 0, "all sampled images are 512x512 RGB PNG")

print("\n=== manifests ===")
rows_by_split = defaultdict(list)
master = manifests / "dataset_manifest.csv"
check(master.exists(), "dataset_manifest.csv exists")
with master.open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    rows_by_split[r["split"]].append(r)
print("  manifest counts:", {k: len(v) for k, v in rows_by_split.items()})

# every manifest path exists on disk
missing = [r["id"] for r in rows if not (DATA / r["path"]).exists()]
check(not missing, f"all {len(rows)} manifest paths exist on disk")

# every manifest row has a non-empty artist (require_artist=True)
no_artist = [r["id"] for r in rows if not r["artist"]]
check(not no_artist, "every image has an artist tag")

print("\n=== artist-disjointness (from manifest) ===")
artists = defaultdict(set)
for r in rows:
    artists[r["split"]].add(r["artist"])
names = list(artists)
overlap_found = False
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        ov = artists[names[i]] & artists[names[j]]
        if ov:
            overlap_found = True
            print(f"   OVERLAP {names[i]}/{names[j]}: {sorted(ov)[:5]}")
check(not overlap_found, "no artist appears in more than one split")

# no duplicate md5 / phash collisions across final set
md5s = [r["md5"] for r in rows]
check(len(md5s) == len(set(md5s)), "no duplicate MD5 in final set")

print("\n=== reproducibility + cleanup ===")
check((DATA / "candidates" / "build_meta.json").exists(), "build_meta.json saved")
check((DATA / "candidates" / "candidate_ids.json").exists(), "candidate_ids.json saved")
check(not (DATA / "pool").exists(), "pool/ working copies cleaned up")

# disk usage of curated
total = sum(p.stat().st_size for p in curated.rglob("*.png"))
print(f"\ncurated size: {total/1e6:.1f} MB for {len(rows)} images "
      f"({total/max(len(rows),1)/1e3:.0f} KB/img avg)")

print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
