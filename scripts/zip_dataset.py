# -*- coding: utf-8 -*-
"""Package the curated dataset + manifests into a single zip for sharing.

PNGs are already compressed, so we store (no recompression) for speed. ZIP64 is
enabled for the >4GB-capable archive. Paths are kept relative to the repo root so
unzipping recreates data/curated/, data/manifests/, data/candidates/.
"""
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"C:\Users\PC\Downloads\ARTIFACT-dataset-11500.zip")
INCLUDE = ["data/curated", "data/manifests", "data/candidates"]

t0 = time.time()
n = 0
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
    for inc in INCLUDE:
        base = ROOT / inc
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
    guide = ROOT / "docs" / "DATASET-REBUILD.md"
    if guide.exists():
        z.write(guide, "DATASET-REBUILD.md")
        n += 1

size_gb = OUT.stat().st_size / 1e9
print(f"wrote {OUT}")
print(f"  {n} files, {size_gb:.2f} GB, {time.time()-t0:.1f}s")
