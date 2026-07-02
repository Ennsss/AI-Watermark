"""Acquisition: stream Safebooru metadata, select candidates, download + preprocess.

Network/disk I/O lives here (the filter/dedup/split modules stay pure). To honor a
tight disk budget, images are downloaded in small chunks; each chunk is converted
immediately to a 512x512 PNG working copy under ``pool_dir`` and its full-size
originals are deleted before the next chunk. The perceptual hash is computed on the
512 PNG, so dedup and split can run on the manifest alone.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

from . import dedup, preprocess
from .filter import FilterConfig, select

REPO = "deepghs/safebooru_full"
TABLES = ("tables/table-1.parquet", "tables/table-2.parquet")
METADATA_COLUMNS = ["id", "rating", "width", "height", "hash", "mimetype", "tags"]


def load_artist_tags(repo: str = REPO) -> set[str]:
    """Load the set of artist tag names (tag type == 1) from tags.parquet."""
    import pandas as pd

    path = hf_hub_download(repo, "tags.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["name", "type"])
    return set(df.loc[df["type"] == 1, "name"].astype(str))


def iter_metadata(repo: str = REPO, columns: list[str] | None = None,
                  tables: tuple[str, ...] = TABLES, batch_size: int = 5000
                  ) -> Iterator[dict[str, Any]]:
    """Stream metadata records from the parquet tables without full download."""
    columns = columns or METADATA_COLUMNS
    fs = HfFileSystem()
    for table in tables:
        with fs.open(f"datasets/{repo}/{table}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                yield from batch.to_pylist()


def select_candidates(artist_tags: set[str], cfg: FilterConfig, limit: int,
                      repo: str = REPO
                      ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Scan metadata and return up to ``limit`` candidate records."""
    return select(iter_metadata(repo=repo), artist_tags, cfg, limit=limit)


def _new_pool():
    # Imported lazily so unit tests importing this module don't need cheesechaser.
    from cheesechaser.datapool import SafebooruDataPool
    return SafebooruDataPool()


def download_and_preprocess(
    candidates: list[dict[str, Any]],
    pool_dir: str | Path,
    chunk_size: int = 250,
    max_workers: int = 12,
    size: int = preprocess.TARGET_SIZE,
    bg: tuple[int, int, int] = preprocess.DEFAULT_BG,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Download each candidate, write a 512 PNG to pool_dir, and compute its phash.

    Returns (kept, stats) where kept are candidates that downloaded and processed
    cleanly (now carrying a ``phash`` and ``path``), and stats counts outcomes.
    Originals are deleted chunk-by-chunk to bound disk use.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool = _new_pool()

    by_id = {int(c["id"]): c for c in candidates}
    kept: list[dict[str, Any]] = []
    stats = {"requested": len(candidates), "downloaded": 0, "resumed": 0,
             "missing": 0, "corrupt": 0, "processed": 0}

    # Resume: reuse 512 PNGs already present in pool_dir from a prior run.
    ids: list[int] = []
    for rid in by_id:
        dst = pool_dir / f"{rid}.png"
        if dst.exists():
            try:
                with Image.open(dst) as im:
                    ph = dedup.compute_phash(im)
                rec = dict(by_id[rid])
                rec["phash"] = ph
                rec["path"] = str(dst)
                kept.append(rec)
                stats["resumed"] += 1
                stats["processed"] += 1
                continue
            except Exception:
                dst.unlink(missing_ok=True)
        ids.append(rid)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        tmp = Path(tempfile.mkdtemp(prefix="sb_dl_"))
        try:
            pool.batch_download_to_directory(
                chunk, str(tmp), max_workers=max_workers, save_metainfo=False)
            present = {}
            for fn in os.listdir(tmp):
                stem, _ = os.path.splitext(fn)
                try:
                    present[int(stem)] = tmp / fn
                except ValueError:
                    continue
            for rid in chunk:
                src = present.get(rid)
                if src is None:
                    stats["missing"] += 1
                    continue
                stats["downloaded"] += 1
                dst = pool_dir / f"{rid}.png"
                try:
                    preprocess.preprocess_file(src, dst, size=size, bg=bg)
                    with Image.open(dst) as im:
                        ph = dedup.compute_phash(im)
                except Exception:
                    stats["corrupt"] += 1
                    if dst.exists():
                        dst.unlink()
                    continue
                rec = dict(by_id[rid])
                rec["phash"] = ph
                rec["path"] = str(dst)
                kept.append(rec)
                stats["processed"] += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return kept, stats
