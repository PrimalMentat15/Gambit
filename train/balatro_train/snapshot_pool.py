"""Snapshot start-state pool files (M3 backward curriculum).

A pool is a single packed file of serialized mid-run env snapshots (the
bytes returned by ``SimBalatroEnv.snapshot(i)``) plus each entry's ante.
The Rust vec-env loads it once per env object and shares it across all env
slots (``BalatroVecEnv.load_snapshot_pool``); auto-resets then start from a
sampled entry with probability ``set_snapshot_fraction(f)``.

This module owns the WRITER (and a reader for tests/tools); the Rust reader
is ``sim/py/src/pool.rs`` — keep the two in sync.

File format (little-endian), version 1::

    magic    8 bytes   b"BLTRPOOL"
    version  u32       1
    count    u32       >= 1
    antes    count x u8            entry i's run ante at snapshot time
    offsets  (count+1) x u64       byte offsets into the blob section
    blob     concatenated snapshot payloads

A pool file ships with a sidecar manifest ``<pool>.manifest.json`` recording
per-ante counts and generator provenance (see ``gen_snapshots.py``).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

MAGIC = b"BLTRPOOL"
VERSION = 1


def write_pool(path: str | Path, entries: list[tuple[int, bytes]]) -> None:
    """Write ``(ante, snapshot_bytes)`` entries as a version-1 pool file."""
    if not entries:
        raise ValueError("refusing to write an empty pool")
    for i, (ante, blob) in enumerate(entries):
        if not 1 <= ante <= 8:
            raise ValueError(f"entry {i}: ante {ante} out of 1..=8")
        if not blob:
            raise ValueError(f"entry {i}: empty snapshot")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(entries)))
        f.write(bytes(ante for ante, _ in entries))
        off = 0
        f.write(struct.pack("<Q", off))
        for _, blob in entries:
            off += len(blob)
            f.write(struct.pack("<Q", off))
        for _, blob in entries:
            f.write(blob)


def read_pool_antes(path: str | Path) -> list[int]:
    """Read only the header ante table (cheap; used by the mock env)."""
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) < 16 or header[:8] != MAGIC:
            raise ValueError(f"{path}: not a BLTRPOOL file")
        version, count = struct.unpack("<II", header[8:16])
        if version != VERSION:
            raise ValueError(f"{path}: unsupported pool version {version}")
        if count == 0:
            raise ValueError(f"{path}: empty pool")
        antes = list(f.read(count))
        if len(antes) != count or not all(1 <= a <= 8 for a in antes):
            raise ValueError(f"{path}: corrupt ante table")
        return antes


def read_pool(path: str | Path) -> list[tuple[int, bytes]]:
    """Read the full pool back as ``(ante, snapshot_bytes)`` entries."""
    data = Path(path).read_bytes()
    antes = read_pool_antes(path)
    count = len(antes)
    off_base = 16 + count
    offsets = struct.unpack_from(f"<{count + 1}Q", data, off_base)
    blob_base = off_base + 8 * (count + 1)
    if blob_base + offsets[-1] != len(data):
        raise ValueError(f"{path}: offset table does not span the blob")
    return [
        (antes[i], data[blob_base + offsets[i] : blob_base + offsets[i + 1]])
        for i in range(count)
    ]


def manifest_path(pool_path: str | Path) -> Path:
    return Path(str(pool_path) + ".manifest.json")


def write_manifest(pool_path: str | Path, meta: dict) -> Path:
    p = manifest_path(pool_path)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return p
