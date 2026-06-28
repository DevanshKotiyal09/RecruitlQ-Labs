"""
Module 2: Candidate Ingester
Stream-parses candidates.jsonl with memory-efficient batching.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Generator, Iterator


def stream_candidates(jsonl_path: str | Path) -> Generator[dict, None, None]:
    """
    Yield one candidate dict at a time from a .jsonl file.
    Skips blank lines. Does not load the full file into memory.
    """
    path = Path(jsonl_path)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def batch_stream(
    jsonl_path: str | Path,
    batch_size: int = 2000,
) -> Generator[list[dict], None, None]:
    """
    Yield lists of candidate dicts in batches for parallel processing.
    """
    batch: list[dict] = []
    for cand in stream_candidates(jsonl_path):
        batch.append(cand)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_candidates(jsonl_path: str | Path) -> int:
    """Count total lines (candidates) without parsing JSON."""
    count = 0
    with open(Path(jsonl_path), "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def compute_file_hash(path: str | Path) -> str:
    """SHA-256 hash of file for cache-invalidation checks."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_all_candidate_ids(jsonl_path: str | Path) -> set[str]:
    """Return the complete set of candidate_ids (for submission validation)."""
    ids: set[str] = set()
    for cand in stream_candidates(jsonl_path):
        cid = cand.get("candidate_id", "")
        if cid:
            ids.add(cid)
    return ids


def build_candidate_index(jsonl_path: str | Path) -> dict[str, dict]:
    """
    Load all candidates into a dict keyed by candidate_id.
    Only use when the full pool must be in memory (reason generation pass).
    With 100K candidates at ~15 KB each this is ~1.5 GB worst case;
    use streaming wherever possible.
    """
    index: dict[str, dict] = {}
    for cand in stream_candidates(jsonl_path):
        cid = cand.get("candidate_id", "")
        if cid:
            index[cid] = cand
    return index
