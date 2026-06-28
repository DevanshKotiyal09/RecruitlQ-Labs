"""
Pre-computation pipeline.
Run this ONCE before ranking to build:
  - Embedding index (embeddings.npy + candidate_id_map.json)
  - JD embeddings (jd_embeddings.npy)
  - Signal statistics for normalisation (signal_stats.json)
  - Data hash for cache validation (data_hash.txt)

Usage:
    python precompute.py --candidates ./candidates.jsonl --index_dir ./index

This step has no time constraint. The ranking step (rank.py) must complete in 5 minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embedding index and signal stats.")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--index_dir",
        default="./index",
        help="Directory to store pre-computed artifacts (default: ./index)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if cache is valid"
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    index_dir = Path(args.index_dir)

    print("=" * 60)
    print("Redrob Candidate Ranking — Pre-computation Phase")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Compute file hash
    print("\n[1/3] Computing data file hash...")
    from ranker.ingester import compute_file_hash
    data_hash = compute_file_hash(candidates_path)
    print(f"  Data hash: {data_hash[:16]}...")

    # Check if cache is valid
    from ranker.semantic_encoder import EmbeddingIndex, HASH_FILENAME, INDEX_FILENAME
    hash_file = index_dir / HASH_FILENAME
    index_file = index_dir / INDEX_FILENAME
    if (not args.force
            and hash_file.exists()
            and index_file.exists()
            and hash_file.read_text().strip() == data_hash):
        print(f"\nCache is valid (same data hash). Skipping re-computation.")
        print("Use --force to rebuild anyway.")
        print(f"\nPre-computation complete (skipped) in {time.time() - t_start:.1f}s")
        return

    # Step 2: Build signal statistics
    print("\n[2/3] Computing signal statistics for normalisation...")
    from ranker.semantic_encoder import build_signal_stats
    build_signal_stats(candidates_path, index_dir)

    # Step 3: Build embedding index
    print("\n[3/3] Building semantic embedding index...")
    print("  This step takes ~15–30 minutes on CPU for 100K candidates.")
    print("  (Progress printed every 20 batches)")
    from ranker.semantic_encoder import build_embedding_index
    build_embedding_index(candidates_path, index_dir, data_hash)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pre-computation complete in {elapsed/60:.1f} minutes")
    print(f"Index saved to: {index_dir}/")
    print("You can now run rank.py within the 5-minute constraint.")
    print("=" * 60)


if __name__ == "__main__":
    main()
