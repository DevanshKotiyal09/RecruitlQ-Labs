"""
Main ranking pipeline — must complete in ≤ 5 minutes on CPU.

Architecture:
1. Load pre-computed embedding index (< 1s)
2. Compute cosine similarities via matrix multiplication (< 1s)
3. Stream candidates.jsonl once, extract features per-candidate (batch, parallelised)
4. Apply anti-trap detection + behavioral analysis + composite scoring
5. Select top 100, assign ranks, generate reasons
6. Write submission CSV + validate

Usage:
    python rank.py --candidates ./candidates.jsonl --index_dir ./index --out ./submission.csv

The embedding index must have been built first by running:
    python precompute.py --candidates ./candidates.jsonl --index_dir ./index
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Use all available CPUs for feature extraction
_CPU_COUNT = max(1, os.cpu_count() or 4)
_FEATURE_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Worker function for parallel feature extraction (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _extract_batch(
    batch: list[dict],
    jd_requirements_dict: dict,
) -> list[tuple[str, object, object, dict]]:
    """
    Worker function: extract features + trust flags for a batch of candidates.
    Runs in a separate process. Returns list of
    (candidate_id, features_dict, trust_serialized, signals_dict).
    trust_serialized is a plain dict with candidate_id, flags, honeypot_risk, trust_penalty_multiplier.
    """
    from ranker.jd_parser import JDRequirements
    from ranker.feature_extractor import extract_features
    from ranker.anti_trap import evaluate_trust
    import dataclasses

    # Reconstruct frozen dataclass from dict
    jd = JDRequirements(**jd_requirements_dict)

    results = []
    for cand in batch:
        cid = cand.get("candidate_id", "")
        try:
            feat = extract_features(cand, jd)
            trust = evaluate_trust(cand, feat)
            # Serialize to plain picklable dicts for IPC
            trust_dict = {
                "candidate_id": trust.candidate_id,
                "flags": list(trust.flags),
                "honeypot_risk": trust.honeypot_risk,
                "trust_penalty_multiplier": trust.trust_penalty_multiplier,
            }
            results.append((
                cid,
                dataclasses.asdict(feat),
                trust_dict,
                cand.get("redrob_signals") or {},
            ))
        except Exception:
            # Never let one bad candidate crash the batch
            results.append((cid, None, None, {}))
    return results


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the Senior AI Engineer JD.")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--index_dir",
        default="./index",
        help="Directory containing pre-computed embedding index (default: ./index)"
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)"
    )
    parser.add_argument(
        "--team_id",
        default=None,
        help="Team participant ID (used as output filename if --out not set)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_CPU_COUNT,
        help=f"Number of worker processes (default: {_CPU_COUNT})"
    )
    args = parser.parse_args()

    # Override output filename if team_id given
    out_path = Path(args.out)
    if args.team_id and args.out == "./submission.csv":
        out_path = Path(f"./{args.team_id}.csv")

    candidates_path = Path(args.candidates)
    index_dir = Path(args.index_dir)

    if not candidates_path.exists():
        print(f"Error: candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)
    if not index_dir.exists():
        print(f"Error: index directory not found: {index_dir}", file=sys.stderr)
        print("Run precompute.py first.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Redrob Candidate Ranking — Runtime Phase")
    print(f"CPU workers: {args.workers}")
    print("=" * 60)

    wall_clock_start = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load JD requirements and embedding index
    # -----------------------------------------------------------------------
    print("\n[1] Loading JD requirements and embedding index...")
    t = time.time()

    from ranker.jd_parser import load_jd_requirements
    jd = load_jd_requirements()

    from ranker.semantic_encoder import EmbeddingIndex
    index = EmbeddingIndex(index_dir)

    # Pre-compute all similarities (one matrix multiply, < 1s)
    print("  Computing cosine similarities...")
    all_similarities = index.get_all_similarities()

    print(f"  Done in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 2: Stream + batch feature extraction
    # -----------------------------------------------------------------------
    print(f"\n[2] Extracting features from candidates ({_CPU_COUNT} workers)...")
    t = time.time()

    from ranker.ingester import batch_stream
    from ranker.feature_extractor import CandidateFeatures
    from ranker.anti_trap import TrustFlags
    import dataclasses

    # Serialize JD requirements for IPC (frozensets -> lists)
    jd_dict = jd.to_dict()

    # Accumulate all scored candidates + their raw data for top-100 reason generation
    feat_index: dict[str, CandidateFeatures] = {}
    trust_index: dict[str, TrustFlags] = {}
    cand_signals_map: dict[str, dict] = {}  # for behavioral twin detection

    # We'll also collect (cand, feat) pairs for twin detection after all features extracted
    all_feat_pairs: list[tuple[dict, CandidateFeatures]] = []

    batches = list(batch_stream(candidates_path, batch_size=_FEATURE_BATCH_SIZE))
    total_candidates = sum(len(b) for b in batches)
    print(f"  {total_candidates} candidates in {len(batches)} batches")

    # Process batches in parallel
    processed = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_extract_batch, batch, jd_dict): batch
                for batch in batches
            }
            for future in as_completed(futures):
                results = future.result()
                for cid, feat_dict, trust_dict, signals in results:
                    if feat_dict is None:
                        continue
                    feat = CandidateFeatures(**feat_dict)
                    # Inject semantic similarities
                    sims = all_similarities.get(cid, (0.0, 0.0, 0.0))
                    feat.jd_semantic_similarity = sims[0]
                    feat.retrieval_probe_similarity = sims[1]
                    feat.nlp_probe_similarity = sims[2]
                    feat_index[cid] = feat
                    # Reconstruct TrustFlags from serialized dict
                    trust_rebuilt = TrustFlags(candidate_id=cid)
                    trust_rebuilt.flags = list(trust_dict.get("flags", []))
                    trust_rebuilt.honeypot_risk = trust_dict.get("honeypot_risk", 0.0)
                    trust_rebuilt.trust_penalty_multiplier = trust_dict.get(
                        "trust_penalty_multiplier", 1.0
                    )
                    trust_index[cid] = trust_rebuilt
                    cand_signals_map[cid] = signals
                processed += len(results)
                if processed % 10000 == 0:
                    print(f"  Processed {processed}/{total_candidates} candidates...")
    else:
        # Single-process fallback (needed for some environments that don't allow fork)
        from ranker.feature_extractor import extract_features
        from ranker.anti_trap import evaluate_trust
        for batch in batches:
            for cand in batch:
                cid = cand.get("candidate_id", "")
                try:
                    feat = extract_features(cand, jd)
                    sims = all_similarities.get(cid, (0.0, 0.0, 0.0))
                    feat.jd_semantic_similarity = sims[0]
                    feat.retrieval_probe_similarity = sims[1]
                    feat.nlp_probe_similarity = sims[2]
                    trust = evaluate_trust(cand, feat)
                    feat_index[cid] = feat
                    trust_index[cid] = trust
                    cand_signals_map[cid] = cand.get("redrob_signals") or {}
                except Exception:
                    pass
                processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed}/{total_candidates} candidates...")

    print(f"  Feature extraction done in {time.time() - t:.2f}s "
          f"({len(feat_index)} candidates)")

    # -----------------------------------------------------------------------
    # Step 3: Behavioral twin detection (batch, post-extraction)
    # -----------------------------------------------------------------------
    print("\n[3] Detecting behavioral twins...")
    t = time.time()

    from ranker.anti_trap import detect_behavioral_twins

    # Build lightweight list for twin detection (only signals needed)
    twin_pairs = [
        ({"redrob_signals": signals}, feat_index[cid])
        for cid, signals in cand_signals_map.items()
        if cid in feat_index
    ]
    twin_flags = detect_behavioral_twins(twin_pairs)

    # Merge twin flags into trust_index
    for cid, extra_trust in twin_flags.items():
        if cid in trust_index:
            for flag_name, severity, reason in extra_trust.flags:
                trust_index[cid].add_flag(flag_name, severity, reason)
        else:
            trust_index[cid] = extra_trust

    print(f"  {len(twin_flags)} candidates flagged as behavioral twins in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 4: Composite scoring for all candidates
    # -----------------------------------------------------------------------
    print("\n[4] Computing composite scores...")
    t = time.time()

    from ranker.scorer import compute_composite_score
    from ranker.behavioral_analyser import compute_behavioral_composite
    from ranker.ranker import ScoredCandidate

    scored_candidates: list[ScoredCandidate] = []

    for cid, feat in feat_index.items():
        trust = trust_index.get(cid) or TrustFlags(candidate_id=cid)
        composite, confidence = compute_composite_score(feat, trust, jd)
        behavioral = compute_behavioral_composite(feat)

        scored_candidates.append(ScoredCandidate(
            candidate_id=cid,
            composite_score=composite,
            confidence_score=confidence,
            behavioral_composite=behavioral,
            location_score=feat.location_score,
            trust_penalty_mult=trust.trust_penalty_multiplier,
        ))

    print(f"  Scored {len(scored_candidates)} candidates in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 5: Select top 100 and assign ranks
    # -----------------------------------------------------------------------
    print("\n[5] Selecting top 100...")
    t = time.time()

    from ranker.ranker import select_top_n, assign_ranks
    top100 = select_top_n(scored_candidates, n=100)
    ranked = assign_ranks(top100)

    top10_preview = [(r, s.candidate_id, f"{s.composite_score:.4f}") for r, s in ranked[:10]]
    print("  Top-10 preview:")
    for r, cid, score in top10_preview:
        print(f"    Rank {r:3d}: {cid} — score {score}")
    print(f"  Selection done in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 6: Load raw candidate data for top 100 (for reason generation)
    # -----------------------------------------------------------------------
    print("\n[6] Loading candidate profiles for top 100...")
    t = time.time()

    top100_ids = {scored.candidate_id for _, scored in ranked}
    candidate_index: dict[str, dict] = {}
    from ranker.ingester import stream_candidates
    for cand in stream_candidates(candidates_path):
        cid = cand.get("candidate_id", "")
        if cid in top100_ids:
            candidate_index[cid] = cand
        if len(candidate_index) == len(top100_ids):
            break  # all found, stop early

    print(f"  Loaded {len(candidate_index)} profiles in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 7: Generate reasons for top 100
    # -----------------------------------------------------------------------
    print("\n[7] Generating reasoning for top 100...")
    t = time.time()

    from ranker.reason_generator import generate_reasons_for_top100
    reasons = generate_reasons_for_top100(ranked, candidate_index, feat_index, trust_index, jd)

    print(f"  Done in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 8: Write submission CSV
    # -----------------------------------------------------------------------
    print(f"\n[8] Writing submission CSV to {out_path}...")
    t = time.time()

    # Load valid IDs set for validation
    from ranker.ingester import load_all_candidate_ids
    print("  Loading all candidate IDs for validation...")
    valid_ids = load_all_candidate_ids(candidates_path)

    from ranker.output_assembler import write_submission_csv, validate_csv_locally
    errors = write_submission_csv(ranked, reasons, out_path, valid_ids)

    if errors:
        print(f"  ASSEMBLY ERRORS ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"    - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Written in {time.time() - t:.2f}s")

    # -----------------------------------------------------------------------
    # Step 9: Local validation pass
    # -----------------------------------------------------------------------
    print("\n[9] Running local validation pass...")
    val_errors = validate_csv_locally(out_path)
    if val_errors:
        print(f"  VALIDATION FAILED ({len(val_errors)} issues):")
        for e in val_errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  All validation checks passed.")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    wall_elapsed = time.time() - wall_clock_start
    print(f"\n{'=' * 60}")
    print(f"Ranking complete in {wall_elapsed:.1f}s ({wall_elapsed/60:.2f} minutes)")
    print(f"Submission: {out_path}")
    print(f"Candidates processed: {len(feat_index)}")
    print(f"Top-100 score range: "
          f"{top100[-1].composite_score:.4f} – {top100[0].composite_score:.4f}")
    print("=" * 60)

    if wall_elapsed > 300:
        print(f"\nWARNING: Runtime exceeded 5 minutes ({wall_elapsed:.0f}s). "
              f"Reduce workers or optimize if needed.", file=sys.stderr)


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows
    multiprocessing.freeze_support()
    main()
