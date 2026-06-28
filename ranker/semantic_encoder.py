"""
Module 4: Semantic Encoder
Pre-computes dense vector representations of candidate narratives and JD probes.
Runs offline (no time limit). Ranking-time code only loads pre-built index.
Uses sentence-transformers with a compact CPU-friendly model.
"""
from __future__ import annotations

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

# Lazy import — only needed during pre-computation
_model = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Compact CPU-friendly model: 22M parameters, 384 dimensions
# Alternatives in order of preference: all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 512       # candidates per encoding batch
MAX_SEQ_LEN = 256      # token cap for each candidate document

INDEX_FILENAME = "embeddings.npy"
IDMAP_FILENAME = "candidate_id_map.json"
STATS_FILENAME = "signal_stats.json"
HASH_FILENAME = "data_hash.txt"
JD_FILENAME = "jd_embeddings.npy"


# ---------------------------------------------------------------------------
# JD probe documents (one general, two domain-specific)
# Each is a natural-language description of what the role needs,
# NOT just a list of keywords.
# ---------------------------------------------------------------------------

JD_GENERAL_PROBE = (
    "Senior AI engineer with production experience building embedding-based retrieval "
    "and ranking systems. Shipped real-world search and recommendation products to users "
    "at scale. Strong Python engineering. Designed evaluation frameworks for ranking using "
    "NDCG, MRR, A/B testing. Applied ML at product companies, not pure research or consulting. "
    "Five to nine years of experience. Located in India, preferably Pune or Noida."
)

JD_RETRIEVAL_PROBE = (
    "Production experience with vector databases, approximate nearest neighbour search, "
    "dense retrieval, hybrid BM25 and embedding retrieval, semantic search, FAISS, Pinecone, "
    "Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch. Shipped embedding-based ranking "
    "systems to real users. Handled embedding drift, index refresh, retrieval quality "
    "regression in production. Learning to rank, XGBoost ranking, LambdaMART. "
    "Candidate-JD matching at scale. Recommendation systems and search engines."
)

JD_NLP_PROBE = (
    "Natural language processing, information retrieval, text understanding, BERT, "
    "sentence transformers, language model fine-tuning, LoRA, QLoRA, PEFT, "
    "transformer models for NLP tasks. NLP production systems. "
    "Named entity recognition, text classification, semantic similarity, "
    "retrieval augmented generation, RAG systems."
)

ALL_JD_PROBES = [JD_GENERAL_PROBE, JD_RETRIEVAL_PROBE, JD_NLP_PROBE]


# ---------------------------------------------------------------------------
# Candidate narrative construction
# ---------------------------------------------------------------------------

def build_candidate_narrative(candidate: dict) -> str:
    """
    Construct the semantic document for a candidate.
    Excludes skills section intentionally to avoid rewarding keyword stuffing.
    Includes: headline + summary + all career descriptions.
    """
    profile = candidate.get("profile") or {}
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""

    career_parts: list[str] = []
    for job in (candidate.get("career_history") or []):
        title = job.get("title") or ""
        desc = job.get("description") or ""
        company = job.get("company") or ""
        if desc or title:
            career_parts.append(f"{title} at {company}: {desc}")

    narrative = " ".join(filter(None, [headline, summary] + career_parts))
    # Trim to reasonable length to respect MAX_SEQ_LEN
    return narrative[:2000]


# ---------------------------------------------------------------------------
# Model loading (lazy, once per process)
# ---------------------------------------------------------------------------

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = MAX_SEQ_LEN
    return _model


# ---------------------------------------------------------------------------
# Pre-computation: encode all candidates and save index
# ---------------------------------------------------------------------------

def build_embedding_index(
    jsonl_path: str | Path,
    output_dir: str | Path,
    data_hash: str,
) -> None:
    """
    Encode all candidates in JSONL and write:
        embeddings.npy         — float32 array (N, EMBED_DIM)
        candidate_id_map.json  — list of candidate_ids in row order
        jd_embeddings.npy      — float32 array (3, EMBED_DIM) for JD probes
        data_hash.txt          — hash for cache invalidation
    """
    from ranker.ingester import stream_candidates
    import time

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = _get_model()

    print(f"[Encoder] Loading candidates from {jsonl_path}...")
    all_ids: list[str] = []
    all_narratives: list[str] = []

    for cand in stream_candidates(jsonl_path):
        cid = cand.get("candidate_id", "")
        if cid:
            all_ids.append(cid)
            all_narratives.append(build_candidate_narrative(cand))

    n = len(all_narratives)
    print(f"[Encoder] Encoding {n} candidates in batches of {BATCH_SIZE}...")
    t0 = time.time()

    all_embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = all_narratives[start:end]
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings[start:end] = embs
        if (start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            pct = end / n * 100
            print(f"  {end}/{n} ({pct:.1f}%) — {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"[Encoder] Encoding complete in {elapsed:.0f}s")

    # Encode JD probes
    print("[Encoder] Encoding JD probes...")
    jd_embeddings = model.encode(
        ALL_JD_PROBES,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Save
    np.save(str(out / INDEX_FILENAME), all_embeddings)
    np.save(str(out / JD_FILENAME), jd_embeddings)

    with open(out / IDMAP_FILENAME, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    with open(out / HASH_FILENAME, "w", encoding="utf-8") as f:
        f.write(data_hash)

    print(f"[Encoder] Index saved to {out}/")
    print(f"  Embedding matrix: {all_embeddings.shape} = "
          f"{all_embeddings.nbytes / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Pre-computation: compute signal statistics for normalisation
# ---------------------------------------------------------------------------

def build_signal_stats(jsonl_path: str | Path, output_dir: str | Path) -> None:
    """
    Compute min/max/p10/p50/p90 for all numeric redrob_signals fields
    across the full candidate pool. Saved to signal_stats.json.
    """
    from ranker.ingester import stream_candidates

    numeric_fields = [
        "profile_completeness_score",
        "profile_views_received_30d",
        "applications_submitted_30d",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "github_activity_score",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
        "interview_completion_rate",
        "offer_acceptance_rate",
        "years_of_experience",
    ]

    buckets: dict[str, list[float]] = {f: [] for f in numeric_fields}

    for cand in stream_candidates(jsonl_path):
        signals = cand.get("redrob_signals") or {}
        profile = cand.get("profile") or {}
        row = dict(signals)
        row["years_of_experience"] = profile.get("years_of_experience") or 0.0
        for field in numeric_fields:
            val = row.get(field)
            if val is not None and val != -1:
                try:
                    buckets[field].append(float(val))
                except (TypeError, ValueError):
                    pass

    stats: dict[str, dict] = {}
    for field, values in buckets.items():
        if not values:
            stats[field] = {"min": 0, "max": 1, "p10": 0, "p50": 0.5, "p90": 1, "mean": 0.5}
            continue
        arr = np.array(values, dtype=np.float32)
        stats[field] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / STATS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Stats] Signal statistics saved to {out / STATS_FILENAME}")


# ---------------------------------------------------------------------------
# Runtime: load pre-computed index and compute similarities
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """
    Loads the pre-computed embedding index from disk.
    Computes cosine similarities via matrix multiplication (BLAS-accelerated).
    All embeddings are unit-normalised so dot product = cosine similarity.
    """

    def __init__(self, index_dir: str | Path):
        d = Path(index_dir)

        self.embeddings: np.ndarray = np.load(str(d / INDEX_FILENAME))
        self.jd_embeddings: np.ndarray = np.load(str(d / JD_FILENAME))

        with open(d / IDMAP_FILENAME, "r", encoding="utf-8") as f:
            self.id_list: list[str] = json.load(f)

        # Build fast id -> row index map
        self.id_to_idx: dict[str, int] = {
            cid: i for i, cid in enumerate(self.id_list)
        }

        # Pre-compute all similarity matrices (N × 3)
        # Each column is similarity against one JD probe
        self._similarities: np.ndarray = self.embeddings @ self.jd_embeddings.T

        print(f"[EmbeddingIndex] Loaded {len(self.id_list)} embeddings, "
              f"shape={self.embeddings.shape}")

    def get_similarities(self, candidate_id: str) -> tuple[float, float, float]:
        """
        Returns (general_sim, retrieval_sim, nlp_sim) for one candidate.
        All values in [-1, 1] (typically 0–1 for semantic similarity).
        """
        idx = self.id_to_idx.get(candidate_id)
        if idx is None:
            return 0.0, 0.0, 0.0
        sims = self._similarities[idx]
        return float(sims[0]), float(sims[1]), float(sims[2])

    def get_all_similarities(self) -> dict[str, tuple[float, float, float]]:
        """
        Returns the full map of candidate_id -> (general, retrieval, nlp).
        Used to bulk-populate CandidateFeatures.
        """
        result: dict[str, tuple[float, float, float]] = {}
        for i, cid in enumerate(self.id_list):
            sims = self._similarities[i]
            result[cid] = (float(sims[0]), float(sims[1]), float(sims[2]))
        return result

    def is_cache_valid(self, data_hash: str, index_dir: str | Path) -> bool:
        """Check if the index was built from the same data file."""
        hash_file = Path(index_dir) / HASH_FILENAME
        if not hash_file.exists():
            return False
        return hash_file.read_text().strip() == data_hash.strip()


def load_signal_stats(index_dir: str | Path) -> dict:
    """Load signal statistics from the pre-computed stats file."""
    p = Path(index_dir) / STATS_FILENAME
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
