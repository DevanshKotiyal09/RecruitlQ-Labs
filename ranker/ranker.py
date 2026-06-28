"""
Module 8: Ranker & Selector
Sorts all 100K scores, selects the top 100, resolves ties deterministically.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class ScoredCandidate:
    candidate_id: str
    composite_score: float
    confidence_score: float
    # Sub-scores kept for reason generation and tiebreaking
    behavioral_composite: float = 0.0
    tech_fit: float = 0.0
    career_quality: float = 0.0
    location_score: float = 0.0
    penalties: float = 0.0
    trust_penalty_mult: float = 1.0

    def __lt__(self, other: "ScoredCandidate") -> bool:
        """
        Comparison for sort: higher score = better (comes first).
        Full tiebreak chain ensures strict total order for determinism.
        """
        if self.composite_score != other.composite_score:
            return self.composite_score > other.composite_score
        # Tie 1: behavioral composite descending
        if self.behavioral_composite != other.behavioral_composite:
            return self.behavioral_composite > other.behavioral_composite
        # Tie 2: confidence descending
        if self.confidence_score != other.confidence_score:
            return self.confidence_score > other.confidence_score
        # Tie 3: candidate_id ASCENDING (lower ID = earlier = better rank)
        return self.candidate_id < other.candidate_id


def select_top_n(
    scored_candidates: list[ScoredCandidate],
    n: int = 100,
) -> list[ScoredCandidate]:
    """
    Select and rank top N candidates deterministically.
    Uses a stable key-based sort to guarantee total ordering including tiebreaks.
    """
    # Use a key tuple for fully deterministic ordering:
    # primary: composite_score descending
    # secondary: behavioral_composite descending
    # tertiary: confidence_score descending
    # quaternary: candidate_id ascending (lexicographic)
    scored_candidates.sort(
        key=lambda s: (
            -s.composite_score,
            -s.behavioral_composite,
            -s.confidence_score,
            s.candidate_id,       # ascending = lower ID ranks better on tie
        )
    )
    top = scored_candidates[:n]

    # Post-selection monotonicity enforcement for the score column only.
    # candidate_id ordering within equal scores is preserved by the sort above.
    for i in range(1, len(top)):
        if top[i].composite_score > top[i - 1].composite_score:
            top[i].composite_score = top[i - 1].composite_score

    return top


def assign_ranks(top_candidates: list[ScoredCandidate]) -> list[tuple[int, ScoredCandidate]]:
    """
    Assigns integer ranks 1..N to the sorted list.
    Returns list of (rank, ScoredCandidate).
    """
    return [(i + 1, cand) for i, cand in enumerate(top_candidates)]
