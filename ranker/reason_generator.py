"""
Module 9: Reason Generator
Produces 1–2 sentence fact-grounded reasoning strings for each selected candidate.
Every claim traces to an actual field in the candidate's profile.
No hallucination: the generator selects and formats; it does not compose.
Tone is calibrated to rank position.
"""
from __future__ import annotations

from ranker.feature_extractor import CandidateFeatures, JD_CORE_SKILLS
from ranker.anti_trap import TrustFlags
from ranker.jd_parser import JDRequirements

# ---------------------------------------------------------------------------
# Rank-band tone vocabulary
# ---------------------------------------------------------------------------
RANK_BANDS = {
    "strong": range(1, 11),    # ranks 1–10
    "good":   range(11, 31),   # ranks 11–30
    "medium": range(31, 61),   # ranks 31–60
    "weak":   range(61, 101),  # ranks 61–100
}

STRENGTH_OPENERS = {
    "strong": "Strong match:",
    "good":   "Good candidate:",
    "medium": "Moderate fit:",
    "weak":   "Adjacent profile:",
}

CONCERN_INTRODUCERS = {
    "strong": "Potential concern:",
    "good":   "Note:",
    "medium": "Key gap:",
    "weak":   "Primary gap:",
}


def _get_band(rank: int) -> str:
    for band, rng in RANK_BANDS.items():
        if rank in rng:
            return band
    return "weak"


# ---------------------------------------------------------------------------
# Fact selectors
# ---------------------------------------------------------------------------

def _select_strongest_strength(
    feat: CandidateFeatures,
    candidate: dict,
) -> str | None:
    """
    Select the single most important strength fact for this candidate.
    Returns a short factual clause.
    """
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []

    # Priority 1: Production deployment evidence in career descriptions
    if feat.production_deployment_evidence > 0.25:
        from ranker.jd_parser import RETRIEVAL_KEYWORDS, PRODUCTION_KEYWORDS, count_keyword_hits
        for job in career:
            desc = (job.get("description") or "").lower()
            if count_keyword_hits(desc, RETRIEVAL_KEYWORDS) >= 1 and \
               count_keyword_hits(desc, PRODUCTION_KEYWORDS) >= 1:
                company = job.get("company") or "a product company"
                role_title = job.get("title") or "engineer"
                return (f"shipped a retrieval/ranking system as {role_title} at {company} "
                        f"with production deployment evidence")

    # Priority 2: Retrieval / semantic search career evidence
    if feat.retrieval_career_evidence > 0.3:
        from ranker.jd_parser import RETRIEVAL_KEYWORDS, count_keyword_hits
        for job in career:
            desc = (job.get("description") or "").lower()
            if count_keyword_hits(desc, RETRIEVAL_KEYWORDS) >= 1:
                company = job.get("company") or "previous employer"
                return f"hands-on retrieval/search experience at {company}"

    # Priority 3: Years of experience + relevant title
    yoe = feat.years_of_experience
    curr_title = profile.get("current_title") or "professional"
    if feat.current_title_ai_relevance >= 0.8 and yoe >= 4:
        return f"{yoe:.1f} years as {curr_title} with AI/ML focus"

    # Priority 4: Strong semantic similarity
    if feat.jd_semantic_similarity > 0.45:
        return (f"career narrative semantically aligned with JD requirements "
                f"({feat.jd_semantic_similarity:.2f} similarity)")

    # Priority 5: JD core skills with corroboration (uses module-level JD_CORE_SKILLS)
    if feat.jd_core_skill_corroborated >= 2:
        skills = candidate.get("skills") or []
        relevant_names = [
            s.get("name") for s in skills
            if any(kw in (s.get("name") or "").lower() for kw in JD_CORE_SKILLS)
            and (s.get("duration_months") or 0) >= 6
        ]
        if relevant_names:
            return f"corroborated skills: {', '.join(relevant_names[:3])}"

    # Priority 6: Experience in band
    if 5 <= yoe <= 9:
        return f"{yoe:.1f} years of experience within the 5–9 year target band"

    # Priority 7: Platform assessment scores
    if feat.skill_assessment_composite > 0.5:
        return f"platform skill assessment composite score {feat.skill_assessment_composite:.2f}"

    # Fallback
    if yoe > 0:
        return f"{yoe:.1f} years of professional experience"

    return None


def _select_secondary_strength(
    feat: CandidateFeatures,
    candidate: dict,
    already_mentioned: str,
) -> str | None:
    """Select a second supporting strength fact."""
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    # Evaluation framework evidence
    if feat.eval_career_evidence > 0.2 or feat.eval_skill_signal > 0.4:
        if "eval" not in already_mentioned and "ndcg" not in already_mentioned.lower():
            return "evidence of ranking evaluation framework knowledge"

    # Pre-LLM ML evidence
    if feat.pre_llm_ml_evidence and "pre-llm" not in already_mentioned:
        return "ML production experience predating the LLM era"

    # LLM fine-tuning
    if feat.llm_finetune_skill > 0.4 and "fine-tun" not in already_mentioned:
        return "LLM fine-tuning experience (LoRA/QLoRA/PEFT)"

    # Location fit
    if feat.location_score >= 0.9 and "pune" not in already_mentioned and "noida" not in already_mentioned:
        loc = profile.get("location") or "preferred location"
        return f"located in {loc} (preferred)"

    # GitHub activity
    if feat.github_activity_score > 40 and "github" not in already_mentioned:
        return f"active GitHub contributor (score {feat.github_activity_score:.0f}/100)"

    # Recruiter market interest
    saved = int(signals.get("saved_by_recruiters_30d") or 0)
    if saved >= 8 and "saved" not in already_mentioned:
        return f"saved by {saved} recruiters in the last 30 days"

    # Short notice period
    notice = feat.notice_period_days
    if notice <= 15 and "notice" not in already_mentioned:
        return f"immediately available (notice period: {notice} days)"

    return None


def _select_primary_concern(
    feat: CandidateFeatures,
    trust: TrustFlags,
    candidate: dict,
) -> str | None:
    """
    Select the most significant concern for this candidate.
    If there is a penalty that meaningfully reduced the score, it must be stated.
    """
    # Consulting-only career
    if feat.consulting_only_flag:
        return "entire career at services/consulting firms with no product company tenure"

    # Research-only
    if feat.research_only_flag:
        return "career limited to research environments with no production deployment evidence"

    # Recent LLM-only
    if feat.recent_llm_only:
        return "ML experience appears to be recent LLM-wrapper work without pre-LLM production ML background"

    # Inactive architect
    if feat.inactive_architect_flag:
        return "recent roles indicate reduced hands-on coding (architecture/tech lead without coding evidence)"

    # CV / speech dominance
    if feat.cv_speech_dominance > 0.6:
        return "ML background primarily in computer vision/speech, not NLP/retrieval"

    # Very long notice period
    if feat.notice_period_days > 90:
        notice = feat.notice_period_days
        return f"long notice period ({notice} days) vs JD preference for sub-30 days"

    # Stale profile
    if feat.days_since_last_active > 150:
        days = feat.days_since_last_active
        return f"profile inactive for ~{days} days; engagement uncertain"

    # Low recruiter response rate
    if feat.recruiter_response_rate < 0.2:
        rr = feat.recruiter_response_rate
        return f"low recruiter response rate ({rr:.0%}) may indicate limited availability"

    # Non-India, willing to relocate
    if feat.location_category == "international_relocatable":
        country = feat.country
        return f"based outside India ({country}); relocation required"

    # Outside experience band
    yoe = feat.years_of_experience
    if yoe < 4:
        return f"limited experience ({yoe:.1f} years) below the 5–9 year target band"
    if yoe > 12:
        return f"senior experience ({yoe:.1f} years) may indicate seniority above role scope"

    # Keyword stuffing / inconsistency
    if trust.has_flag("keyword_stuffing_severe"):
        return "AI skill claims poorly corroborated by career description content"

    # Profile inconsistency
    if feat.profile_consistency_score < 0.6:
        return "profile internal consistency is low (title/experience gap)"

    return None


# ---------------------------------------------------------------------------
# Main reason assembly
# ---------------------------------------------------------------------------

def generate_reason(
    rank: int,
    candidate: dict,
    feat: CandidateFeatures,
    trust: TrustFlags,
    jd: JDRequirements,
) -> str:
    """
    Produce a 1–2 sentence reason string for a ranked candidate.
    Every claim is traceable to a field in the candidate's profile.
    """
    band = _get_band(rank)
    profile = candidate.get("profile") or {}
    current_title = profile.get("current_title") or "Professional"
    yoe = feat.years_of_experience
    company = profile.get("current_company") or ""

    # Collect facts
    strength1 = _select_strongest_strength(feat, candidate)
    strength2 = _select_secondary_strength(feat, candidate, strength1 or "")
    concern = _select_primary_concern(feat, trust, candidate)

    # Must have at least one fact to work with
    if not strength1 and not concern:
        return (
            f"{current_title} with {yoe:.1f} yrs experience; "
            f"included based on platform signals and partial profile match."
        )

    # Build strength clause
    if strength1 and strength2:
        strength_clause = f"{strength1}; {strength2}"
    elif strength1:
        strength_clause = strength1
    else:
        strength_clause = f"{current_title} with {yoe:.1f} years experience"

    # Assemble reason by band
    opener = STRENGTH_OPENERS[band]
    concern_intro = CONCERN_INTRODUCERS[band]

    if concern:
        if band in ("strong", "good"):
            reason = (
                f"{current_title} at {company} — {strength_clause}. "
                f"{concern_intro} {concern}."
            )
        else:
            reason = (
                f"{current_title} ({yoe:.1f} yrs) — {strength_clause}. "
                f"{concern_intro} {concern}."
            )
    else:
        reason = (
            f"{current_title} at {company} — {strength_clause}."
        )

    # Trim to a safe length for CSV
    if len(reason) > 280:
        reason = reason[:277] + "..."

    return reason


def generate_reasons_for_top100(
    ranked: list[tuple[int, object]],  # list of (rank, ScoredCandidate)
    candidate_index: dict[str, dict],
    feat_index: dict[str, CandidateFeatures],
    trust_index: dict[str, TrustFlags],
    jd: JDRequirements,
) -> dict[str, str]:
    """
    Generate reasons for all 100 ranked candidates.
    Returns dict of candidate_id -> reason string.
    """
    reasons: dict[str, str] = {}
    for rank, scored in ranked:
        cid = scored.candidate_id
        candidate = candidate_index.get(cid) or {}
        feat = feat_index.get(cid)
        trust = trust_index.get(cid) or TrustFlags(candidate_id=cid)

        if feat is None:
            reasons[cid] = "Profile data unavailable."
            continue

        reasons[cid] = generate_reason(rank, candidate, feat, trust, jd)

    return reasons
