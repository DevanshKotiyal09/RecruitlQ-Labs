"""
Module 7: Composite Scorer
Combines all feature sub-scores into a single final score per candidate.
Design principle: career descriptions beat declared attributes;
behavioral signals are multiplicative (modulate tier, never substitute for it).
"""
from __future__ import annotations

import math
from ranker.feature_extractor import CandidateFeatures
from ranker.anti_trap import TrustFlags
from ranker.behavioral_analyser import compute_behavioral_composite
from ranker.jd_parser import JDRequirements


# ---------------------------------------------------------------------------
# Hard-filter score cap
# Candidates failing a hard filter get a score at or below this cap,
# keeping them outside any realistic top-100 regardless of pool size.
# ---------------------------------------------------------------------------
HARD_FILTER_CAP = 0.08


def compute_composite_score(
    feat: CandidateFeatures,
    trust: TrustFlags,
    jd: JDRequirements,
) -> tuple[float, float]:
    """
    Returns (composite_score, confidence_score) both in [0, 1].

    composite_score: final ranking score
    confidence_score: how much evidence supports the composite (used for tiebreaking)
    """
    # -----------------------------------------------------------------------
    # Stage 1: Hard filters
    # -----------------------------------------------------------------------
    if _fails_hard_filter(feat, trust, jd):
        # Add a per-candidate differentiator so hard-filtered candidates have
        # distinct scores that satisfy the tie-break ordering rule.
        # id_num ranges 0–9999999; dividing by 1e8 gives range 0–0.1 — too large.
        # Divide by 1e9 to stay within [HARD_FILTER_CAP - 0.01, HARD_FILTER_CAP + 0.01].
        # Lower candidate_id number → higher nudge → ranks higher when scores equal.
        id_num = int(feat.candidate_id.split("_")[-1])
        # Invert so smaller ID = higher score (ascending rank = smaller ID)
        max_id = 9999999
        id_nudge = (max_id - id_num) / 1e9
        base = HARD_FILTER_CAP * feat.profile_consistency_score
        return base + id_nudge, 0.1

    # -----------------------------------------------------------------------
    # Stage 2: Technical fit score (primary component)
    # -----------------------------------------------------------------------
    tech_score = _technical_fit(feat)

    # -----------------------------------------------------------------------
    # Stage 3: Career quality score
    # -----------------------------------------------------------------------
    career_score = _career_quality(feat)

    # -----------------------------------------------------------------------
    # Stage 4: Education score (minor)
    # -----------------------------------------------------------------------
    edu_score = _education_fit(feat)

    # -----------------------------------------------------------------------
    # Stage 5: Minor bonuses
    # -----------------------------------------------------------------------
    bonus = _compute_bonuses(feat)

    # -----------------------------------------------------------------------
    # Stage 6: Penalty components (additive deductions)
    # -----------------------------------------------------------------------
    penalty = _compute_penalties(feat)

    # -----------------------------------------------------------------------
    # Stage 7: Location score
    # -----------------------------------------------------------------------
    loc_score = feat.location_score

    # -----------------------------------------------------------------------
    # Stage 8: Trust penalty multiplier (from anti-trap module)
    # -----------------------------------------------------------------------
    trust_mult = trust.trust_penalty_multiplier

    # -----------------------------------------------------------------------
    # Stage 9: Composite assembly
    # Weights reflect architecture spec: technical fit dominates
    # -----------------------------------------------------------------------
    raw = (
        tech_score   * 0.42
        + career_score * 0.25
        + loc_score    * 0.08
        + edu_score    * 0.06
        + bonus        * 0.05
    )

    # Apply penalty (subtractive, capped so it can't invert the score)
    raw = raw * (1.0 - penalty * 0.5) - penalty * 0.05
    raw = max(0.0, raw)

    # Apply trust multiplier
    raw = raw * trust_mult

    # -----------------------------------------------------------------------
    # Stage 10: Behavioral composite as multiplier
    # Behavioral signals modulate rank within tier; cannot substitute for tech.
    # Multiplier range: [0.2, 1.0] — even perfect-tech + zero-behavioral = 0.2×tech
    # -----------------------------------------------------------------------
    behavioral = compute_behavioral_composite(feat)
    behavioral_mult = 0.20 + 0.80 * behavioral

    composite = raw * behavioral_mult

    # Clamp to [0, 1]
    composite = min(1.0, max(0.0, composite))

    # -----------------------------------------------------------------------
    # Confidence score: how much evidence backs the composite
    # -----------------------------------------------------------------------
    confidence = _compute_confidence(feat, tech_score, career_score)

    return composite, confidence


# ---------------------------------------------------------------------------
# Hard filter logic
# ---------------------------------------------------------------------------

def _fails_hard_filter(
    feat: CandidateFeatures,
    trust: TrustFlags,
    jd: JDRequirements,
) -> bool:
    """Returns True if candidate should receive the hard-filter score cap."""

    # Honeypot
    if trust.honeypot_risk >= 0.8:
        return True

    # Confirmed honeypot flag
    if trust.has_flag("future_start_date"):
        return True
    if trust.has_flag("expert_zero_duration") and trust.max_severity() >= 4:
        return True
    if trust.has_flag("ai_title_non_tech_career") and trust.max_severity() >= 3:
        return True

    # Non-India, not willing to relocate
    if feat.location_category == "international_no_relocate":
        return True

    # Pure research career (JD says "we will not move forward")
    if feat.research_only_flag:
        return True

    # Zero technical relevance AND zero AI skill corroboration
    if (feat.current_title_ai_relevance == 0.0
            and feat.career_direction_score < 0.02
            and feat.jd_core_skill_corroborated == 0
            and feat.production_deployment_evidence < 0.02):
        return True

    return False


# ---------------------------------------------------------------------------
# Technical fit components
# ---------------------------------------------------------------------------

def _technical_fit(feat: CandidateFeatures) -> float:
    """
    Measures how well the candidate's demonstrated technical background
    matches the must-have requirements. This is the heaviest component.

    Sub-components:
      - Production retrieval/search evidence (most important)
      - Python engineering signal
      - Evaluation framework knowledge
      - Semantic similarity to JD
      - NLP / IR depth
      - Skill assessment corroboration
    """
    # --- Production retrieval evidence ---
    # This is THE core requirement: shipped embedding/retrieval in production
    prod_retrieval = (
        feat.production_deployment_evidence * 0.45
        + feat.retrieval_career_evidence * 0.35
        + feat.retrieval_skill_depth * 0.20
    )

    # --- Python engineering signal ---
    # python_quality_signal rewards engineering craft (testing, async, packaging, etc.)
    # beyond simple "python" mentions; blended in conservatively.
    python_base = (
        feat.python_career_evidence * 0.6
        + feat.python_skill_signal * 0.4
    )
    python_signal = min(1.0, python_base + feat.python_quality_signal * 0.15)

    # --- Evaluation framework knowledge ---
    eval_signal = (
        feat.eval_career_evidence * 0.6
        + feat.eval_skill_signal * 0.4
    )

    # --- Semantic similarity to JD ---
    # Use max of the three probes to reward single-domain strength
    semantic = max(
        feat.jd_semantic_similarity,
        feat.retrieval_probe_similarity,
        feat.nlp_probe_similarity * 0.8,  # NLP slightly less critical than retrieval
    )

    # --- NLP / IR depth ---
    nlp_ir = (
        feat.nlp_career_evidence * 0.5
        + feat.nlp_ir_skill_depth * 0.3
        + feat.career_narrative_nlp_score * 0.2
    )

    # --- Platform assessment corroboration ---
    assessment = feat.skill_assessment_composite

    # --- LLM fine-tuning (preferred, not required) ---
    llm_bonus = feat.llm_finetune_skill * 0.3

    # --- Career direction overall ---
    direction = feat.career_direction_score

    tech = (
        prod_retrieval * 0.35
        + python_signal * 0.15
        + eval_signal  * 0.12
        + semantic     * 0.18
        + nlp_ir       * 0.10
        + assessment   * 0.05
        + direction    * 0.05
    )

    # Semantic similarity lift for plain-language matches:
    # If semantic is high but keyword-based scores are low, boost proportionally
    if feat.jd_semantic_similarity > 0.40 and prod_retrieval < 0.2:
        semantic_rescue = (feat.jd_semantic_similarity - 0.40) * 0.5
        tech = min(1.0, tech + semantic_rescue)

    return min(1.0, max(0.0, tech))


# ---------------------------------------------------------------------------
# Career quality components
# ---------------------------------------------------------------------------

def _career_quality(feat: CandidateFeatures) -> float:
    """
    Measures the trajectory and professional environment of the career.
    """
    # Product company experience
    product_score = feat.product_fraction

    # Stability / commitment signal (longest tenure)
    if feat.longest_tenure_months >= 36:
        stability = 1.0
    elif feat.longest_tenure_months >= 24:
        stability = 0.75
    elif feat.longest_tenure_months >= 12:
        stability = 0.5
    else:
        stability = 0.2

    # Experience band fit
    yoe = feat.years_of_experience
    if 5 <= yoe <= 9:
        exp_band = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        exp_band = 0.8
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        exp_band = 0.55
    elif yoe < 3:
        exp_band = 0.3
    else:  # > 15 years
        exp_band = 0.4

    # Pre-LLM ML evidence (direct JD requirement signal)
    pre_llm = 1.0 if feat.pre_llm_ml_evidence else 0.3

    # Title progression toward AI/ML
    title_rel = feat.current_title_ai_relevance

    # Consulting-dominant penalty (not all consulting, just mostly)
    if feat.consulting_fraction > 0.8:
        consulting_penalty = 0.3
    elif feat.consulting_fraction > 0.5:
        consulting_penalty = 0.6
    else:
        consulting_penalty = 1.0

    career = (
        product_score * 0.30
        + stability   * 0.20
        + exp_band    * 0.20
        + pre_llm     * 0.15
        + title_rel   * 0.15
    ) * consulting_penalty

    return min(1.0, max(0.0, career))


# ---------------------------------------------------------------------------
# Education fit
# ---------------------------------------------------------------------------

def _education_fit(feat: CandidateFeatures) -> float:
    """Minor education signal — field relevance + tier + degree level."""
    field_score = feat.education_field_relevance

    tier = feat.education_tier_best
    if tier == 1:
        tier_score = 1.0
    elif tier == 2:
        tier_score = 0.75
    elif tier == 3:
        tier_score = 0.5
    else:
        tier_score = 0.3

    degree_map = {0: 0.4, 1: 0.7, 2: 0.85, 3: 0.7}  # PhD slightly less for this role
    degree_score = degree_map.get(feat.highest_degree_level, 0.5)

    ai_bonus = 0.15 if feat.education_ai_ml_focus else 0.0

    return min(1.0, field_score * 0.5 + tier_score * 0.25 + degree_score * 0.15 + ai_bonus)


# ---------------------------------------------------------------------------
# Bonuses
# ---------------------------------------------------------------------------

def _compute_bonuses(feat: CandidateFeatures) -> float:
    """Compute all additive bonus components (0–1 scale)."""
    total = 0.0

    # Evaluation framework knowledge bonus
    if feat.eval_career_evidence > 0.3 or feat.eval_skill_signal > 0.5:
        total += 0.25

    # LLM fine-tuning bonus (preferred signal)
    if feat.llm_finetune_skill > 0.4:
        total += 0.2

    # GitHub activity bonus
    if feat.github_activity_score > 50:
        total += 0.2
    elif feat.github_activity_score > 20:
        total += 0.1

    # Relevant certifications
    if feat.relevant_cert_count >= 2:
        total += 0.15
    elif feat.relevant_cert_count == 1:
        total += 0.07

    # Platform engagement bonus (saved_by_recruiters is strong signal)
    if feat.saved_by_recruiters_30d >= 10:
        total += 0.15
    elif feat.saved_by_recruiters_30d >= 5:
        total += 0.08

    # Skill assessment score bonus (platform-validated skills)
    if feat.skill_assessment_composite > 0.7:
        total += 0.15
    elif feat.skill_assessment_composite > 0.4:
        total += 0.07

    # Preferred skills bonus — conservative, must not overpower must-have skills.
    # Corroborated preferred skills add a small signal; raw count alone is weaker.
    if feat.preferred_skill_corroborated >= 3:
        total += 0.12
    elif feat.preferred_skill_corroborated >= 1:
        total += 0.06
    elif feat.preferred_skill_count >= 2:
        total += 0.03  # unverified but non-zero preferred skill breadth

    # Python engineering quality bonus (craft beyond just knowing Python)
    if feat.python_quality_signal > 0.3:
        total += 0.08
    elif feat.python_quality_signal > 0.1:
        total += 0.04

    return min(1.0, total)


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

def _compute_penalties(feat: CandidateFeatures) -> float:
    """
    Compute penalty factor [0, 1].
    A penalty of 1.0 effectively nullifies the raw score.
    """
    penalty = 0.0

    # Consulting-only career (JD explicitly says bad fit)
    if feat.consulting_only_flag:
        penalty += 0.55

    # Research-only career (JD says "will not move forward")
    if feat.research_only_flag:
        penalty += 0.65

    # Recent LLM-only experience without pre-LLM ML background
    if feat.recent_llm_only:
        penalty += 0.40

    # Inactive architect (not coding in 18+ months)
    if feat.inactive_architect_flag:
        penalty += 0.30

    # CV / speech domain dominance (JD says re-learning fundamentals)
    if feat.cv_speech_dominance > 0.6:
        penalty += 0.30
    elif feat.cv_speech_dominance > 0.4:
        penalty += 0.15

    # CV / speech skill dominance in skills section
    if feat.cv_speech_skill_fraction > 0.5:
        penalty += 0.20

    # Very short average tenure (title-chasing pattern)
    if feat.avg_tenure_months < 12 and feat.num_positions >= 3:
        penalty += 0.20
    elif feat.avg_tenure_months < 18 and feat.num_positions >= 4:
        penalty += 0.10

    # Keyword stuffing penalty
    if feat.skill_keyword_density_ratio > 3.0:
        penalty += 0.25
    elif feat.skill_keyword_density_ratio > 2.0:
        penalty += 0.12

    # Long notice period penalty
    if feat.notice_period_days > 120:
        penalty += 0.12
    elif feat.notice_period_days > 90:
        penalty += 0.06

    # Salary incompatibility
    if feat.salary_band_compatible < 0.6:
        penalty += 0.08

    # Profile inconsistency
    if feat.profile_consistency_score < 0.5:
        penalty += 0.15
    elif feat.profile_consistency_score < 0.7:
        penalty += 0.07

    # Clamp penalty to max 0.90 (never fully zero from penalties alone)
    return min(0.90, penalty)


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _compute_confidence(
    feat: CandidateFeatures,
    tech_score: float,
    career_score: float,
) -> float:
    """
    Returns 0–1 confidence: how much evidence backs the composite score.
    High confidence = rich career descriptions + corroborated skills.
    Used for tiebreaking; higher confidence beats lower at equal composite.
    """
    evidence_points = 0.0

    # Career description depth
    if feat.production_deployment_evidence > 0.3:
        evidence_points += 0.30
    elif feat.production_deployment_evidence > 0.1:
        evidence_points += 0.15

    # Corroborated skill count
    if feat.jd_core_skill_corroborated >= 3:
        evidence_points += 0.25
    elif feat.jd_core_skill_corroborated >= 1:
        evidence_points += 0.12

    # Assessment scores available
    if feat.skill_assessment_composite > 0.3:
        evidence_points += 0.15

    # Verified contacts
    evidence_points += feat.verified_contact_count / 3.0 * 0.10

    # Profile completeness
    evidence_points += (feat.profile_completeness / 100.0) * 0.10

    # Profile consistency
    evidence_points += feat.profile_consistency_score * 0.10

    return min(1.0, evidence_points)
