"""
Module 5: Anti-Trap Detector
Detects honeypots, keyword stuffing, behavioral twins, fake experts,
impossible timelines, title inflation, and other profile integrity issues.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ranker.jd_parser import (
    RETRIEVAL_KEYWORDS,
    NLP_KEYWORDS,
    CV_SPEECH_KEYWORDS,
    LLM_WRAPPER_KEYWORDS,
    extract_year_from_date,
    text_contains_any,
    count_keyword_hits,
)
from ranker.feature_extractor import CandidateFeatures, REFERENCE_DATE

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
SEVERITY_NONE = 0
SEVERITY_LOW = 1      # small soft penalty
SEVERITY_MEDIUM = 2   # moderate penalty
SEVERITY_HIGH = 3     # severe penalty
SEVERITY_HONEYPOT = 4 # hard filter: score approaches zero


@dataclass
class TrustFlags:
    candidate_id: str = ""
    # Accumulated severity flags
    flags: list[tuple[str, int, str]] = field(default_factory=list)  # (flag_name, severity, reason)
    # Combined outputs
    honeypot_risk: float = 0.0        # 0–1
    trust_penalty_multiplier: float = 1.0  # 1.0 = no penalty; approaches 0 for severe issues

    def add_flag(self, name: str, severity: int, reason: str) -> None:
        self.flags.append((name, severity, reason))
        self._recompute()

    def _recompute(self) -> None:
        """Recompute trust_penalty_multiplier from all flags."""
        max_sev = max((s for _, s, _ in self.flags), default=0)

        # Honeypot: near-zero score
        if max_sev >= SEVERITY_HONEYPOT:
            self.honeypot_risk = 1.0
            self.trust_penalty_multiplier = 0.02
            return

        # High severity: heavy penalty
        high_count = sum(1 for _, s, _ in self.flags if s >= SEVERITY_HIGH)
        med_count = sum(1 for _, s, _ in self.flags if s == SEVERITY_MEDIUM)
        low_count = sum(1 for _, s, _ in self.flags if s == SEVERITY_LOW)

        # Multiplicative penalty stacking
        penalty = 1.0
        penalty *= (0.4 ** high_count)
        penalty *= (0.75 ** med_count)
        penalty *= (0.92 ** low_count)

        self.trust_penalty_multiplier = max(0.02, penalty)
        self.honeypot_risk = min(1.0, (high_count * 0.4 + med_count * 0.2))

    def has_flag(self, name: str) -> bool:
        return any(f[0] == name for f in self.flags)

    def max_severity(self) -> int:
        return max((s for _, s, _ in self.flags), default=0)


# ---------------------------------------------------------------------------
# Honeypot detection helpers
# ---------------------------------------------------------------------------

def _safe_date_to_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _check_timeline_impossibility(
    candidate: dict,
    flags: TrustFlags,
) -> None:
    """
    Check for impossible or highly suspicious career timelines.
    """
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    education = candidate.get("education") or []

    declared_years = float(profile.get("years_of_experience") or 0)

    # Get earliest possible start year from education
    earliest_edu_end = None
    for edu in education:
        end_yr = edu.get("end_year")
        if end_yr:
            if earliest_edu_end is None or end_yr < earliest_edu_end:
                earliest_edu_end = end_yr

    # Sum of career durations vs declared experience
    total_career_months = sum(max(0, j.get("duration_months") or 0) for j in career)
    declared_months = int(declared_years * 12)

    # Flag large gaps (> 4 years discrepancy)
    gap_months = abs(total_career_months - declared_months)
    if gap_months > 48:
        flags.add_flag(
            "timeline_gap_large",
            SEVERITY_MEDIUM,
            f"Declared {declared_years:.1f}yrs vs career history "
            f"{total_career_months/12:.1f}yrs (gap={gap_months}mo)"
        )

    # Check for future start dates
    for job in career:
        start = _safe_date_to_date(job.get("start_date"))
        if start and start > REFERENCE_DATE:
            flags.add_flag(
                "future_start_date",
                SEVERITY_HONEYPOT,
                f"Career role starts in the future: {start} at {job.get('company')}"
            )
            break

    # Check if role duration claims predate education completion
    if earliest_edu_end:
        for job in career:
            start = _safe_date_to_date(job.get("start_date"))
            if start and start.year < (earliest_edu_end - 1):
                # Started professional role before finishing education (>1 yr overlap suspicious)
                dur = job.get("duration_months") or 0
                if dur > 24:  # long overlapping role is suspicious
                    flags.add_flag(
                        "role_before_education",
                        SEVERITY_MEDIUM,
                        f"Long role ({dur}mo) at {job.get('company')} "
                        f"started {start.year}, education ended {earliest_edu_end}"
                    )
                    break

    # Check for total career months vs age proxy
    # If total career > 35 years worth of months, something is wrong
    if total_career_months > 420:
        flags.add_flag(
            "implausible_total_experience",
            SEVERITY_HIGH,
            f"Total career history = {total_career_months/12:.1f} years"
        )


def _check_skill_impossibility(
    candidate: dict,
    flags: TrustFlags,
) -> None:
    """
    Check for impossible skill claims (expert with 0 duration, etc.)
    """
    skills = candidate.get("skills") or []
    career_history = candidate.get("career_history") or []

    # Get earliest possible professional year
    profile = candidate.get("profile") or {}
    declared_years = float(profile.get("years_of_experience") or 0)
    career_start_year = REFERENCE_DATE.year - max(1, int(declared_years))

    expert_zero_duration = 0
    impossible_duration_sum = 0

    for sk in skills:
        name = sk.get("name") or ""
        prof = sk.get("proficiency") or "beginner"
        dur = sk.get("duration_months") or 0
        end = sk.get("endorsements") or 0

        # Expert with 0 months duration is impossible
        if prof == "expert" and dur == 0:
            expert_zero_duration += 1

        # Advanced with 0 months and 0 endorsements
        if prof in {"advanced", "expert"} and dur == 0 and end == 0:
            expert_zero_duration += 1

        # Duration exceeding career length
        max_possible = int(declared_years * 12) + 6
        if dur > max_possible and max_possible > 0:
            impossible_duration_sum += 1

    if expert_zero_duration >= 3:
        flags.add_flag(
            "expert_zero_duration",
            SEVERITY_HONEYPOT,
            f"{expert_zero_duration} skills claimed expert/advanced with 0 duration and 0 endorsements"
        )
    elif expert_zero_duration >= 2:
        flags.add_flag(
            "expert_zero_duration",
            SEVERITY_HIGH,
            f"{expert_zero_duration} expert/advanced skills with 0 duration"
        )

    if impossible_duration_sum >= 3:
        flags.add_flag(
            "impossible_skill_duration",
            SEVERITY_HIGH,
            f"{impossible_duration_sum} skills with duration exceeding career length"
        )


def _check_keyword_stuffing(
    candidate: dict,
    features: CandidateFeatures,
    flags: TrustFlags,
) -> None:
    """
    Detect skills section inflated with AI keywords not corroborated by career descriptions.
    """
    skills = candidate.get("skills") or []
    career_history = candidate.get("career_history") or []

    skills_text = " ".join(s.get("name", "").lower() for s in skills)
    career_text = " ".join(
        (j.get("description") or "") + " " + (j.get("title") or "")
        for j in career_history
    ).lower()

    skill_ai_hits = count_keyword_hits(skills_text, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
    career_ai_hits = count_keyword_hits(career_text, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)

    if skill_ai_hits >= 6 and career_ai_hits == 0:
        flags.add_flag(
            "keyword_stuffing_severe",
            SEVERITY_HIGH,
            f"Skills contain {skill_ai_hits} AI keywords with zero career description corroboration"
        )
    elif skill_ai_hits >= 4 and career_ai_hits <= 1:
        flags.add_flag(
            "keyword_stuffing_moderate",
            SEVERITY_MEDIUM,
            f"Skills contain {skill_ai_hits} AI keywords; career descriptions contain {career_ai_hits}"
        )

    # Check for suspiciously many skills overall
    if len(skills) > 25:
        # Only flag if skill_duration_mean is very low (bulk-listed)
        avg_dur = sum(s.get("duration_months") or 0 for s in skills) / len(skills)
        if avg_dur < 8.0:
            flags.add_flag(
                "skill_list_inflated",
                SEVERITY_LOW,
                f"{len(skills)} skills with mean duration {avg_dur:.1f}mo"
            )


def _check_behavioral_contradictions(
    candidate: dict,
    flags: TrustFlags,
) -> None:
    """
    Check for contradictions between behavioral signals.
    """
    signals = candidate.get("redrob_signals") or {}
    last_active = _safe_date_to_date(signals.get("last_active_date"))

    # Active signal vs inactivity
    days_since = (REFERENCE_DATE - last_active).days if last_active else 999
    offer_rate = float(signals.get("offer_acceptance_rate") or -1.0)
    interview_rate = float(signals.get("interview_completion_rate") or 0.0)
    recruiter_rate = float(signals.get("recruiter_response_rate") or 0.0)
    apps = int(signals.get("applications_submitted_30d") or 0)
    open_work = bool(signals.get("open_to_work_flag") or False)

    # Contradiction: highly engaged signals with very old last-active date
    high_engagement = (offer_rate > 0.8 and interview_rate > 0.9 and recruiter_rate > 0.8)
    if high_engagement and days_since > 365:
        flags.add_flag(
            "engagement_vs_inactivity_contradiction",
            SEVERITY_MEDIUM,
            f"Very high engagement scores (offer={offer_rate:.2f}, interview={interview_rate:.2f}) "
            f"but inactive for {days_since} days"
        )

    # Open to work flag vs no applications and no views
    if open_work and apps == 0 and days_since > 120:
        flags.add_flag(
            "open_to_work_passive_contradiction",
            SEVERITY_LOW,
            "open_to_work=True but no recent applications and long inactivity"
        )


def _check_title_inflation(
    candidate: dict,
    flags: TrustFlags,
) -> None:
    """
    Check if titles appear inflated vs career description substance.
    """
    career = candidate.get("career_history") or []

    leadership_titles = frozenset({
        "principal", "staff", "distinguished", "fellow",
        "head of", "vp", "vice president", "director", "cto", "ceo",
    })
    technical_substance_keywords = frozenset({
        "design", "architect", "system design", "lead", "own", "built",
        "deploy", "production", "scale", "mentor", "review", "strategy",
        "roadmap", "cross-functional",
    })

    inflated_count = 0
    for job in career:
        title = (job.get("title") or "").lower()
        desc = (job.get("description") or "").lower()
        if any(lt in title for lt in leadership_titles):
            if not text_contains_any(desc, technical_substance_keywords):
                inflated_count += 1

    if inflated_count >= 2:
        flags.add_flag(
            "title_inflation",
            SEVERITY_MEDIUM,
            f"{inflated_count} senior/leadership titles with no substance in descriptions"
        )
    elif inflated_count == 1:
        flags.add_flag(
            "title_inflation_minor",
            SEVERITY_LOW,
            "One senior title with limited description substance"
        )


def _check_corroboration_absence(
    candidate: dict,
    flags: TrustFlags,
) -> None:
    """
    Detect profiles where all keyword signals are present in skills
    but career descriptions have no matching professional content.
    This is a key honeypot signal.
    """
    profile = candidate.get("profile") or {}
    skills = candidate.get("skills") or []
    career = candidate.get("career_history") or []

    current_title = (profile.get("current_title") or "").lower()
    career_text = " ".join(
        (j.get("description") or "") for j in career
    ).lower()

    # Strong AI title but career descriptions are completely non-technical
    ai_title = any(kw in current_title for kw in {
        "ai", "ml", "machine learning", "data scientist", "nlp", "engineer"
    })

    if ai_title and career_text:
        ai_career_hits = count_keyword_hits(career_text, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
        # But career text mentions only generic non-technical work
        non_tech_phrases = frozenset({
            "marketing", "accounting", "customer support", "sales", "hr manager",
            "human resources", "mechanical", "civil", "chemical", "content writer",
            "operations manager", "project manager",
        })
        non_tech_hits = sum(1 for p in non_tech_phrases if p in career_text)

        if ai_career_hits == 0 and non_tech_hits >= 3:
            flags.add_flag(
                "ai_title_non_tech_career",
                SEVERITY_HIGH,
                f"AI-related current title '{current_title}' but career descriptions "
                f"contain no AI content and {non_tech_hits} non-tech markers"
            )


def _behavioral_signal_hash(signals: dict) -> str:
    """Create a hash of behavioral signals for twin detection."""
    relevant = {
        k: signals.get(k)
        for k in [
            "profile_completeness_score", "recruiter_response_rate",
            "interview_completion_rate", "offer_acceptance_rate",
            "github_activity_score", "notice_period_days",
        ]
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-candidate trust evaluation
# ---------------------------------------------------------------------------

def evaluate_trust(candidate: dict, features: CandidateFeatures) -> TrustFlags:
    """
    Run all trust/integrity checks for one candidate.
    Returns a TrustFlags object with all detected violations.
    """
    flags = TrustFlags(candidate_id=features.candidate_id)

    # 1. Timeline impossibility
    _check_timeline_impossibility(candidate, flags)

    # 2. Skill impossibility
    _check_skill_impossibility(candidate, flags)

    # 3. Keyword stuffing
    _check_keyword_stuffing(candidate, features, flags)

    # 4. Behavioral contradictions
    _check_behavioral_contradictions(candidate, flags)

    # 5. Title inflation
    _check_title_inflation(candidate, flags)

    # 6. Corroboration absence (AI title + non-tech career)
    _check_corroboration_absence(candidate, flags)

    # 7. Stale-profile risk (not a penalty but affects availability)
    signals = candidate.get("redrob_signals") or {}
    days = features.days_since_last_active
    rr = features.recruiter_response_rate
    open_work = features.open_to_work
    if days > 180 and rr < 0.15 and not open_work:
        flags.add_flag(
            "stale_profile",
            SEVERITY_LOW,
            f"Inactive {days}d, response rate {rr:.2f}, not open to work"
        )

    return flags


# ---------------------------------------------------------------------------
# Batch behavioral twin detection
# Run after all candidates have been processed.
# ---------------------------------------------------------------------------

def detect_behavioral_twins(
    candidates_and_features: list[tuple[dict, CandidateFeatures]]
) -> dict[str, TrustFlags]:
    """
    Detect candidates with identical behavioral signal hashes (twins).
    Returns a dict of candidate_id -> TrustFlags for flagged candidates only.
    Does not re-run full trust evaluation — only adds twin flags.
    """
    hash_to_ids: dict[str, list[str]] = {}

    for cand, feat in candidates_and_features:
        signals = cand.get("redrob_signals") or {}
        sig_hash = _behavioral_signal_hash(signals)
        cid = feat.candidate_id
        hash_to_ids.setdefault(sig_hash, []).append(cid)

    twin_flags: dict[str, TrustFlags] = {}
    for sig_hash, ids in hash_to_ids.items():
        if len(ids) > 3:  # more than 3 identical profiles = suspicious
            for cid in ids:
                if cid not in twin_flags:
                    twin_flags[cid] = TrustFlags(candidate_id=cid)
                twin_flags[cid].add_flag(
                    "behavioral_twin",
                    SEVERITY_MEDIUM,
                    f"Behavioral signals identical to {len(ids)-1} other candidates"
                )

    return twin_flags
