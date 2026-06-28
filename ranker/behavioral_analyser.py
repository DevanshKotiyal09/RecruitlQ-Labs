"""
Module 6: Behavioral Signal Analyser
Derives composite behavioral sub-scores from the 23 redrob_signals fields.
"""
from __future__ import annotations

from ranker.feature_extractor import CandidateFeatures


def compute_behavioral_composite(feat: CandidateFeatures) -> float:
    """
    Returns a 0–1 behavioral composite that acts as a multiplier
    on the technical fit score. Architecture decision: behavioral signals
    modulate rank within technical tiers but cannot substitute for
    technical requirements.

    Sub-scores:
      - Availability score  (35%)
      - Engagement score    (30%)
      - Reliability score   (20%)
      - Market interest     (15%)
    """
    availability = _availability_score(feat)
    engagement = _engagement_score(feat)
    reliability = _reliability_score(feat)
    market = _market_interest_score(feat)

    composite = (
        0.35 * availability
        + 0.30 * engagement
        + 0.20 * reliability
        + 0.15 * market
    )
    return min(1.0, max(0.0, composite))


def _availability_score(feat: CandidateFeatures) -> float:
    """
    Measures the probability that the candidate is genuinely available today.
    Components: last_active, open_to_work, notice_period.
    """
    # Recency of activity (decays to 0 over 12 months)
    days = feat.days_since_last_active
    if days <= 7:
        activity = 1.0
    elif days <= 30:
        activity = 0.9
    elif days <= 60:
        activity = 0.75
    elif days <= 90:
        activity = 0.6
    elif days <= 180:
        activity = 0.4
    elif days <= 365:
        activity = 0.2
    else:
        activity = 0.05

    # Open to work flag
    open_score = 1.0 if feat.open_to_work else 0.35

    # Notice period
    notice = feat.notice_period_days
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.45
    elif notice <= 120:
        notice_score = 0.25
    else:
        notice_score = 0.1

    return activity * 0.45 + open_score * 0.25 + notice_score * 0.30


def _engagement_score(feat: CandidateFeatures) -> float:
    """
    Measures how actively the candidate is engaging with recruiters and the platform.
    """
    # Recruiter response rate (direct signal)
    rr = feat.recruiter_response_rate

    # Response time score (faster = better)
    rt = feat.avg_response_time_hours
    if rt <= 4:
        rt_score = 1.0
    elif rt <= 12:
        rt_score = 0.9
    elif rt <= 24:
        rt_score = 0.8
    elif rt <= 48:
        rt_score = 0.65
    elif rt <= 72:
        rt_score = 0.5
    elif rt <= 120:
        rt_score = 0.35
    else:
        rt_score = 0.15

    # Active job-seeking signals
    apps_score = min(1.0, feat.applications_submitted_30d / 5.0)

    return rr * 0.5 + rt_score * 0.3 + apps_score * 0.2


def _reliability_score(feat: CandidateFeatures) -> float:
    """
    Measures historical follow-through: do they show up and accept offers.
    """
    # Interview completion rate
    icr = feat.interview_completion_rate

    # Offer acceptance rate (-1 = no history, treat as neutral)
    oar = feat.offer_acceptance_rate
    if oar < 0:
        oar_score = 0.65  # neutral prior
    elif oar > 0.7:
        oar_score = 1.0
    elif oar > 0.4:
        oar_score = 0.75
    elif oar > 0.2:
        oar_score = 0.5
    else:
        oar_score = 0.25  # very low acceptance rate (uses offers as leverage)

    # Profile completeness as a proxy for professional investment
    comp_score = min(1.0, feat.profile_completeness / 100.0)

    # Verified contacts add trust
    verified_score = feat.verified_contact_count / 3.0

    return icr * 0.35 + oar_score * 0.3 + comp_score * 0.2 + verified_score * 0.15


def _market_interest_score(feat: CandidateFeatures) -> float:
    """
    Measures independent market validation by recruiters.
    High saved/views means other recruiters independently value this candidate.
    """
    # Saved by recruiters (strong signal: active recruiter intent)
    saved = min(1.0, feat.saved_by_recruiters_30d / 10.0)

    # Profile views (passive interest)
    views = min(1.0, feat.profile_views_30d / 50.0)

    # Search appearances (platform discovery signal)
    searches = min(1.0, feat.search_appearance_30d / 200.0)

    # GitHub activity (open contribution signal)
    if feat.github_activity_score < 0:
        github = 0.3  # no GitHub: neutral, not negative
    else:
        github = min(1.0, feat.github_activity_score / 80.0)

    return saved * 0.40 + views * 0.25 + searches * 0.20 + github * 0.15
