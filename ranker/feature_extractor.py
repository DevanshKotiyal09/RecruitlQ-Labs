"""
Module 3: Feature Extractor
Derives every numeric and categorical feature for a candidate from their raw profile.
All operations are pure arithmetic / string matching — no model inference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from ranker.jd_parser import (
    JDRequirements,
    PRODUCTION_KEYWORDS,
    RETRIEVAL_KEYWORDS,
    NLP_KEYWORDS,
    CV_SPEECH_KEYWORDS,
    RESEARCH_KEYWORDS,
    LLM_WRAPPER_KEYWORDS,
    EVAL_KEYWORDS,
    text_contains_any,
    count_keyword_hits,
    extract_year_from_date,
)

# ---------------------------------------------------------------------------
# Reference date for "days since active" calculations
# ---------------------------------------------------------------------------
REFERENCE_DATE = date(2026, 6, 1)  # dataset reference point

# ---------------------------------------------------------------------------
# Company taxonomy
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = frozenset({
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    "mindtree", "niit technologies", "zensar", "mastech",
})

ACADEMIC_INSTITUTIONS = frozenset({
    "iit", "iim", "university", "college", "institute", "school",
    "academia", "laboratory", "lab", "research center", "centre",
})

PRODUCT_COMPANY_KEYWORDS = frozenset({
    "google", "meta", "microsoft", "amazon", "apple", "netflix", "uber",
    "airbnb", "linkedin", "twitter", "spotify", "stripe", "databricks",
    "openai", "anthropic", "cohere", "hugging face", "huggingface",
    "zomato", "swiggy", "flipkart", "ola", "paytm", "phonepe",
    "razorpay", "cred", "meesho", "dream11", "byju", "unacademy",
    "freshworks", "zoho", "browserstack", "postman", "chargebee",
    "redrob", "pied piper", "hooli", "initech",  # fictional product cos in dataset
    "startup", "series a", "series b", "saas", "product company",
})

PREFERRED_LOCATIONS = frozenset({
    "pune", "noida",
})

ACCEPTABLE_INDIA_LOCATIONS = frozenset({
    "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
    "bengaluru", "bangalore", "new delhi", "ncr", "kolkata", "chennai",
    "ahmedabad", "india",
})

# Degree field relevance mapping
STEM_AI_FIELDS = frozenset({
    "computer science", "computer engineering", "software engineering",
    "artificial intelligence", "machine learning", "data science",
    "statistics", "mathematics", "maths", "physics", "information technology",
    "information science", "computational", "electronics", "electrical engineering",
    "signal processing",
})

NON_STEM_FIELDS = frozenset({
    "marketing", "finance", "accounting", "business administration",
    "mba", "management", "commerce", "humanities", "arts", "history",
    "chemistry", "biology", "mechanical engineering", "civil engineering",
    "chemical engineering",
})

# Skills that directly map to JD must-haves
JD_CORE_SKILLS = frozenset({
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "sentence-transformers",
    "vector database", "vector search", "hybrid search",
    "bm25", "dense retrieval", "semantic search",
    "ranking", "learning to rank", "ltr",
    "ndcg", "mrr", "information retrieval",
    "python",
    "recommendation systems", "recommender systems",
    "nlp", "natural language processing",
    "bert", "transformers", "hugging face",
    "rag", "retrieval augmented generation",
})

# Skills that map to JD preferred
JD_PREFERRED_SKILLS = frozenset({
    "lora", "qlora", "peft", "fine-tuning llms", "fine tuning",
    "xgboost", "lightgbm", "langchain", "llama index",
    "mlflow", "weights & biases", "wandb",
    "distributed systems", "kubernetes", "docker",
    "spark", "kafka", "airflow",
    "a/b testing", "experimentation",
})


# ---------------------------------------------------------------------------
# Feature record dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    candidate_id: str = ""

    # --- Profile features ---
    years_of_experience: float = 0.0
    experience_band: str = "unknown"          # under_band | in_band | over_band
    current_title_ai_relevance: float = 0.0  # 0–1
    current_company_is_consulting: bool = False
    current_company_is_product: bool = False
    current_industry_is_tech: bool = False
    profile_completeness: float = 0.0
    location_score: float = 0.0              # 0–1
    location_category: str = "unknown"
    country: str = ""
    is_india_based: bool = False
    willing_to_relocate: bool = False

    # --- Career features ---
    total_career_months: int = 0
    product_company_months: int = 0
    consulting_company_months: int = 0
    product_fraction: float = 0.0
    consulting_fraction: float = 0.0
    longest_tenure_months: int = 0
    avg_tenure_months: float = 0.0
    num_positions: int = 0
    career_direction_score: float = 0.0      # 0–1 toward AI/ML
    production_deployment_evidence: float = 0.0  # 0–1
    retrieval_career_evidence: float = 0.0   # 0–1
    nlp_career_evidence: float = 0.0         # 0–1
    eval_career_evidence: float = 0.0        # 0–1 (requires AI/ML context)
    python_career_evidence: float = 0.0      # 0–1
    python_quality_signal: float = 0.0       # 0–1 engineering quality beyond simple "python"
    research_career_fraction: float = 0.0    # fraction of career in research
    pre_llm_ml_evidence: bool = False        # ML work before 2022
    recent_llm_only: bool = False            # only post-2022 LLM wrapper work
    inactive_architect_flag: bool = False    # no code in 18+ months
    consulting_only_flag: bool = False
    research_only_flag: bool = False
    cv_speech_dominance: float = 0.0         # 0–1 fraction CV/speech focus

    # Weighted career narrative score for all descriptions combined
    career_narrative_retrieval_score: float = 0.0
    career_narrative_nlp_score: float = 0.0

    # --- Skill features ---
    retrieval_skill_depth: float = 0.0       # 0–1 (corroborated)
    python_skill_signal: float = 0.0         # 0–1
    eval_skill_signal: float = 0.0           # 0–1
    llm_finetune_skill: float = 0.0          # 0–1
    nlp_ir_skill_depth: float = 0.0          # 0–1
    cv_speech_skill_fraction: float = 0.0    # 0–1 (how much skill space is CV/speech)
    skill_keyword_density_ratio: float = 0.0 # AI kw in skills / in descriptions
    skill_assessment_composite: float = 0.0  # 0–1 from platform assessments
    skill_duration_mean: float = 0.0         # avg months across all skills
    num_skills: int = 0
    jd_core_skill_count: int = 0             # number of JD core skills listed
    jd_core_skill_corroborated: int = 0      # corroborated by career text
    preferred_skill_count: int = 0           # number of JD preferred skills listed
    preferred_skill_match_ratio: float = 0.0 # fraction of preferred skills matched
    preferred_skill_corroborated: int = 0    # corroborated preferred skills

    # --- Education features ---
    highest_degree_level: int = 0            # 1=BSc, 2=MSc, 3=PhD, 4=MBA
    education_field_relevance: float = 0.0   # 0–1
    education_tier_best: int = 4             # 1–4 (1=best)
    education_ai_ml_focus: bool = False

    # --- Certification features ---
    relevant_cert_count: int = 0

    # --- Behavioral features (from redrob_signals) ---
    days_since_last_active: int = 999
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 999.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    applications_submitted_30d: int = 0
    profile_views_30d: int = 0
    saved_by_recruiters_30d: int = 0
    search_appearance_30d: int = 0
    connection_count: int = 0
    github_activity_score: float = -1.0

    # --- Availability features ---
    notice_period_days: int = 90
    effective_availability_score: float = 0.0  # 0–1 composite

    # --- Trust features ---
    verified_contact_count: int = 0
    profile_consistency_score: float = 1.0    # 0–1 (1=consistent)
    experience_timeline_gap_months: int = 0

    # --- Salary features ---
    salary_min_lpa: float = 0.0
    salary_max_lpa: float = 0.0
    salary_band_compatible: float = 1.0       # 0–1

    # --- Risk flags ---
    honeypot_risk: float = 0.0               # 0–1 (set by anti-trap module)
    trust_penalty: float = 0.0               # 0–1 multiplier penalty

    # --- Semantic features (filled by semantic encoder) ---
    jd_semantic_similarity: float = 0.0
    retrieval_probe_similarity: float = 0.0
    nlp_probe_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(date_str: str | None, ref: date = REFERENCE_DATE) -> int:
    d = _safe_date(date_str)
    if d is None:
        return 999
    delta = (ref - d).days
    return max(0, delta)


def _is_consulting_company(company: str) -> bool:
    c = company.lower().strip()
    return any(firm in c for firm in CONSULTING_FIRMS)


def _is_product_company(company: str, industry: str = "") -> bool:
    c = company.lower()
    i = industry.lower()
    if _is_consulting_company(company):
        return False
    if any(kw in c for kw in PRODUCT_COMPANY_KEYWORDS):
        return True
    if any(kw in i for kw in {"software", "saas", "product", "fintech", "edtech"}):
        return True
    return False


def _is_academic(company: str, industry: str = "") -> bool:
    c = company.lower()
    i = industry.lower()
    if any(kw in c for kw in {"university", "college", "institute", "iit", "iim",
                               "research lab", "research center"}):
        return True
    if "academic" in i or "research" in i:
        return True
    return False


def _score_title_ai_relevance(title: str) -> float:
    t = title.lower()
    if any(x in t for x in {
        "machine learning", "ml engineer", "ai engineer", "nlp engineer",
        "search engineer", "ranking engineer", "relevance engineer",
        "applied scientist", "research scientist", "data scientist",
        "retrieval engineer", "recommendation engineer",
        "mlops", "ml platform", "applied ml",
    }):
        return 1.0
    if any(x in t for x in {
        "data engineer", "software engineer", "backend engineer",
        "platform engineer", "infrastructure engineer",
    }):
        return 0.35
    if any(x in t for x in {
        "analytics", "analyst", "statistician", "quantitative",
    }):
        return 0.2
    return 0.0


def _score_location(location: str, country: str, willing_to_relocate: bool) -> tuple[float, str]:
    loc = location.lower()
    cntry = country.lower()

    # India check
    is_india = ("india" in cntry) or any(
        city in loc for city in {
            "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
            "gurgaon", "gurugram", "bengaluru", "bangalore", "kolkata",
            "chennai", "ahmedabad",
        }
    )

    if any(city in loc for city in {"pune"}):
        return 1.0, "preferred"
    if any(city in loc for city in {"noida", "greater noida"}):
        return 1.0, "preferred"
    if any(city in loc for city in {
        "delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
        "mumbai", "bengaluru", "bangalore",
    }):
        return 0.75, "acceptable_india"
    if is_india:
        return 0.55, "other_india"
    if not is_india and willing_to_relocate:
        return 0.35, "international_relocatable"
    return 0.1, "international_no_relocate"


def _score_education_field(field_of_study: str) -> float:
    f = field_of_study.lower()
    if any(s in f for s in {
        "artificial intelligence", "machine learning", "data science",
        "computer science", "computer engineering",
    }):
        return 1.0
    if any(s in f for s in {
        "information technology", "information science", "statistics",
        "mathematics", "software engineering", "signal processing",
        "electronics", "electrical",
    }):
        return 0.7
    if any(s in f for s in {
        "physics", "computational", "cognitive science",
    }):
        return 0.5
    if any(s in f for s in NON_STEM_FIELDS):
        return 0.1
    return 0.3


def _degree_level(degree: str) -> int:
    d = degree.lower().strip()
    if any(x in d for x in {"phd", "ph.d", "doctorate", "d.phil"}):
        return 3
    if any(x in d for x in {"m.tech", "m.e.", "m.sc", "m.s.", "msc", "ms",
                              "master", "mba", "m.b.a", "pg", "m.eng"}):
        return 2
    if any(x in d for x in {"b.tech", "b.e.", "b.sc", "b.s.", "bsc", "bs",
                              "bachelor", "be ", "be."}):
        return 1
    return 0


def _compute_skill_depth(
    skills: list[dict],
    target_keywords: frozenset[str],
    career_text: str,
    min_duration_corroborate: int = 6,
    min_endorsements_corroborate: int = 3,
) -> float:
    """
    Score skill depth for a set of target keywords.
    Requires some corroboration from career text or duration/endorsements.
    Returns 0–1.
    """
    if not skills:
        return 0.0
    career_lower = career_text.lower()
    hits = 0
    corroborated = 0
    proficiency_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

    for sk in skills:
        name = sk.get("name", "").lower()
        if not any(kw in name for kw in target_keywords):
            continue
        hits += 1
        duration = sk.get("duration_months", 0) or 0
        endorsements = sk.get("endorsements", 0) or 0
        prof = proficiency_map.get(sk.get("proficiency", "beginner"), 0.25)
        # Corroboration: career description mentions the skill domain
        career_mention = any(kw in career_lower for kw in target_keywords if kw in name or len(kw) > 4)
        if duration >= min_duration_corroborate or endorsements >= min_endorsements_corroborate or career_mention:
            corroborated += prof
        else:
            corroborated += prof * 0.2  # unverified self-declaration

    if hits == 0:
        return 0.0
    # Normalize; cap at 1.0
    return min(1.0, corroborated / max(3.0, hits))


def _is_pre_llm_ml_evidence(career_history: list[dict]) -> bool:
    """
    Check if any role before 2022 contains ML/IR/search production evidence.
    """
    for job in career_history:
        start_year = extract_year_from_date(job.get("start_date"))
        if start_year is None or start_year >= 2022:
            continue
        desc = (job.get("description") or "").lower()
        title = (job.get("title") or "").lower()
        if text_contains_any(desc + " " + title, RETRIEVAL_KEYWORDS | NLP_KEYWORDS):
            return True
        # Also check for classic ML work
        if any(kw in desc for kw in {
            "machine learning", "neural network", "deep learning",
            "gradient boosting", "random forest", "xgboost", "sklearn",
            "scikit", "tensorflow", "pytorch", "ml model", "train",
            "classification", "regression", "clustering",
        }):
            return True
    return False


def _is_recent_llm_only(career_history: list[dict]) -> bool:
    """
    Returns True if ALL ML-related work is post-2022 AND primarily LLM wrappers.
    """
    has_pre_2022_ml = _is_pre_llm_ml_evidence(career_history)
    if has_pre_2022_ml:
        return False

    has_any_ml = False
    for job in career_history:
        desc = (job.get("description") or "").lower()
        title = (job.get("title") or "").lower()
        combined = desc + " " + title
        if text_contains_any(combined, NLP_KEYWORDS | RETRIEVAL_KEYWORDS):
            has_any_ml = True
            break

    if not has_any_ml:
        return False

    # Check if what's there is all LLM wrappers
    total_ml_descs = 0
    llm_only_descs = 0
    for job in career_history:
        desc = (job.get("description") or "").lower()
        title = (job.get("title") or "").lower()
        combined = desc + " " + title
        if text_contains_any(combined, NLP_KEYWORDS | RETRIEVAL_KEYWORDS):
            total_ml_descs += 1
            if text_contains_any(combined, LLM_WRAPPER_KEYWORDS) and not text_contains_any(
                combined, RETRIEVAL_KEYWORDS - {"rag"}
            ):
                llm_only_descs += 1

    if total_ml_descs == 0:
        return False
    return (llm_only_descs / total_ml_descs) >= 0.8


def _career_text_for_position(job: dict) -> str:
    desc = (job.get("description") or "")
    title = (job.get("title") or "")
    return f"{title} {desc}"


# Keywords that indicate genuine AI/ML ranking or retrieval evaluation context
# (must accompany EVAL_KEYWORDS to avoid false-positive experiment language)
_EVAL_AI_CONTEXT_KEYWORDS = frozenset({
    "ranking", "retrieval", "search", "recommendation", "recommender",
    "embedding", "embeddings", "semantic", "information retrieval",
    "ndcg", "mrr", "map", "mean average precision", "mean reciprocal rank",
    "precision@", "recall@", "a/b test", "a/b testing", "ab testing",
    "offline evaluation", "online evaluation", "ranking metric",
    "retrieval metric", "re-ranking", "reranking", "ltr", "learning to rank",
})

# Python engineering quality keywords (engineering craft, not just "python")
_PYTHON_QUALITY_KEYWORDS = frozenset({
    "pytest", "unit test", "unit testing", "type hint", "type hints",
    "pydantic", "asyncio", "async", "multiprocessing", "profiling",
    "optimization", "packaging", "setup.py", "pyproject", "pip package",
    "code review", "refactor", "clean code", "design pattern",
    "mypy", "ruff", "black", "linting", "ci/cd", "continuous integration",
    "docker", "containeriz", "production python",
})


def _extract_career_features(
    career_history: list[dict],
    jd: JDRequirements,
) -> dict:
    """Extract all career-derived features. Returns a dict of feature values."""
    if not career_history:
        return {}

    total_months = 0
    product_months = 0
    consulting_months = 0
    research_months = 0
    tenures: list[int] = []
    all_descriptions: list[str] = []
    production_hits = 0
    retrieval_hits = 0
    nlp_hits = 0
    eval_hits = 0
    python_hits = 0
    python_quality_hits = 0
    cv_speech_hits = 0
    research_hits = 0
    num_roles_with_descriptions = 0
    has_recent_coding = False  # any code evidence in last 18 months

    for job in career_history:
        dur = job.get("duration_months") or 0
        dur = max(0, dur)
        total_months += dur
        tenures.append(dur)

        company = job.get("company") or ""
        industry = job.get("industry") or ""
        desc = job.get("description") or ""
        title = job.get("title") or ""
        all_descriptions.append(f"{title} {desc}")

        start_date = job.get("start_date") or ""
        is_current = job.get("is_current") or False

        is_cons = _is_consulting_company(company)
        is_prod = _is_product_company(company, industry)
        is_acad = _is_academic(company, industry)

        if is_cons:
            consulting_months += dur
        elif is_prod:
            product_months += dur
        elif is_acad:
            research_months += dur

        combined = (title + " " + desc).lower()

        # Production evidence
        prod_count = count_keyword_hits(combined, PRODUCTION_KEYWORDS)
        if prod_count >= 2:
            production_hits += dur

        # Retrieval evidence
        if count_keyword_hits(combined, RETRIEVAL_KEYWORDS) >= 1:
            retrieval_hits += dur

        # NLP evidence
        if count_keyword_hits(combined, NLP_KEYWORDS) >= 1:
            nlp_hits += dur

        # Evaluation evidence — only count when accompanied by genuine AI/ML context.
        # Generic "experimentation" or "a/b test" in non-technical roles must not inflate
        # AI relevance. Require co-occurrence with at least one retrieval/ranking term.
        if (count_keyword_hits(combined, EVAL_KEYWORDS) >= 1
                and count_keyword_hits(combined, _EVAL_AI_CONTEXT_KEYWORDS) >= 1):
            eval_hits += dur

        # Python evidence
        if "python" in combined:
            python_hits += dur

        # Python engineering quality evidence (beyond just mentioning python)
        if "python" in combined and count_keyword_hits(combined, _PYTHON_QUALITY_KEYWORDS) >= 1:
            python_quality_hits += dur

        # CV / speech evidence
        if count_keyword_hits(combined, CV_SPEECH_KEYWORDS) >= 1:
            cv_speech_hits += dur

        # Research evidence
        if count_keyword_hits(combined, RESEARCH_KEYWORDS) >= 2:
            research_hits += dur

        # Recent coding evidence (is_current or started within 18 months)
        start_yr = extract_year_from_date(start_date)
        if is_current or (start_yr and start_yr >= 2024):
            if any(kw in combined for kw in {"python", "code", "coding", "implement",
                                             "build", "built", "ship", "shipped",
                                             "develop", "developed"}):
                has_recent_coding = True

        if desc.strip():
            num_roles_with_descriptions += 1

    full_career_text = " ".join(all_descriptions)
    norm = max(total_months, 1)

    # Career direction: fraction of career pointing toward AI/ML roles
    ai_months = retrieval_hits + nlp_hits
    career_direction = min(1.0, ai_months / norm)

    # Production evidence score (fraction of career with production signals)
    prod_evidence = min(1.0, production_hits / norm)

    # Retrieval career evidence
    retrieval_evidence = min(1.0, retrieval_hits / norm)

    # NLP career evidence
    nlp_evidence = min(1.0, nlp_hits / norm)

    # Eval evidence (already gated on AI/ML context in the loop above)
    eval_evidence = min(1.0, eval_hits / norm)

    # Python evidence
    python_evidence = min(1.0, python_hits / norm)

    # Python engineering quality signal (needs python + quality craft indicators)
    python_quality = min(1.0, python_quality_hits / norm)

    # CV/speech dominance
    cv_speech_dom = min(1.0, cv_speech_hits / norm)

    # Research fraction
    research_fraction = min(1.0, research_months / norm)

    # Product fraction
    product_fraction = min(1.0, product_months / norm)
    consulting_fraction = min(1.0, consulting_months / norm)

    # Consulting only?
    consulting_only = (consulting_fraction >= 0.95) and (product_months == 0)

    # Research only?
    research_only = (research_fraction >= 0.85)

    # Inactive architect?
    inactive_architect = not has_recent_coding and (
        any("architect" in (j.get("title") or "").lower() for j in career_history) or
        any("tech lead" in (j.get("title") or "").lower() for j in career_history) or
        any("vp" in (j.get("title") or "").lower() for j in career_history)
    )

    # Corroborated career narrative scores
    retrieval_kw_count = count_keyword_hits(full_career_text.lower(), RETRIEVAL_KEYWORDS)
    nlp_kw_count = count_keyword_hits(full_career_text.lower(), NLP_KEYWORDS)

    return {
        "total_career_months": total_months,
        "product_company_months": product_months,
        "consulting_company_months": consulting_months,
        "product_fraction": product_fraction,
        "consulting_fraction": consulting_fraction,
        "longest_tenure_months": max(tenures) if tenures else 0,
        "avg_tenure_months": sum(tenures) / len(tenures) if tenures else 0,
        "num_positions": len(career_history),
        "career_direction_score": career_direction,
        "production_deployment_evidence": prod_evidence,
        "retrieval_career_evidence": retrieval_evidence,
        "nlp_career_evidence": nlp_evidence,
        "eval_career_evidence": eval_evidence,
        "python_career_evidence": python_evidence,
        "python_quality_signal": python_quality,
        "research_career_fraction": research_fraction,
        "pre_llm_ml_evidence": _is_pre_llm_ml_evidence(career_history),
        "recent_llm_only": _is_recent_llm_only(career_history),
        "inactive_architect_flag": inactive_architect,
        "consulting_only_flag": consulting_only,
        "research_only_flag": research_only,
        "cv_speech_dominance": cv_speech_dom,
        "career_narrative_retrieval_score": min(1.0, retrieval_kw_count / 8.0),
        "career_narrative_nlp_score": min(1.0, nlp_kw_count / 6.0),
        "all_career_text": full_career_text,
    }


def _extract_skill_features(
    skills: list[dict],
    career_text: str,
) -> dict:
    """Extract all skill-derived features."""
    if not skills:
        return {
            "retrieval_skill_depth": 0.0, "python_skill_signal": 0.0,
            "eval_skill_signal": 0.0, "llm_finetune_skill": 0.0,
            "nlp_ir_skill_depth": 0.0, "cv_speech_skill_fraction": 0.0,
            "skill_keyword_density_ratio": 0.0, "skill_assessment_composite": 0.0,
            "skill_duration_mean": 0.0, "num_skills": 0,
            "jd_core_skill_count": 0, "jd_core_skill_corroborated": 0,
            "preferred_skill_count": 0, "preferred_skill_match_ratio": 0.0,
            "preferred_skill_corroborated": 0,
        }

    career_lower = career_text.lower()
    durations = [s.get("duration_months") or 0 for s in skills]
    skill_duration_mean = sum(durations) / len(durations) if durations else 0

    # JD core skill count and corroboration
    jd_core_count = 0
    jd_core_corroborated = 0
    for sk in skills:
        name = sk.get("name", "").lower()
        if any(kw in name for kw in JD_CORE_SKILLS):
            jd_core_count += 1
            dur = sk.get("duration_months") or 0
            end = sk.get("endorsements") or 0
            if dur >= 6 or end >= 3 or any(kw in career_lower for kw in JD_CORE_SKILLS if kw in name):
                jd_core_corroborated += 1

    # Retrieval skill depth
    retrieval_depth = _compute_skill_depth(
        skills, RETRIEVAL_KEYWORDS, career_text, min_duration_corroborate=6
    )

    # Python signal
    python_depth = 0.0
    for sk in skills:
        if "python" in sk.get("name", "").lower():
            dur = sk.get("duration_months") or 0
            end = sk.get("endorsements") or 0
            prof_map = {"beginner": 0.4, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
            prof = prof_map.get(sk.get("proficiency", "beginner"), 0.4)
            corr = 1.0 if ("python" in career_lower and dur >= 12) else (0.5 if dur >= 6 else 0.2)
            python_depth = max(python_depth, prof * corr)

    # Eval signal
    eval_signal = 0.0
    for sk in skills:
        name = sk.get("name", "").lower()
        if any(kw in name for kw in EVAL_KEYWORDS):
            dur = sk.get("duration_months") or 0
            corr = 1.0 if dur >= 6 else 0.4
            prof_map = {"beginner": 0.3, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
            eval_signal = max(eval_signal, prof_map.get(sk.get("proficiency", "beginner"), 0.3) * corr)

    # LLM fine-tuning signal
    llm_ft_keywords = frozenset({"lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning"})
    llm_ft_signal = 0.0
    for sk in skills:
        name = sk.get("name", "").lower()
        if any(kw in name for kw in llm_ft_keywords):
            dur = sk.get("duration_months") or 0
            corr = 1.0 if dur >= 3 else 0.5
            prof_map = {"beginner": 0.3, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
            llm_ft_signal = max(llm_ft_signal, prof_map.get(sk.get("proficiency", "beginner"), 0.3) * corr)

    # NLP/IR depth
    nlp_ir_depth = _compute_skill_depth(
        skills, NLP_KEYWORDS | RETRIEVAL_KEYWORDS, career_text
    )

    # CV/speech fraction of skill space
    cv_speech_skill_count = sum(
        1 for sk in skills
        if any(kw in sk.get("name", "").lower() for kw in CV_SPEECH_KEYWORDS)
    )
    cv_speech_fraction = min(1.0, cv_speech_skill_count / max(len(skills), 1))

    # Keyword density ratio: AI keywords in skills vs career descriptions
    skills_text = " ".join(s.get("name", "") for s in skills).lower()
    skill_kw_hits = count_keyword_hits(skills_text, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
    career_kw_hits = count_keyword_hits(career_lower, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
    if career_kw_hits == 0 and skill_kw_hits > 3:
        density_ratio = 5.0  # many skill keywords, zero career corroboration
    elif career_kw_hits > 0:
        density_ratio = min(5.0, skill_kw_hits / max(career_kw_hits, 1))
    else:
        density_ratio = 0.0

    # Preferred skill count / match ratio / corroboration
    pref_count = 0
    pref_corroborated = 0
    for sk in skills:
        name = sk.get("name", "").lower()
        if not any(kw in name for kw in JD_PREFERRED_SKILLS):
            continue
        pref_count += 1
        dur = sk.get("duration_months") or 0
        end = sk.get("endorsements") or 0
        if dur >= 6 or end >= 3 or any(kw in career_lower for kw in JD_PREFERRED_SKILLS if kw in name):
            pref_corroborated += 1

    pref_total = len(JD_PREFERRED_SKILLS)
    pref_ratio = pref_count / pref_total if pref_total > 0 else 0.0

    return {
        "retrieval_skill_depth": retrieval_depth,
        "python_skill_signal": python_depth,
        "eval_skill_signal": eval_signal,
        "llm_finetune_skill": llm_ft_signal,
        "nlp_ir_skill_depth": nlp_ir_depth,
        "cv_speech_skill_fraction": cv_speech_fraction,
        "skill_keyword_density_ratio": density_ratio,
        "skill_duration_mean": skill_duration_mean,
        "num_skills": len(skills),
        "jd_core_skill_count": jd_core_count,
        "jd_core_skill_corroborated": jd_core_corroborated,
        "preferred_skill_count": pref_count,
        "preferred_skill_match_ratio": pref_ratio,
        "preferred_skill_corroborated": pref_corroborated,
    }


def _extract_assessment_composite(skill_assessment_scores: dict) -> float:
    """Weighted composite of JD-relevant platform assessment scores (0–1)."""
    if not skill_assessment_scores:
        return 0.0
    relevant_keys = {
        "nlp": 2.0,
        "fine-tuning llms": 2.0,
        "faiss": 1.5,
        "recommendation systems": 1.5,
        "feature engineering": 1.0,
        "prompt engineering": 0.5,
        "langchain": 0.5,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for raw_key, score in skill_assessment_scores.items():
        key = raw_key.lower()
        for rk, w in relevant_keys.items():
            if rk in key:
                weighted_sum += (score / 100.0) * w
                total_weight += w
                break

    if total_weight == 0:
        return 0.0
    return min(1.0, weighted_sum / total_weight)


def _extract_education_features(education: list[dict]) -> dict:
    if not education:
        return {
            "highest_degree_level": 0, "education_field_relevance": 0.3,
            "education_tier_best": 4, "education_ai_ml_focus": False,
        }
    best_degree = 0
    best_field_score = 0.0
    best_tier = 4
    has_ai_ml = False

    for edu in education:
        deg = edu.get("degree") or ""
        fld = edu.get("field_of_study") or ""
        tier_str = edu.get("tier") or "tier_4"

        dl = _degree_level(deg)
        best_degree = max(best_degree, dl)

        fs = _score_education_field(fld)
        best_field_score = max(best_field_score, fs)

        tier_num = int(tier_str.replace("tier_", "") if "tier_" in tier_str else "4")
        best_tier = min(best_tier, tier_num)

        fld_lower = fld.lower()
        if any(x in fld_lower for x in {
            "artificial intelligence", "machine learning", "data science",
            "computer science", "nlp", "information retrieval",
        }):
            has_ai_ml = True

    return {
        "highest_degree_level": best_degree,
        "education_field_relevance": best_field_score,
        "education_tier_best": best_tier,
        "education_ai_ml_focus": has_ai_ml,
    }


def _count_relevant_certifications(certifications: list[dict]) -> int:
    relevant_issuers = frozenset({
        "aws", "google", "microsoft", "databricks", "nvidia",
        "deeplearning.ai", "coursera", "stanford", "mit",
    })
    relevant_topics = frozenset({
        "machine learning", "deep learning", "nlp", "ai", "data science",
        "ml", "cloud", "kubernetes", "mlops",
    })
    count = 0
    for cert in certifications:
        name = (cert.get("name") or "").lower()
        issuer = (cert.get("issuer") or "").lower()
        if any(t in name for t in relevant_topics) or any(i in issuer for i in relevant_issuers):
            count += 1
    return count


def _extract_behavioral_features(signals: dict) -> dict:
    if not signals:
        return {}

    last_active = signals.get("last_active_date")
    days_inactive = _days_since(last_active)

    recruiter_response = float(signals.get("recruiter_response_rate") or 0.0)
    avg_response_time = float(signals.get("avg_response_time_hours") or 999.0)
    interview_rate = float(signals.get("interview_completion_rate") or 0.0)
    offer_rate = float(signals.get("offer_acceptance_rate") or -1.0)
    open_to_work = bool(signals.get("open_to_work_flag") or False)
    notice = int(signals.get("notice_period_days") or 90)
    apps_30d = int(signals.get("applications_submitted_30d") or 0)
    views_30d = int(signals.get("profile_views_received_30d") or 0)
    saved_30d = int(signals.get("saved_by_recruiters_30d") or 0)
    search_30d = int(signals.get("search_appearance_30d") or 0)
    connections = int(signals.get("connection_count") or 0)
    github = float(signals.get("github_activity_score") or -1.0)
    completeness = float(signals.get("profile_completeness_score") or 0.0)

    # Verified contact count (0–3)
    verified_count = (
        int(bool(signals.get("verified_email"))) +
        int(bool(signals.get("verified_phone"))) +
        int(bool(signals.get("linkedin_connected")))
    )

    # Effective availability: 0–1 composite
    activity_score = max(0.0, 1.0 - days_inactive / 365.0)  # decays over a year
    response_score = recruiter_response  # direct 0–1
    open_score = 1.0 if open_to_work else 0.4

    # Notice penalty (0–1, higher = better)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.75
    elif notice <= 90:
        notice_score = 0.5
    elif notice <= 120:
        notice_score = 0.3
    else:
        notice_score = 0.1

    availability = (activity_score * 0.35 + response_score * 0.3 +
                    open_score * 0.2 + notice_score * 0.15)

    # Salary
    sal_range = signals.get("expected_salary_range_inr_lpa") or {}
    sal_min = float(sal_range.get("min") or 0.0)
    sal_max = float(sal_range.get("max") or 0.0)

    # Salary band compatibility (25–65 LPA for Senior AI Eng at Series A)
    sal_compat = 1.0
    if sal_max > 0:
        if sal_min > 70:
            sal_compat = 0.5  # above band
        elif sal_max < 15:
            sal_compat = 0.7  # below band (may indicate weak profile expectation)

    return {
        "days_since_last_active": days_inactive,
        "open_to_work": open_to_work,
        "recruiter_response_rate": recruiter_response,
        "avg_response_time_hours": avg_response_time,
        "interview_completion_rate": interview_rate,
        "offer_acceptance_rate": offer_rate,
        "applications_submitted_30d": apps_30d,
        "profile_views_30d": views_30d,
        "saved_by_recruiters_30d": saved_30d,
        "search_appearance_30d": search_30d,
        "connection_count": connections,
        "github_activity_score": github,
        "notice_period_days": notice,
        "effective_availability_score": availability,
        "verified_contact_count": verified_count,
        "salary_min_lpa": sal_min,
        "salary_max_lpa": sal_max,
        "salary_band_compatible": sal_compat,
        "profile_completeness": completeness,
        "skill_assessment_composite": _extract_assessment_composite(
            signals.get("skill_assessment_scores") or {}
        ),
    }


def _compute_experience_timeline_gap(
    declared_years: float,
    total_career_months: int,
) -> int:
    """
    Difference in months between declared years and career history total.
    Large positive gap = declared experience > career history (possible inflation).
    """
    declared_months = int(declared_years * 12)
    return abs(declared_months - total_career_months)


def _compute_profile_consistency(
    declared_years: float,
    total_career_months: int,
    current_title: str,
    career_history: list[dict],
    skills: list[dict],
    career_text: str,
) -> float:
    """
    Returns a 0–1 consistency score. Deducted for inconsistencies.
    """
    score = 1.0

    # Experience timeline check
    gap = _compute_experience_timeline_gap(declared_years, total_career_months)
    if gap > 36:
        score -= 0.3
    elif gap > 24:
        score -= 0.15
    elif gap > 12:
        score -= 0.05

    # Current title vs most recent career history title
    if career_history:
        most_recent_title = career_history[0].get("title") or ""
        ct_lower = current_title.lower()
        mrt_lower = most_recent_title.lower()
        # Simple overlap check: any word in common
        ct_words = set(ct_lower.split())
        mrt_words = set(mrt_lower.split())
        if not ct_words.intersection(mrt_words):
            score -= 0.15

    # Skills declared vs career evidence
    if skills:
        skills_text = " ".join(s.get("name", "").lower() for s in skills)
        career_lower = career_text.lower()
        ai_skills_count = sum(
            1 for s in skills
            if any(kw in s.get("name", "").lower()
                   for kw in RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
        )
        ai_career_count = count_keyword_hits(career_lower, RETRIEVAL_KEYWORDS | NLP_KEYWORDS)
        if ai_skills_count > 5 and ai_career_count < 2:
            score -= 0.2  # many AI skills, no career evidence

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(candidate: dict, jd: JDRequirements) -> CandidateFeatures:
    """
    Extract the complete feature record for one candidate.
    Pure computation, no model inference.
    """
    feat = CandidateFeatures()
    feat.candidate_id = candidate.get("candidate_id", "")

    profile = candidate.get("profile") or {}
    career_history = candidate.get("career_history") or []
    education = candidate.get("education") or []
    skills = candidate.get("skills") or []
    certifications = candidate.get("certifications") or []
    signals = candidate.get("redrob_signals") or {}

    # Sort career by start_date descending (most recent first)
    career_history = sorted(
        career_history,
        key=lambda j: j.get("start_date") or "0000-00-00",
        reverse=True,
    )

    # --- Profile features ---
    feat.years_of_experience = float(profile.get("years_of_experience") or 0.0)
    if feat.years_of_experience < jd.exp_min_years:
        feat.experience_band = "under_band"
    elif feat.years_of_experience <= jd.exp_target_max:
        feat.experience_band = "in_band"
    else:
        feat.experience_band = "over_band"

    feat.current_title_ai_relevance = _score_title_ai_relevance(
        profile.get("current_title") or ""
    )

    current_company = profile.get("current_company") or ""
    current_industry = profile.get("current_industry") or ""
    feat.current_company_is_consulting = _is_consulting_company(current_company)
    feat.current_company_is_product = _is_product_company(current_company, current_industry)
    feat.current_industry_is_tech = any(
        kw in current_industry.lower()
        for kw in {"software", "technology", "ai", "ml", "data", "internet", "saas"}
    )

    # Location
    location = profile.get("location") or ""
    country = profile.get("country") or ""
    feat.country = country
    feat.is_india_based = "india" in country.lower()
    willing = bool(signals.get("willing_to_relocate") or False)
    feat.willing_to_relocate = willing
    feat.location_score, feat.location_category = _score_location(location, country, willing)

    # --- Career features ---
    career_feats = _extract_career_features(career_history, jd)
    feat.total_career_months = career_feats.get("total_career_months", 0)
    feat.product_company_months = career_feats.get("product_company_months", 0)
    feat.consulting_company_months = career_feats.get("consulting_company_months", 0)
    feat.product_fraction = career_feats.get("product_fraction", 0.0)
    feat.consulting_fraction = career_feats.get("consulting_fraction", 0.0)
    feat.longest_tenure_months = career_feats.get("longest_tenure_months", 0)
    feat.avg_tenure_months = career_feats.get("avg_tenure_months", 0.0)
    feat.num_positions = career_feats.get("num_positions", 0)
    feat.career_direction_score = career_feats.get("career_direction_score", 0.0)
    feat.production_deployment_evidence = career_feats.get("production_deployment_evidence", 0.0)
    feat.retrieval_career_evidence = career_feats.get("retrieval_career_evidence", 0.0)
    feat.nlp_career_evidence = career_feats.get("nlp_career_evidence", 0.0)
    feat.eval_career_evidence = career_feats.get("eval_career_evidence", 0.0)
    feat.python_career_evidence = career_feats.get("python_career_evidence", 0.0)
    feat.python_quality_signal = career_feats.get("python_quality_signal", 0.0)
    feat.research_career_fraction = career_feats.get("research_career_fraction", 0.0)
    feat.pre_llm_ml_evidence = career_feats.get("pre_llm_ml_evidence", False)
    feat.recent_llm_only = career_feats.get("recent_llm_only", False)
    feat.inactive_architect_flag = career_feats.get("inactive_architect_flag", False)
    feat.consulting_only_flag = career_feats.get("consulting_only_flag", False)
    feat.research_only_flag = career_feats.get("research_only_flag", False)
    feat.cv_speech_dominance = career_feats.get("cv_speech_dominance", 0.0)
    feat.career_narrative_retrieval_score = career_feats.get("career_narrative_retrieval_score", 0.0)
    feat.career_narrative_nlp_score = career_feats.get("career_narrative_nlp_score", 0.0)

    career_text = career_feats.get("all_career_text", "")

    # --- Skill features ---
    skill_feats = _extract_skill_features(skills, career_text)
    feat.retrieval_skill_depth = skill_feats.get("retrieval_skill_depth", 0.0)
    feat.python_skill_signal = skill_feats.get("python_skill_signal", 0.0)
    feat.eval_skill_signal = skill_feats.get("eval_skill_signal", 0.0)
    feat.llm_finetune_skill = skill_feats.get("llm_finetune_skill", 0.0)
    feat.nlp_ir_skill_depth = skill_feats.get("nlp_ir_skill_depth", 0.0)
    feat.cv_speech_skill_fraction = skill_feats.get("cv_speech_skill_fraction", 0.0)
    feat.skill_keyword_density_ratio = skill_feats.get("skill_keyword_density_ratio", 0.0)
    feat.skill_assessment_composite = skill_feats.get("skill_assessment_composite", 0.0)
    feat.skill_duration_mean = skill_feats.get("skill_duration_mean", 0.0)
    feat.num_skills = skill_feats.get("num_skills", 0)
    feat.jd_core_skill_count = skill_feats.get("jd_core_skill_count", 0)
    feat.jd_core_skill_corroborated = skill_feats.get("jd_core_skill_corroborated", 0)
    feat.preferred_skill_count = skill_feats.get("preferred_skill_count", 0)
    feat.preferred_skill_match_ratio = skill_feats.get("preferred_skill_match_ratio", 0.0)
    feat.preferred_skill_corroborated = skill_feats.get("preferred_skill_corroborated", 0)

    # --- Education features ---
    edu_feats = _extract_education_features(education)
    feat.highest_degree_level = edu_feats.get("highest_degree_level", 0)
    feat.education_field_relevance = edu_feats.get("education_field_relevance", 0.3)
    feat.education_tier_best = edu_feats.get("education_tier_best", 4)
    feat.education_ai_ml_focus = edu_feats.get("education_ai_ml_focus", False)

    # --- Certifications ---
    feat.relevant_cert_count = _count_relevant_certifications(certifications)

    # --- Behavioral features ---
    beh_feats = _extract_behavioral_features(signals)
    feat.days_since_last_active = beh_feats.get("days_since_last_active", 999)
    feat.open_to_work = beh_feats.get("open_to_work", False)
    feat.recruiter_response_rate = beh_feats.get("recruiter_response_rate", 0.0)
    feat.avg_response_time_hours = beh_feats.get("avg_response_time_hours", 999.0)
    feat.interview_completion_rate = beh_feats.get("interview_completion_rate", 0.0)
    feat.offer_acceptance_rate = beh_feats.get("offer_acceptance_rate", -1.0)
    feat.applications_submitted_30d = beh_feats.get("applications_submitted_30d", 0)
    feat.profile_views_30d = beh_feats.get("profile_views_30d", 0)
    feat.saved_by_recruiters_30d = beh_feats.get("saved_by_recruiters_30d", 0)
    feat.search_appearance_30d = beh_feats.get("search_appearance_30d", 0)
    feat.connection_count = beh_feats.get("connection_count", 0)
    feat.github_activity_score = beh_feats.get("github_activity_score", -1.0)
    feat.notice_period_days = beh_feats.get("notice_period_days", 90)
    feat.effective_availability_score = beh_feats.get("effective_availability_score", 0.0)
    feat.verified_contact_count = beh_feats.get("verified_contact_count", 0)
    feat.salary_min_lpa = beh_feats.get("salary_min_lpa", 0.0)
    feat.salary_max_lpa = beh_feats.get("salary_max_lpa", 0.0)
    feat.salary_band_compatible = beh_feats.get("salary_band_compatible", 1.0)
    feat.profile_completeness = beh_feats.get("profile_completeness", 0.0)
    feat.skill_assessment_composite = beh_feats.get("skill_assessment_composite", 0.0)

    # --- Trust / consistency features ---
    feat.experience_timeline_gap_months = _compute_experience_timeline_gap(
        feat.years_of_experience, feat.total_career_months
    )
    feat.profile_consistency_score = _compute_profile_consistency(
        feat.years_of_experience,
        feat.total_career_months,
        profile.get("current_title") or "",
        career_history,
        skills,
        career_text,
    )

    # Non-India, non-relocatable hard filter
    if not feat.is_india_based and not feat.willing_to_relocate:
        feat.location_category = "international_no_relocate"

    return feat
