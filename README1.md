# Redrob AI Recruiter — India Runs Challenge

An end-to-end candidate ranking system built for the **Redrob India Runs Data & AI Hackathon**. Given a pool of up to **100,000 candidate profiles** (JSONL format) and a **Senior AI Engineer** job description, the system scores every candidate, selects the top 100, and produces a validated `submission.csv` within **5 minutes on CPU**.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Challenge Context](#2-challenge-context)
3. [Candidate Data Schema](#3-candidate-data-schema)
4. [Project Structure](#4-project-structure)
5. [All Files — Detailed Reference](#5-all-files--detailed-reference)
6. [Ten-Module Pipeline Architecture](#6-ten-module-pipeline-architecture)
7. [Scoring System — Full Breakdown](#7-scoring-system--full-breakdown)
8. [Anti-Trap Detection System](#8-anti-trap-detection-system)
9. [Semantic Encoding](#9-semantic-encoding)
10. [Behavioral Signal Analysis](#10-behavioral-signal-analysis)
11. [Output Format & Validation Rules](#11-output-format--validation-rules)
12. [Quick Start](#12-quick-start)
13. [Architecture Decisions](#13-architecture-decisions)
14. [Tiebreak Chain](#14-tiebreak-chain)
15. [Keyword Sets Used for Scoring](#15-keyword-sets-used-for-scoring)
16. [Hard Disqualifiers](#16-hard-disqualifiers)
17. [Company Taxonomy](#17-company-taxonomy)
18. [Requirements & Environment](#18-requirements--environment)

---

## 1. What This Project Does

The system answers one question: **"Which 100 candidates best fit a Senior AI Engineer role focused on embedding-based retrieval, semantic search, and production ML?"**

It processes every candidate through a ten-stage pipeline:

1. Parse the JD into structured requirements (skills, experience bands, salary, locations, disqualifiers)
2. Pre-compute dense vector embeddings for all candidate narratives offline
3. Stream the full pool at rank time — no loading 100K records into RAM
4. Extract 60+ numeric and categorical features per candidate from career descriptions, skills, education, and behavioral signals
5. Detect honeypots, keyword stuffers, behavioral twins, impossible timelines, and title inflators
6. Score each candidate through a weighted composite formula
7. Select the top 100 deterministically with a four-level tiebreak chain
8. Generate grounded 1–2 sentence reasoning strings — every claim traces to a real profile field
9. Write and validate a `submission.csv` satisfying all challenge format constraints

---

## 2. Challenge Context

- **Competition:** Redrob India Runs Data & AI Hackathon
- **Task:** Rank candidates 1–100 from a pool of ≤100,000 profiles for a **Senior AI Engineer** role
- **Runtime constraint:** `rank.py` must complete in **≤ 5 minutes** on CPU, 16 GB RAM, no network
- **Pre-computation:** `precompute.py` has no time limit (builds embedding index once)
- **Output:** `<participant_id>.csv` — exactly 100 rows, validated against the official spec
- **Reference date for all "days since" calculations:** `2026-06-01`

### The Role Being Hired For

| Attribute | Value |
|---|---|
| Title | Senior AI Engineer |
| Company | Redrob (Series A) |
| Experience band (target) | 5–9 years |
| Experience band (acceptable) | 4–15 years |
| Ideal band | 6–8 years |
| Preferred locations | Pune, Noida |
| Acceptable locations | Hyderabad, Mumbai, Delhi NCR, Gurgaon, Bengaluru, Bangalore |
| Work mode | Hybrid |
| Salary band | ₹25–65 LPA |
| Notice period (ideal) | ≤ 30 days |
| Notice period (acceptable) | ≤ 60 days |
| Notice period (penalised) | > 90 days |

---

## 3. Candidate Data Schema

Every candidate record is a JSON object with these top-level keys (defined in `candidate_schema.json`):

### `profile`
| Field | Type | Description |
|---|---|---|
| `anonymized_name` | string | Name (anonymized) |
| `headline` | string | One-line professional headline |
| `summary` | string | Multi-sentence professional summary |
| `location` | string | City / region |
| `country` | string | Country |
| `years_of_experience` | number | Declared total years (0–50) |
| `current_title` | string | Current job title |
| `current_company` | string | Current employer |
| `current_company_size` | enum | `1-10` … `10001+` |
| `current_industry` | string | Industry of current employer |

### `career_history` (1–10 items)
| Field | Type | Description |
|---|---|---|
| `company` | string | Employer name |
| `title` | string | Role title |
| `start_date` / `end_date` | date | ISO dates; `end_date` null if current |
| `duration_months` | integer | Months in role |
| `is_current` | boolean | Whether currently active |
| `industry` | string | Employer industry |
| `company_size` | enum | Same values as profile |
| `description` | string | Role responsibilities and achievements |

### `education` (0–5 items)
| Field | Type | Description |
|---|---|---|
| `institution` | string | School / university |
| `degree` | string | e.g. `B.Tech`, `M.Sc`, `PhD` |
| `field_of_study` | string | e.g. `Computer Science` |
| `start_year` / `end_year` | integer | Academic years |
| `grade` | string | GPA / percentage / class |
| `tier` | enum | `tier_1` … `tier_4`, `unknown` |

### `skills` (array)
| Field | Type | Description |
|---|---|---|
| `name` | string | Skill name |
| `proficiency` | enum | `beginner`, `intermediate`, `advanced`, `expert` |
| `endorsements` | integer | Number of endorsements |
| `duration_months` | integer | Months the skill has been used |

### `certifications` (optional array)
| Field | Type | Description |
|---|---|---|
| `name` | string | Certificate name |
| `issuer` | string | Issuing organisation |
| `year` | integer | Year awarded |

### `languages` (optional array)
| Field | Type | Description |
|---|---|---|
| `language` | string | Language name |
| `proficiency` | enum | `basic`, `conversational`, `professional`, `native` |

### `redrob_signals` — 23 platform signals
| Field | Type | Description |
|---|---|---|
| `profile_completeness_score` | 0–100 | % of profile filled |
| `signup_date` | date | When candidate joined platform |
| `last_active_date` | date | Last login/activity |
| `open_to_work_flag` | boolean | Actively seeking |
| `profile_views_received_30d` | integer | Recruiter views last 30 days |
| `applications_submitted_30d` | integer | Applications sent last 30 days |
| `recruiter_response_rate` | 0–1 | Fraction of recruiter messages answered |
| `avg_response_time_hours` | float | Average hours to respond |
| `skill_assessment_scores` | dict | skill → score (0–100); assessments taken on platform |
| `connection_count` | integer | Network connections |
| `endorsements_received` | integer | Total endorsements across all skills |
| `notice_period_days` | 0–180 | Current notice period |
| `expected_salary_range_inr_lpa` | `{min, max}` | Expected salary in INR LPA |
| `preferred_work_mode` | enum | `remote`, `hybrid`, `onsite`, `flexible` |
| `willing_to_relocate` | boolean | Willing to move cities |
| `github_activity_score` | -1–100 | GitHub commits/PRs/stars (−1 = no GitHub) |
| `search_appearance_30d` | integer | Times appeared in recruiter searches |
| `saved_by_recruiters_30d` | integer | Times saved by a recruiter |
| `interview_completion_rate` | 0–1 | Interviews attended / scheduled |
| `offer_acceptance_rate` | -1–1 | Offers accepted / total (−1 = no history) |
| `verified_email` | boolean | Email verified |
| `verified_phone` | boolean | Phone verified |
| `linkedin_connected` | boolean | LinkedIn account linked |

---

## 4. Project Structure

```
.
├── ranker/                       # Core ranking engine — pure Python
│   ├── jd_parser.py              # Module 1  — JD → JDRequirements + keyword frozensets
│   ├── ingester.py               # Module 2  — stream JSONL, batch, hash, index
│   ├── feature_extractor.py      # Module 3  — 65+ features from raw candidate dict
│   ├── semantic_encoder.py       # Module 4  — sentence-transformer index build + query
│   ├── anti_trap.py              # Module 5  — honeypot / stuffing / twin detection
│   ├── behavioral_analyser.py    # Module 6  — 23 redrob_signals → composite 0–1
│   ├── scorer.py                 # Module 7  — 10-stage composite score [0, 1]
│   ├── ranker.py                 # Module 8  — top-N selection, deterministic tiebreak
│   ├── reason_generator.py       # Module 9  — fact-grounded reasoning strings
│   ├── output_assembler.py       # Module 10 — write + validate submission.csv
│   └── __init__.py
│
├── Data Set/                     # Provided challenge data (not committed to source control)
│   └── [PUB] India_runs_data_and_ai_challenge/
│       └── India_runs_data_and_ai_challenge/
│           ├── candidates.jsonl          # Full pool (up to 100K)
│           ├── sample_candidates.json    # Small sample for dev/test
│           ├── sample_candidates.jsonl   # Same, JSONL format
│           ├── candidate_schema.json     # JSON Schema for a profile record
│           ├── sample_submission.csv     # Example output format
│           ├── validate_submission.py    # Official validator script
│           ├── submission_metadata_template.yaml
│           ├── job_description.docx
│           ├── redrob_signals_doc.docx
│           ├── README.docx
│           └── submission_spec.docx
│
├── precompute.py                 # CLI — build embedding index (one-time)
├── rank.py                       # CLI — rank full pool → submission.csv (≤ 5 min)
├── test_integration.py           # End-to-end pipeline integration test
└── requirements.txt              # numpy, sentence-transformers
```

---

## 5. All Files — Detailed Reference

### `ranker/jd_parser.py`
- **`JDRequirements`** dataclass — 19 fields covering experience bands, must-have skills, preferred skills, locations, salary, notice period, consulting firms, matching titles, and hard disqualifiers. Fully serialisable via `to_dict()` / `from_dict()` for multiprocessing IPC.
- **`load_jd_requirements()`** — single source of truth for all scoring thresholds; returns a populated `JDRequirements`.
- **8 compiled keyword `frozenset`s** used by every downstream module for fast `in` lookups:
  - `PRODUCTION_KEYWORDS` — deployed, shipped, latency, A/B test, SLA, millions …
  - `RETRIEVAL_KEYWORDS` — embedding, FAISS, Pinecone, HNSW, BM25, NDCG, semantic search …
  - `NLP_KEYWORDS` — BERT, transformers, fine-tuning, LoRA, LLM, tokenization …
  - `CV_SPEECH_KEYWORDS` — YOLO, ResNet, ASR, TTS, lidar, robotics … *(negative signal)*
  - `RESEARCH_KEYWORDS` — arXiv, NeurIPS, CVPR, PhD, lab, citation, h-index …
  - `LLM_WRAPPER_KEYWORDS` — LangChain, LlamaIndex, OpenAI API, prompt engineering …
  - `EVAL_KEYWORDS` — NDCG, MRR, MAP, precision@, AUC, A/B testing, holdout …
  - `PYTHON_QUALITY_KEYWORDS` — pytest, type hints, asyncio, pydantic, fastapi …
- **`text_contains_any()`** / **`count_keyword_hits()`** — case-insensitive substring matchers
- **`extract_year_from_date()`** — pulls year from `YYYY-MM-DD` strings

### `ranker/ingester.py`
- **`stream_candidates()`** — line-by-line JSONL generator; never loads full file into RAM
- **`batch_stream()`** — yields lists of `batch_size` candidates for parallel processing
- **`count_candidates()`** — counts lines without parsing JSON
- **`compute_file_hash()`** — SHA-256 of data file for embedding cache invalidation
- **`load_all_candidate_ids()`** — set of all IDs used in CSV validation
- **`build_candidate_index()`** — loads full pool into dict keyed by `candidate_id` (reason-generation pass only)

### `ranker/feature_extractor.py`
- **`CandidateFeatures`** dataclass — 65+ typed fields:
  - *Profile:* years of experience, experience band, AI title relevance, company type flags, location score/category, India-based flag
  - *Career:* total/product/consulting/research months, product/consulting fractions, tenure stats, career direction score, production deployment evidence, retrieval/NLP/eval/Python evidence, **python_quality_signal** (engineering craft), pre-LLM ML flag, recent-LLM-only flag, inactive architect flag, CV/speech dominance, career narrative scores
  - *Skills:* retrieval/Python/eval/LLM fine-tune/NLP-IR depth, CV/speech fraction, keyword density ratio, assessment composite, JD core skill count + corroboration count, **preferred_skill_count / preferred_skill_match_ratio / preferred_skill_corroborated**
  - *Education:* degree level (0–3), field relevance, institution tier (1–4), AI/ML focus flag
  - *Certifications:* relevant cert count
  - *Behavioral:* days since active, open-to-work, response rate/time, interview completion, offer acceptance, applications/views/saved/search stats, connections, GitHub score
  - *Availability:* notice period days, effective availability composite
  - *Trust:* verified contact count, profile consistency score, experience timeline gap
  - *Salary:* min/max LPA, band compatibility score
  - *Semantic:* JD similarity, retrieval probe similarity, NLP probe similarity *(injected after encoding)*
- **`extract_features(candidate, jd)`** — main entry point; orchestrates all sub-extractors
- Private helpers: `_extract_career_features()`, `_extract_skill_features()`, `_extract_assessment_composite()`, `_extract_education_features()`, `_extract_behavioral_features()`, `_compute_profile_consistency()`, `_is_pre_llm_ml_evidence()`, `_is_recent_llm_only()`

### `ranker/semantic_encoder.py`
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` — 22M parameters, 384-dim, CPU-friendly
- **`build_candidate_narrative()`** — headline + summary + career descriptions (max 2000 chars); intentionally excludes skills to avoid rewarding keyword stuffing
- **`build_embedding_index()`** — encodes all candidates in batches of 512, saves `embeddings.npy` (N × 384 float32), `candidate_id_map.json`, `jd_embeddings.npy` (3 × 384), `data_hash.txt`
- **`build_signal_stats()`** — computes min/max/p10/p50/p90/mean for 14 numeric signal fields; saves `signal_stats.json`
- **`EmbeddingIndex`** — loads pre-built arrays; computes full `N × 3` similarity matrix via `embeddings @ jd_embeddings.T` (one BLAS call); `get_similarities(id)` / `get_all_similarities()`
- **3 JD probe documents:**
  - `JD_GENERAL_PROBE` — general senior AI engineer description
  - `JD_RETRIEVAL_PROBE` — retrieval/vector-DB specific
  - `JD_NLP_PROBE` — NLP / fine-tuning / RAG specific

### `ranker/anti_trap.py`
- **`TrustFlags`** dataclass — list of `(flag_name, severity, reason)` tuples; auto-recomputes `honeypot_risk` and `trust_penalty_multiplier` on every `add_flag()` call
- **Severity levels:** `NONE=0`, `LOW=1`, `MEDIUM=2`, `HIGH=3`, `HONEYPOT=4`
- **`evaluate_trust(candidate, features)`** — runs all 6 per-candidate checks and returns `TrustFlags`
- **`detect_behavioral_twins(candidates_and_features)`** — batch MD5-hash-based detection; flags groups of >3 identical behavioral signal fingerprints as `SEVERITY_MEDIUM`
- Six integrity checks — see [Anti-Trap Detection System](#8-anti-trap-detection-system)

### `ranker/behavioral_analyser.py`
- **`compute_behavioral_composite(feat)`** — returns 0–1 from 4 sub-scores:
  - `_availability_score()` — last active date, open-to-work flag, notice period (35%)
  - `_engagement_score()` — recruiter response rate, avg response time, applications (30%)
  - `_reliability_score()` — interview completion, offer acceptance, completeness, verified contacts (20%)
  - `_market_interest_score()` — saved by recruiters, profile views, search appearances, GitHub (15%)

### `ranker/scorer.py`
- **`compute_composite_score(feat, trust, jd)`** — returns `(composite_score, confidence_score)` both in [0, 1]
- **`HARD_FILTER_CAP = 0.08`** — maximum score any hard-filtered candidate can receive
- 10 internal stages — see [Scoring System](#7-scoring-system--full-breakdown)

### `ranker/ranker.py`
- **`ScoredCandidate`** dataclass — stores `composite_score`, `confidence_score`, `behavioral_composite`, `tech_fit`, `career_quality`, `location_score`, `penalties`, `trust_penalty_mult`
- **`__lt__`** implements a four-level tiebreak for deterministic sort
- **`select_top_n(scored_candidates, n=100)`** — stable key-sort + post-sort monotonicity enforcement
- **`assign_ranks(top_candidates)`** — returns `[(1, sc), (2, sc), …]`

### `ranker/reason_generator.py`
- **`generate_reason(rank, candidate, feat, trust, jd)`** — 1–2 sentence string
- **Rank bands:** `strong` (1–10), `good` (11–30), `medium` (31–60), `weak` (61–100)
- **`_select_strongest_strength()`** — 7-priority chain: production retrieval evidence → retrieval career evidence → title + YoE → semantic similarity → corroborated JD skills → experience band → assessment score
- **`_select_secondary_strength()`** — eval framework, pre-LLM evidence, LLM fine-tuning, location, GitHub, recruiter saves, notice period
- **`_select_primary_concern()`** — consulting-only, research-only, recent-LLM-only, inactive architect, CV/speech dominance, notice period, stale profile, response rate, international location, experience outside band, keyword stuffing, profile inconsistency
- **`generate_reasons_for_top100()`** — batch wrapper for all 100 ranked candidates

### `ranker/output_assembler.py`
- **`write_submission_csv(ranked, reasons, output_path, valid_ids)`** — validates IDs/ranks while building, sorts by rank, enforces score monotonicity, writes UTF-8 CSV with 10-decimal scores
- **`validate_csv_locally(csv_path)`** — replicates the official `validate_submission.py` logic: exact 100-row count, unique IDs, unique ranks 1–100 all present, non-increasing scores, ascending `candidate_id` tiebreak within equal scores

### `precompute.py`
- Parses `--candidates`, `--index_dir`, `--force` arguments
- Computes SHA-256 hash of data file, skips if cache valid
- Runs `build_signal_stats()` then `build_embedding_index()`
- Reports elapsed time; notes typical ~15–30 min for 100K candidates on CPU

### `rank.py`
- Parses `--candidates`, `--index_dir`, `--out`, `--team_id`, `--workers`
- **Step 1:** Load JD + embedding index + pre-compute all N×3 similarities
- **Step 2:** `ProcessPoolExecutor` parallel feature extraction (one process per batch of 1000); serialises `JDRequirements` and `CandidateFeatures` as plain dicts for safe IPC; single-process fallback for environments that prohibit fork
- **Step 3:** Behavioral twin detection
- **Step 4:** Composite scoring loop
- **Step 5:** `select_top_n` + `assign_ranks`; prints top-10 preview
- **Step 6:** Second stream pass to load raw profiles for only the top-100 IDs
- **Step 7:** Reason generation
- **Step 8:** Write CSV (with valid_ids loaded for validation)
- **Step 9:** Local validation pass; exits non-zero on any error
- Prints wall-clock warning if > 300s

### `test_integration.py`
- Loads sample candidates from `sample_candidates.json`
- Runs full ranking pipeline end-to-end: extract → trust → score → behavioral → rank → reason → CSV write → validate
- Asserts: top candidate has at least one AI signal; scores are monotonically non-increasing
- Verifies new feature fields (`preferred_skill_count`, `preferred_skill_corroborated`, `python_quality_signal`) are present and populated

### `requirements.txt`
```
numpy>=1.24.0
sentence-transformers>=2.2.0
```

---

## 6. Ten-Module Pipeline Architecture

```
candidates.jsonl
      │
      ▼
┌─────────────┐     ┌──────────────┐
│  1. JD      │────▶│ 2. Ingester  │  stream one line at a time
│  Parser     │     └──────┬───────┘
│ (JDReqs)    │            │ batches
└─────────────┘            ▼
      │             ┌──────────────┐
      │             │ 3. Feature   │  60+ features per candidate
      └────────────▶│  Extractor  │  (no model inference)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 4. Semantic  │  (pre-computed; inject similarities)
                    │  Encoder    │  all-MiniLM-L6-v2, 384-dim
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 5. Anti-Trap │  6 integrity checks + twin detection
                    │  Detector   │  → TrustFlags
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 6. Behavioral│  23 signals → 4 sub-scores → 0–1
                    │  Analyser   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 7. Composite │  10-stage weighted formula
                    │  Scorer     │  → (score, confidence)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 8. Ranker &  │  top-100, 4-level tiebreak,
                    │  Selector   │  monotonicity enforcement
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 9. Reason    │  1–2 sentences, fact-grounded,
                    │  Generator  │  band-aware tone
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 10. Output   │  write + validate submission.csv
                    │  Assembler  │
                    └─────────────┘
```

---

## 7. Scoring System — Full Breakdown

### Stage 1 — Hard Filter
If any hard-filter condition is true, score is capped at `≤ 0.08`. See [Hard Disqualifiers](#17-hard-disqualifiers).

### Stage 2 — Technical Fit (weight: 42% of raw)

| Sub-component | Weight | What it measures |
|---|---|---|
| `prod_retrieval` | 35% | `production_deployment_evidence×0.45 + retrieval_career_evidence×0.35 + retrieval_skill_depth×0.20` |
| `semantic` | 18% | `max(jd_similarity, retrieval_probe_sim, nlp_probe_sim×0.8)` |
| `python_signal` | 15% | `python_career_evidence×0.6 + python_skill_signal×0.4` |
| `eval_signal` | 12% | `eval_career_evidence×0.6 + eval_skill_signal×0.4` |
| `nlp_ir` | 10% | `nlp_career_evidence×0.5 + nlp_ir_skill_depth×0.3 + career_narrative_nlp_score×0.2` |
| `assessment` | 5% | Platform skill assessment composite |
| `direction` | 5% | `career_direction_score` (fraction of career toward AI/ML) |

**Semantic rescue:** if `jd_semantic_similarity > 0.40` and `prod_retrieval < 0.2`, adds `(sim - 0.40) × 0.5` to tech score.

### Stage 3 — Career Quality (weight: 25% of raw)

| Sub-component | Weight |
|---|---|
| `product_fraction` (product-company career share) | 30% |
| `stability` (longest tenure: ≥36mo=1.0, ≥24=0.75, ≥12=0.5, else=0.2) | 20% |
| `exp_band` (5–9yr=1.0, 4–5/9–12=0.8, 3–4/12–15=0.55, <3=0.3, >15=0.4) | 20% |
| `pre_llm` (has ML production evidence before 2022) | 15% |
| `title_rel` (current title AI relevance) | 15% |

Multiplied by `consulting_penalty`: >80% consulting → 0.3×, >50% → 0.6×, else → 1.0×

### Stage 4 — Education (weight: 6% of raw)
`field_score×0.5 + tier_score×0.25 + degree_score×0.15 + ai_bonus(0.15 if AI/ML field)`

### Stage 5 — Bonuses (weight: 5% of raw)
Evaluation framework evidence (+0.25), LLM fine-tuning (+0.20), GitHub activity score >50 (+0.20) or >20 (+0.10), relevant certs ≥2 (+0.15) or 1 (+0.07), saved by ≥10 recruiters (+0.15) or ≥5 (+0.08), skill assessment >0.7 (+0.15) or >0.4 (+0.07).

### Stage 6 — Penalties (additive deductions, max 0.90)

| Condition | Penalty |
|---|---|
| Consulting-only career | +0.55 |
| Research-only career | +0.65 |
| Recent LLM-only, no pre-LLM ML | +0.40 |
| Inactive architect (no coding 18+ mo) | +0.30 |
| CV/speech dominance >0.6 | +0.30 |
| CV/speech dominance >0.4 | +0.15 |
| CV/speech skill fraction >0.5 | +0.20 |
| Short avg tenure (<12mo, ≥3 jobs) | +0.20 |
| Short avg tenure (<18mo, ≥4 jobs) | +0.10 |
| Keyword stuffing ratio >3.0 | +0.25 |
| Keyword stuffing ratio >2.0 | +0.12 |
| Notice >120 days | +0.12 |
| Notice >90 days | +0.06 |
| Salary incompatibility <0.6 | +0.08 |
| Profile consistency <0.5 | +0.15 |
| Profile consistency <0.7 | +0.07 |

### Stage 7 — Location (weight: 8% of raw)
| Location | Score | Category |
|---|---|---|
| Pune | 1.0 | preferred |
| Noida / Greater Noida | 1.0 | preferred |
| Delhi, NCR, Gurgaon, Hyderabad, Mumbai, Bengaluru | 0.75 | acceptable_india |
| Other India | 0.55 | other_india |
| International + willing to relocate | 0.35 | international_relocatable |
| International + not relocating | 0.10 | international_no_relocate |

### Stage 8 — Trust Multiplier
`trust_penalty_multiplier` from `TrustFlags`: 1.0 (clean) → 0.02 (honeypot). HIGH flags each multiply by 0.4×; MEDIUM by 0.75×; LOW by 0.92×.

### Stage 9 — Behavioral Multiplier
`behavioral_mult = 0.20 + 0.80 × behavioral_composite`

Range [0.20, 1.0]. Even a perfect tech score is scaled to 20% if behavioral is zero — but a candidate can never score higher than their tech component allows.

### Final Formula

```
raw = tech×0.42 + career×0.25 + location×0.08 + education×0.06 + bonus×0.05
raw = raw × (1 - penalty×0.5) - penalty×0.05
raw = max(0, raw) × trust_multiplier
composite = clamp(raw × (0.20 + 0.80 × behavioral), 0, 1)
```

### Stage 10 — Confidence Score
Used for tiebreaking only (not part of composite):
`production_evidence + corroborated_skill_count + assessment + verified_contacts×0.1 + completeness×0.1 + consistency×0.1`

---

## 8. Anti-Trap Detection System

### Check 1 — Timeline Impossibility
- **Large gap:** `|declared_years × 12 − career_history_total_months| > 48` → `SEVERITY_MEDIUM`
- **Future start date:** any role starts after `2026-06-01` → `SEVERITY_HONEYPOT`
- **Role before education:** role >24 months starting before graduation year → `SEVERITY_MEDIUM`
- **Implausible total:** career history > 420 months (35 years) → `SEVERITY_HIGH`

### Check 2 — Skill Impossibility
- ≥3 skills claiming `expert`/`advanced` with 0 duration AND 0 endorsements → `SEVERITY_HONEYPOT`
- ≥2 such skills → `SEVERITY_HIGH`
- ≥3 skills with `duration_months` exceeding total career length → `SEVERITY_HIGH`

### Check 3 — Keyword Stuffing
- ≥6 AI keywords in skills, 0 in career descriptions → `SEVERITY_HIGH`
- ≥4 AI keywords in skills, ≤1 in career descriptions → `SEVERITY_MEDIUM`
- >25 skills with mean duration <8 months → `SEVERITY_LOW`

### Check 4 — Behavioral Contradictions
- Very high engagement (offer >0.8, interview >0.9, response >0.8) + inactive >365 days → `SEVERITY_MEDIUM`
- `open_to_work=True` + 0 applications + inactive >120 days → `SEVERITY_LOW`

### Check 5 — Title Inflation
- ≥2 senior/leadership titles (Principal, VP, Director, CTO…) with no technical substance in descriptions → `SEVERITY_MEDIUM`
- 1 such title → `SEVERITY_LOW`

### Check 6 — Corroboration Absence
- AI/ML current title + career descriptions contain 0 AI keywords + ≥3 non-tech domain markers (marketing, accounting, HR, sales…) → `SEVERITY_HIGH`

### Batch — Behavioral Twin Detection
- MD5 fingerprint of 6 key behavioral values; if >3 candidates share identical fingerprint → each gets `SEVERITY_MEDIUM` `behavioral_twin` flag

---

## 9. Semantic Encoding

The semantic encoder uses **`sentence-transformers/all-MiniLM-L6-v2`** (22M params, 384-dim, Apache-2.0 licence) to compute dense representations of candidate career narratives. This is done **once** in `precompute.py` and the resulting arrays are loaded at rank time.

**Why skills are excluded from the narrative:**
> Any candidate can list "FAISS" or "Pinecone" in 30 seconds. The narrative is built from headline + summary + career descriptions only, so a high semantic similarity score reflects actual work described, not a stuffed skills section.

**Files produced by `precompute.py`:**

| File | Size (100K candidates) | Description |
|---|---|---|
| `index/embeddings.npy` | ~146 MB | float32 array (N × 384) |
| `index/jd_embeddings.npy` | ~4 KB | float32 array (3 × 384) for 3 JD probes |
| `index/candidate_id_map.json` | ~3 MB | ordered list of candidate IDs |
| `index/signal_stats.json` | ~2 KB | percentile stats for 14 numeric fields |
| `index/data_hash.txt` | <1 KB | SHA-256 of candidates.jsonl for cache check |

**At rank time:**
```python
similarities = embeddings @ jd_embeddings.T   # (N, 384) × (384, 3) → (N, 3)
```
One BLAS matrix multiply — completes in < 1 second for 100K candidates.

---

## 10. Behavioral Signal Analysis

The 23 `redrob_signals` fields are aggregated into a single **behavioral composite** (0–1) that acts as a *multiplier* on the technical score.

| Sub-score | Weight | Key inputs |
|---|---|---|
| **Availability** | 35% | `days_since_last_active` (decays to 0 over 365 days), `open_to_work_flag`, `notice_period_days` |
| **Engagement** | 30% | `recruiter_response_rate`, `avg_response_time_hours`, `applications_submitted_30d` |
| **Reliability** | 20% | `interview_completion_rate`, `offer_acceptance_rate` (−1 = neutral 0.65), `profile_completeness_score`, verified contacts |
| **Market Interest** | 15% | `saved_by_recruiters_30d`, `profile_views_received_30d`, `search_appearance_30d`, `github_activity_score` |

**Design principle:** behavioral signals cannot compensate for technical gaps. A highly responsive candidate with no retrieval experience should not outrank a less responsive candidate who shipped production search systems.

---

## 11. Output Format & Validation Rules

### submission.csv format

```
candidate_id,rank,score,reasoning
CAND_0012345,1,0.8734921083,Senior ML Engineer at Redrob — shipped a retrieval/ranking system as...
...
```

### Official Validation Rules (from `validate_submission.py`)

1. File must be `.csv`, UTF-8 encoded
2. Row 1 must be exactly `candidate_id,rank,score,reasoning` — order matters
3. Exactly **100 data rows** (rows 2–101)
4. Each row must have exactly 4 columns
5. `candidate_id` must match `^CAND_[0-9]{7}$`
6. No duplicate `candidate_id` values
7. `rank` must be an integer 1–100
8. No duplicate rank values; all ranks 1–100 must appear
9. `score` must be a float
10. Scores must be **non-increasing** by rank (rank 1 ≥ rank 2 ≥ … ≥ rank 100)
11. For equal scores, `candidate_id` must be **ascending** (tie-break rule)

The `output_assembler.py` `validate_csv_locally()` function mirrors this logic exactly and is run after every write in `rank.py`.

---

## 12. Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
# numpy>=1.24.0  sentence-transformers>=2.2.0
```

### Step 1 — Pre-compute embedding index (one-time, ~15–30 min for 100K on CPU)
```bash
python precompute.py \
  --candidates "Data Set/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --index_dir ./index
```
Use `--force` to rebuild even if the data file has not changed.

### Step 2 — Rank and produce submission CSV (≤ 5 min)
```bash
python rank.py \
  --candidates "Data Set/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --index_dir ./index \
  --out ./submission.csv
```
The CSV is validated locally before the script exits; non-zero exit on any error.

### Step 3 — Run integration tests
```bash
python test_integration.py
```
Exercises the complete ranking pipeline on `sample_candidates.json`.

### Official validator (challenge-provided)
```bash
python "Data Set/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## 13. Architecture Decisions

### Career descriptions beat declared skills
Any candidate can add "FAISS" to their skills section in 30 seconds. Describing deploying a FAISS index to production with P99 latency targets requires genuine experience. Skills corroborated by career text score ~5× higher than unverified self-declarations. The system uses `duration_months` and `endorsements` on skills as secondary corroboration signals.

### Keyword-first, semantics-second
The JD specifies concrete tools (FAISS, Pinecone, NDCG, HNSW, LambdaMART). A candidate who used them will mention them in career descriptions. Semantic similarity from `all-MiniLM-L6-v2` is used as a *rescue signal*: if `jd_semantic_similarity > 0.40` but keyword-based scores are low, a proportional boost is applied. Semantics supplements keyword matching but never overrides career evidence.

### Behavioral signals are multiplicative, not additive
A highly engaged consultant who spent their entire career at TCS doing IT outsourcing should not outrank an engineer who built production retrieval systems but is slightly slower to reply to recruiters. The behavioral composite is a multiplier in [0.20, 1.0] — it adjusts rank *within* technical tiers but cannot create rank crossings across tier boundaries.

### Pre-LLM evidence is a quality signal
The JD explicitly requires experience that predates the LLM boom. Candidates whose only ML experience is post-2022 LLM wrapper work (LangChain, OpenAI API, prompt engineering) are penalised. Candidates with verifiable ML production work from before 2022 get a 0.15 bonus in career quality scoring.

### Anti-trap philosophy
The dataset contains deliberate honeypots. The system identifies them through cross-signal contradiction:
- "Senior AI Engineer" current title + career descriptions mentioning only accounting and sales
- 12 "expert" skills each with 0 months duration and 0 endorsements
- Career history starting in the future
- Behavioral engagement metrics frozen at suspiciously round values across >3 candidates

These get capped at ≤ 0.08 regardless of how attractive their surface-level profile looks.

### Parallelism without complexity
`rank.py` uses `ProcessPoolExecutor` to extract features from 1000-candidate batches in parallel across all CPUs. The only cross-process data is plain Python dicts (`JDRequirements.to_dict()` and `dataclasses.asdict(feat)`) — no shared memory, no locks, no fragile pickle of complex objects.

---

## 14. Tiebreak Chain

When two candidates share identical composite scores (to float64 precision), ranks are resolved in this order:

| Priority | Field | Direction |
|---|---|---|
| 1 | `behavioral_composite` | Descending (higher = better) |
| 2 | `confidence_score` | Descending (more evidence = better) |
| 3 | `candidate_id` | Ascending (CAND_0000001 beats CAND_0000002) |

This chain is implemented in both `ScoredCandidate.__lt__()` and the `sort()` key tuple in `select_top_n()`, ensuring identical results regardless of sort stability assumptions.

---

## 15. Keyword Sets Used for Scoring

### Must-have skill families (from `jd_parser.py`)
Embedding/retrieval: `embeddings`, `faiss`, `pinecone`, `weaviate`, `qdrant`, `milvus`, `opensearch`, `elasticsearch`, `vector database`, `vector search`, `hybrid search`, `dense retrieval`, `ann`, `hnsw`, `semantic search`, `retrieval`, `ranking`

Evaluation: `ndcg`, `mrr`, `map`, `a/b testing`, `offline evaluation`, `ranking evaluation`, `information retrieval`

Core: `python`, `production ml`, `model serving`, `recommendation system`, `search engine`

### Preferred skills
`lora`, `qlora`, `peft`, `fine-tuning`, `learning to rank`, `ltr`, `lambdamart`, `ranknet`, `langchain`, `llamaindex`, `rag`, `bert`, `transformers`, `hugging face`, `nlp`, `mlflow`, `wandb`, `hr tech`, `marketplace`, `distributed systems`

### Career text keyword sets (8 frozensets)
`PRODUCTION_KEYWORDS`, `RETRIEVAL_KEYWORDS`, `NLP_KEYWORDS`, `CV_SPEECH_KEYWORDS` (negative), `RESEARCH_KEYWORDS`, `LLM_WRAPPER_KEYWORDS` (negative if all-LLM), `EVAL_KEYWORDS`, `PYTHON_QUALITY_KEYWORDS`

---

## 16. Hard Disqualifiers

A candidate is immediately capped at ≤ 0.08 if any of these are true:

| Condition | Source |
|---|---|
| `trust.honeypot_risk >= 0.8` | Anti-trap module |
| `future_start_date` flag with honeypot severity | Anti-trap module |
| `expert_zero_duration` flag with severity ≥ 4 | Anti-trap module |
| `ai_title_non_tech_career` flag with severity ≥ 3 | Anti-trap module |
| `location_category == "international_no_relocate"` | Feature extractor |
| `research_only_flag == True` | Career analysis (>85% research career) |
| `current_title_ai_relevance == 0` AND `career_direction_score < 0.02` AND `jd_core_skill_corroborated == 0` AND `production_deployment_evidence < 0.02` | Zero technical signal |

---

## 17. Company Taxonomy

### Consulting firms (career penalty if dominant)
TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, HCLTech, Tech Mahindra, Mphasis, Hexaware, L&T Infotech, LTIMindtree, Mindtree, NIIT Technologies, Zensar, Mastech

### Product company signals (career quality boost)
Google, Meta, Microsoft, Amazon, Apple, Netflix, Uber, Airbnb, LinkedIn, Twitter, Spotify, Stripe, Databricks, OpenAI, Anthropic, Cohere, Hugging Face, Zomato, Swiggy, Flipkart, Ola, Paytm, PhonePe, Razorpay, CRED, Meesho, Dream11, BYJU's, Unacademy, Freshworks, Zoho, BrowserStack, Postman, Redrob + keywords: `startup`, `series a`, `series b`, `saas`, `product company`

### Academic / research institutions
Matched by keywords: `university`, `college`, `institute`, `iit`, `iim`, `research lab`, `research center`, `academy`

---

## 18. Requirements & Environment

### Python packages
```
numpy>=1.24.0
sentence-transformers>=2.2.0
```
All other imports are from the Python standard library: `re`, `csv`, `json`, `hashlib`, `dataclasses`, `datetime`, `pathlib`, `multiprocessing`, `concurrent.futures`, `argparse`, `time`, `sys`, `os`.

### Compute constraints (per challenge spec)
- CPU only (no GPU inference during ranking)
- No network calls during `rank.py`
- ≤ 16 GB RAM
- `rank.py` must complete in ≤ 5 minutes

### Tested Python version
Python 3.11+ (uses `str | None` union type hints, `from __future__ import annotations` for 3.10 compatibility)

### Platform
Developed and tested on Windows 10/11 (PowerShell). The `ProcessPoolExecutor` code includes `multiprocessing.freeze_support()` for Windows compatibility.
