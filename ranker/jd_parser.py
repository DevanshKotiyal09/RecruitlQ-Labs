"""
Module 1: Job Description Parser
Converts JD prose into a structured hiring intent object used by all downstream modules.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Structured JD requirements (derived once, used everywhere)
# ---------------------------------------------------------------------------

@dataclass
class JDRequirements:
    # Experience
    exp_min_years: float = 4.0
    exp_target_min: float = 5.0
    exp_target_max: float = 9.0
    exp_max_years: float = 15.0
    ideal_exp_min: float = 6.0
    ideal_exp_max: float = 8.0

    # Must-have skill families (lowercased canonical names)
    must_have_skills: FrozenSet[str] = field(default_factory=frozenset)
    preferred_skills: FrozenSet[str] = field(default_factory=frozenset)

    # Location
    preferred_locations: FrozenSet[str] = field(default_factory=frozenset)
    acceptable_locations: FrozenSet[str] = field(default_factory=frozenset)

    # Notice period (days)
    notice_ideal_max: int = 30
    notice_acceptable_max: int = 60
    notice_penalty_threshold: int = 90

    # Work mode
    work_mode: str = "hybrid"

    # Salary (INR LPA)
    salary_band_min: float = 25.0
    salary_band_max: float = 65.0

    # Named consulting firms to watch for
    consulting_firms: FrozenSet[str] = field(default_factory=frozenset)

    # Titles that signal strong match
    matching_titles: FrozenSet[str] = field(default_factory=frozenset)

    # Hard-disqualifying conditions (string keys matched in feature extractor)
    hard_disqualifiers: FrozenSet[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for multiprocessing IPC (frozensets -> lists)."""
        return {
            "exp_min_years": self.exp_min_years,
            "exp_target_min": self.exp_target_min,
            "exp_target_max": self.exp_target_max,
            "exp_max_years": self.exp_max_years,
            "ideal_exp_min": self.ideal_exp_min,
            "ideal_exp_max": self.ideal_exp_max,
            "must_have_skills": list(self.must_have_skills),
            "preferred_skills": list(self.preferred_skills),
            "preferred_locations": list(self.preferred_locations),
            "acceptable_locations": list(self.acceptable_locations),
            "notice_ideal_max": self.notice_ideal_max,
            "notice_acceptable_max": self.notice_acceptable_max,
            "notice_penalty_threshold": self.notice_penalty_threshold,
            "work_mode": self.work_mode,
            "salary_band_min": self.salary_band_min,
            "salary_band_max": self.salary_band_max,
            "consulting_firms": list(self.consulting_firms),
            "matching_titles": list(self.matching_titles),
            "hard_disqualifiers": list(self.hard_disqualifiers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JDRequirements":
        """Reconstruct from serialized dict (lists -> frozensets)."""
        return cls(
            exp_min_years=d.get("exp_min_years", 4.0),
            exp_target_min=d.get("exp_target_min", 5.0),
            exp_target_max=d.get("exp_target_max", 9.0),
            exp_max_years=d.get("exp_max_years", 15.0),
            ideal_exp_min=d.get("ideal_exp_min", 6.0),
            ideal_exp_max=d.get("ideal_exp_max", 8.0),
            must_have_skills=frozenset(d.get("must_have_skills", [])),
            preferred_skills=frozenset(d.get("preferred_skills", [])),
            preferred_locations=frozenset(d.get("preferred_locations", [])),
            acceptable_locations=frozenset(d.get("acceptable_locations", [])),
            notice_ideal_max=d.get("notice_ideal_max", 30),
            notice_acceptable_max=d.get("notice_acceptable_max", 60),
            notice_penalty_threshold=d.get("notice_penalty_threshold", 90),
            work_mode=d.get("work_mode", "hybrid"),
            salary_band_min=d.get("salary_band_min", 25.0),
            salary_band_max=d.get("salary_band_max", 65.0),
            consulting_firms=frozenset(d.get("consulting_firms", [])),
            matching_titles=frozenset(d.get("matching_titles", [])),
            hard_disqualifiers=frozenset(d.get("hard_disqualifiers", [])),
        )


def load_jd_requirements() -> JDRequirements:
    """
    Returns the structured hiring intent object for the Senior AI Engineer JD.
    This is derived from careful reading of the JD and is the single source of
    truth for all scoring and penalty logic.
    """
    must_have = frozenset({
        # Embedding / retrieval domain
        "embeddings", "embedding", "sentence-transformers", "sentence_transformers",
        "openai embeddings", "bge", "e5", "faiss", "pinecone", "weaviate", "qdrant",
        "milvus", "opensearch", "elasticsearch", "vector database", "vector search",
        "hybrid search", "dense retrieval", "approximate nearest neighbour",
        "ann", "hnsw", "semantic search", "retrieval", "ranking",
        # Evaluation
        "ndcg", "mrr", "map", "mean average precision", "a/b test", "a/b testing",
        "offline evaluation", "online evaluation", "ranking evaluation",
        "information retrieval", "ir evaluation",
        # Python
        "python",
        # Production ML
        "production ml", "ml deployment", "model serving", "inference",
        "recommendation system", "recommender", "search engine",
    })

    preferred = frozenset({
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
        "llm fine-tuning", "llm finetuning",
        "learning to rank", "learning-to-rank", "ltr", "xgboost ranking",
        "lambdamart", "ranknet",
        "hr tech", "hrtech", "recruiting tech", "talent intelligence",
        "marketplace", "distributed systems", "large-scale inference",
        "open source", "github", "open-source contributions",
        "langchain", "llama index", "llamaindex", "rag",
        "bert", "transformers", "hugging face", "huggingface",
        "nlp", "natural language processing",
        "mlflow", "weights and biases", "wandb",
    })

    preferred_locs = frozenset({
        "pune", "noida",
    })

    acceptable_locs = frozenset({
        "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
        "bengaluru", "bangalore", "new delhi", "ncr",
    })

    consulting_firms = frozenset({
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
        "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    })

    matching_titles = frozenset({
        "machine learning engineer", "ml engineer", "ai engineer",
        "senior ml engineer", "senior ai engineer", "staff ml engineer",
        "principal ml engineer", "research engineer", "applied scientist",
        "nlp engineer", "search engineer", "ranking engineer",
        "relevance engineer", "data scientist", "applied ml engineer",
        "recommendation engineer", "retrieval engineer",
        "mlops engineer", "ml platform engineer",
    })

    hard_disqualifiers = frozenset({
        "pure_research",          # entire career in labs, no production
        "consulting_only",        # 100% consulting firm career
        "recent_llm_only",        # ML experience entirely post-2022, LLM wrappers only
        "honeypot",               # impossible profile
        "non_india_no_relocate",  # outside India, not willing to relocate
    })

    return JDRequirements(
        must_have_skills=must_have,
        preferred_skills=preferred,
        preferred_locations=preferred_locs,
        acceptable_locations=acceptable_locs,
        consulting_firms=consulting_firms,
        matching_titles=matching_titles,
        hard_disqualifiers=hard_disqualifiers,
    )


# ---------------------------------------------------------------------------
# Compiled keyword sets used for fast text scanning
# ---------------------------------------------------------------------------

# Production deployment language markers
PRODUCTION_KEYWORDS = frozenset({
    "production", "deployed", "serving", "live", "users", "latency",
    "throughput", "on-call", "oncall", "rollout", "scale", "scalable",
    "real-time", "realtime", "real time", "millions", "billion",
    "99th percentile", "p99", "sla", "uptime", "monitoring", "alerting",
    "a/b test", "a/b testing", "experiment", "experimentation",
    "shipped", "launched", "released", "end-to-end", "end to end",
})

# Retrieval / ranking / search domain markers
RETRIEVAL_KEYWORDS = frozenset({
    "embedding", "embeddings", "vector", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "bm25", "tfidf",
    "tf-idf", "dense retrieval", "sparse retrieval", "hybrid retrieval",
    "ann", "hnsw", "approximate nearest", "semantic search",
    "ranking", "ranker", "re-ranking", "reranking", "learning to rank",
    "ltr", "ndcg", "mrr", "map", "information retrieval", "search engine",
    "recommendation", "recommender", "retrieval-augmented",
    "retrieval augmented", "rag",
})

# NLP / IR domain markers
NLP_KEYWORDS = frozenset({
    "nlp", "natural language", "text classification", "named entity",
    "ner", "sentiment", "summarization", "summarisation", "translation",
    "bert", "roberta", "gpt", "llm", "large language model",
    "transformers", "attention", "fine-tuning", "lora", "qlora",
    "token", "tokenization", "tokenisation", "language model",
    "text embedding", "sentence embedding",
})

# Computer vision / speech markers (negative signal for this JD)
CV_SPEECH_KEYWORDS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "cnn", "convolutional", "resnet", "vgg", "segmentation",
    "speech recognition", "asr", "speech synthesis", "tts",
    "text-to-speech", "speaker diarization", "audio processing",
    "video understanding", "action recognition", "pose estimation",
    "lidar", "point cloud", "robotics", "robot",
})

# Research-only language markers
RESEARCH_KEYWORDS = frozenset({
    "published", "publication", "paper", "papers", "arxiv", "journal",
    "conference", "proceedings", "ieee", "neurips", "icml", "iclr",
    "acl", "emnlp", "cvpr", "iccv", "eccv", "phd", "postdoc",
    "postdoctoral", "lab", "laboratory", "thesis", "dissertation",
    "academic", "professor", "faculty", "research assistant",
    "research intern", "grant", "citation", "h-index",
})

# LLM wrapper / framework-only markers (red flag if this is ALL the ML experience)
LLM_WRAPPER_KEYWORDS = frozenset({
    "langchain", "llamaindex", "llama index", "openai api",
    "chatgpt api", "gpt-4 api", "claude api", "anthropic api",
    "prompt engineering", "prompt template", "chain of thought",
    "langsmith", "langgraph", "crewai", "autogen",
})

# Evaluation framework markers (positive)
EVAL_KEYWORDS = frozenset({
    "ndcg", "mrr", "map", "mean average precision", "mean reciprocal rank",
    "precision@", "recall@", "f1", "auc", "roc", "offline evaluation",
    "online evaluation", "a/b test", "a/b testing", "ab testing",
    "experiment", "experimentation", "holdout", "cross-validation",
    "ranking metric", "retrieval metric", "benchmark",
})

# Python quality markers
PYTHON_QUALITY_KEYWORDS = frozenset({
    "python", "pytest", "unit test", "code review", "type hints",
    "pydantic", "fastapi", "flask", "async", "asyncio",
    "multiprocessing", "profiling", "optimization", "performance",
    "refactoring", "clean code", "design pattern",
})


def text_contains_any(text: str, keywords: FrozenSet[str]) -> bool:
    """Case-insensitive check whether text contains any keyword from the set."""
    t = text.lower()
    return any(kw in t for kw in keywords)


def count_keyword_hits(text: str, keywords: FrozenSet[str]) -> int:
    """Count how many distinct keywords from the set appear in text."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def extract_year_from_date(date_str: str | None) -> int | None:
    """Extract year integer from a YYYY-MM-DD date string."""
    if not date_str:
        return None
    m = re.match(r"^(\d{4})", date_str)
    return int(m.group(1)) if m else None
