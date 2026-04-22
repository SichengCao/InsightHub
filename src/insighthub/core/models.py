"""Data models for InsightHub."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Review:
    """Represents a single review."""
    id: str
    platform: str
    author: Optional[str]
    created_utc: float
    text: str
    url: Optional[str] = None
    meta: Optional[dict] = None


@dataclass
class AspectScore:
    """Represents scores for a specific aspect."""
    name: str
    pos: int
    neg: int
    neu: int
    stars: float


@dataclass
class AnalysisSummary:
    """Summary of analysis results."""
    query: str
    total: int
    pos: int
    neg: int
    neu: int
    avg_stars: float
    aspect_averages: Dict[str, float]


@dataclass
class EntityRef:
    """Reference to an entity mentioned in a comment.

    sentiment_score and aspect_scores are entity-specific — not the comment's
    overall score — so that comments mentioning multiple entities don't
    misattribute one entity's sentiment to another.
    """
    name: str
    entity_type: str
    confidence: float                               # 0.0–1.0 extraction confidence
    sentiment_score: float = 3.0                    # Entity-specific 1–5 sentiment
    aspect_scores: Dict[str, float] = field(default_factory=dict)  # Entity-specific aspect scores
    is_primary: bool = True                         # False = mentioned for comparison/context only
    mention_context: str = ""                       # Short verbatim excerpt of how entity was mentioned


@dataclass
class GPTCommentAnno:
    """GPT annotation for a single comment."""
    comment_id: str
    overall_score: float
    aspect_scores: Dict[str, float]  # comment-level aspect scores (GENERIC/SOLUTION intent)
    entities: List[EntityRef]
    primary_entity: Optional[str] = None  # name of the main sentiment-focus entity
    solution_key: Optional[str] = None


@dataclass
class RankingItem:
    """Item in a ranking with scores and quotes."""
    name: str
    overall_stars: float
    aspect_scores: Dict[str, float]
    mentions: int
    confidence: float
    quotes: List[str]


@dataclass
class IntentSchema:
    """Schema for intent detection and aspect generation."""
    intent: str  # "RANKING", "SOLUTION", or "GENERIC"
    entity_type: Optional[str] = None  # For RANKING queries
    aspects: List[str] = None  # List of aspect names

