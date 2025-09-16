"""Data models for InsightHub."""

from dataclasses import dataclass
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
    """Reference to an entity mentioned in comments."""
    name: str
    entity_type: str
    confidence: float


@dataclass
class GPTCommentAnno:
    """GPT annotation for a single comment."""
    comment_id: str
    overall_score: float
    aspect_scores: Dict[str, float]  # aspect_name -> score
    entities: List[EntityRef]
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

