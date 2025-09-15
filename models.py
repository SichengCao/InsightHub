"""Data models for InsightHub."""

from dataclasses import dataclass
from typing import Optional, List, Dict


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

