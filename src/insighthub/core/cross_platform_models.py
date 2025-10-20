"""Cross-platform review data models for InsightHub."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class Platform(Enum):
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    YELP = "yelp"
    # Future platforms (keep for extensibility)
    # GOOGLE = "google"
    # XIAOHONGSHU = "xiaohongshu"

class QueryIntent(Enum):
    RANKING = "RANKING"
    SOLUTION = "SOLUTION"
    GENERIC = "GENERIC"

@dataclass
class UnifiedReview:
    """Unified review model across all platforms."""
    
    # Core fields
    id: str
    source: Platform
    author: str
    text: str
    url: str
    
    # Rating (normalized to 1-5 scale)
    rating: Optional[float] = None
    
    # Engagement metrics
    likes: int = 0
    comments: int = 0
    shares: int = 0
    
    # Temporal
    date: datetime = None
    
    # Language and location
    language: str = "en"
    location: Optional[str] = None
    
    # Computed fields
    sentiment: Optional[Dict[str, float]] = None
    aspects: Optional[Dict[str, float]] = None
    
    # Quality indicators
    text_length: int = 0
    is_verified: bool = False
    author_reputation: float = 0.0
    
    def __post_init__(self):
        if self.text_length == 0:
            self.text_length = len(self.text) if self.text else 0

@dataclass
class PlatformStats:
    """Statistics for a platform's data quality."""
    
    platform: Platform
    total_reviews: int
    avg_text_length: float
    avg_engagement: float
    freshness_score: float  # 0-1, higher = more recent
    diversity_score: float  # 0-1, higher = less duplicate
    trust_score: float      # 0-1, based on author reputation
    availability_score: float  # 0-1, based on API limits
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "total_reviews": self.total_reviews,
            "avg_text_length": self.avg_text_length,
            "avg_engagement": self.avg_engagement,
            "freshness_score": self.freshness_score,
            "diversity_score": self.diversity_score,
            "trust_score": self.trust_score,
            "availability_score": self.availability_score
        }

@dataclass
class WeightedResult:
    """Result with platform weights and confidence."""
    
    overall_rating: float
    aspect_scores: Dict[str, float]
    platform_weights: Dict[str, float]
    confidence: float
    total_reviews: int
    
    # Breakdown by platform
    platform_ratings: Dict[str, float]
    platform_counts: Dict[str, int]
    
    # Metadata
    query_intent: QueryIntent
    domain: str
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_rating": self.overall_rating,
            "aspect_scores": self.aspect_scores,
            "platform_weights": self.platform_weights,
            "confidence": self.confidence,
            "total_reviews": self.total_reviews,
            "platform_ratings": self.platform_ratings,
            "platform_counts": self.platform_counts,
            "query_intent": self.query_intent.value,
            "domain": self.domain,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }

@dataclass
class CrossPlatformQuery:
    """Query for cross-platform analysis."""
    
    query: str
    intent: QueryIntent
    domain: str
    platforms: List[Platform]
    max_reviews_per_platform: int = 100
    time_filter_days: int = 90
    
    def __post_init__(self):
        if not self.platforms:
            self.platforms = list(Platform)
