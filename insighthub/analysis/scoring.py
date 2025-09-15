"""Scoring and analysis modules."""

import logging
from typing import List, Dict, Any
from ..models import Review, AspectScore, AnalysisSummary

logger = logging.getLogger(__name__)


def compute_global(reviews_with_stars: List[Dict[str, Any]]) -> float:
    """Compute global average stars via arithmetic mean of per-review stars."""
    if not reviews_with_stars:
        return 3.0
    
    stars_list = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    for review_data in reviews_with_stars:
        stars = review_data.get('stars', 3.0)
        label = review_data.get('label', 'NEUTRAL')
        
        # Ensure stars is valid
        if stars is None or stars < 1.0 or stars > 5.0:
            stars = 3.0
        
        stars_list.append(stars)
        
        # Count sentiment labels
        if label == 'POSITIVE':
            pos_count += 1
        elif label == 'NEGATIVE':
            neg_count += 1
        else:
            neu_count += 1
    
    total_reviews = len(stars_list)
    
    # Calculate arithmetic mean of per-comment stars
    average_rating = sum(stars_list) / total_reviews
    
    # Calculate baseline for guardrail
    baseline = 1.0 + 4.0 * ((pos_count + 0.5 * neu_count) / total_reviews)
    
    # Apply guardrail if needed
    if abs(average_rating - baseline) > 1.0:
        # Blend toward baseline to damp anomalies
        average_rating = 0.6 * average_rating + 0.4 * baseline
    
    # Clamp to [1, 5]
    average_rating = max(1.0, min(5.0, average_rating))
    
    return average_rating


def compute_aspect_scores(reviews_by_aspect: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """Compute per-aspect stars."""
    aspect_scores = {}
    
    for aspect, reviews in reviews_by_aspect.items():
        if not reviews:
            continue
        
        aspect_stars = []
        for review_data in reviews:
            stars = review_data.get('stars', 3.0)
            if stars is not None:
                aspect_stars.append(max(1.0, min(5.0, stars)))
        
        if aspect_stars:
            aspect_scores[aspect] = sum(aspect_stars) / len(aspect_stars)
    
    return aspect_scores


def create_analysis_summary(
    query: str,
    reviews: List[Review],
    sentiment_results: List[Dict[str, Any]],
    aspect_scores: Dict[str, float]
) -> AnalysisSummary:
    """Create analysis summary."""
    total = len(reviews)
    pos = sum(1 for r in sentiment_results if r.get('label') == 'POSITIVE')
    neg = sum(1 for r in sentiment_results if r.get('label') == 'NEGATIVE')
    neu = total - pos - neg
    
    # Compute global average
    reviews_with_stars = [
        {'stars': r.get('stars', 3.0), 'label': r.get('label', 'NEUTRAL')}
        for r in sentiment_results
    ]
    avg_stars = compute_global(reviews_with_stars)
    
    return AnalysisSummary(
        query=query,
        total=total,
        pos=pos,
        neg=neg,
        neu=neu,
        avg_stars=avg_stars,
        aspect_averages=aspect_scores
    )
