"""Sentiment analysis modules."""

import logging
from typing import Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


def compound_to_stars(compound: float) -> float:
    """Convert VADER compound score to 1-5 star rating."""
    stars = 3.0 + 2.0 * float(compound)
    return max(1.0, min(5.0, stars))


class VADERSentimentAnalyzer:
    """VADER sentiment analyzer."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        stars = compound_to_stars(compound)
        
        # Determine label
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "compound": compound,
            "label": label,
            "stars": stars
        }


