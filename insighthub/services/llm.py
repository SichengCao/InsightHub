"""LLM service for OpenAI integration."""

import logging
import json
from typing import Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from ..config import settings

logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """Factory for creating LLM services."""
    
    @staticmethod
    def create():
        """Create appropriate LLM service."""
        if settings.openai_api_key:
            return OpenAIService()
        else:
            return FallbackLLMService()


class OpenAIService:
    """OpenAI-based LLM service."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI service initialized")
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def analyze_comment(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Analyze comment sentiment and aspects."""
        try:
            prompt = f"""
            Analyze the sentiment of this review text and provide a rating from 1-5 stars:
            
            Text: "{text}"
            
            Return JSON with: {{"sentiment": "POSITIVE/NEGATIVE/NEUTRAL", "stars": 1-5, "reasoning": "brief explanation"}}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
                timeout=10
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Convert to our format
            sentiment = result.get("sentiment", "NEUTRAL")
            stars = float(result.get("stars", 3.0))
            
            # Map sentiment to compound score
            if sentiment == "POSITIVE":
                compound = 0.5
            elif sentiment == "NEGATIVE":
                compound = -0.5
            else:
                compound = 0.0
            
            return {
                "compound": compound,
                "label": sentiment,
                "stars": stars
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return {
                "compound": 0.0,
                "label": "NEUTRAL",
                "stars": 3.0
            }
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def generate_pros_cons(self, reviews: List[Any], query: str) -> Dict[str, str]:
        """Generate pros and cons from reviews."""
        try:
            # Collect review texts - filter for meaningful length
            meaningful_reviews = [r for r in reviews if len(r.text) > 100]
            review_texts = [r.text[:300] for r in meaningful_reviews[:10]]  # Use more reviews
            reviews_summary = "\n".join([f"- {text}" for text in review_texts])
            
            prompt = f"""
            Analyze these detailed reviews for "{query}" and provide specific insights:
            
            Reviews:
            {reviews_summary}
            
            Instructions:
            - Extract SPECIFIC positive and negative aspects mentioned in the reviews
            - Write a precise summary that reflects the actual content and sentiment patterns
            - Include specific details, numbers, and concrete examples from the reviews
            - Avoid generic statements - be specific to what users actually said
            
            Return JSON with:
            {{
                "pros": ["5 specific positive aspects with details from reviews"],
                "cons": ["5 specific negative aspects with details from reviews"],
                "summary": "precise paragraph summarizing the actual review content, specific user experiences, and concrete findings. Include specific details and examples from the reviews."
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
                timeout=20
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI pros/cons generation failed: {e}")
            # Analyze actual review content for fallback
            positive_aspects = []
            negative_aspects = []
            
            for review in reviews[:10]:  # Analyze first 10 reviews
                text_lower = review.text.lower()
                if "battery" in text_lower or "charge" in text_lower:
                    if "good" in text_lower or "great" in text_lower:
                        positive_aspects.append("Battery life and charging performance")
                    elif "poor" in text_lower or "bad" in text_lower:
                        negative_aspects.append("Battery life and charging issues")
                
                if "camera" in text_lower or "photo" in text_lower:
                    if "excellent" in text_lower or "amazing" in text_lower:
                        positive_aspects.append("Camera quality and photo capabilities")
                    elif "mediocre" in text_lower or "disappointing" in text_lower:
                        negative_aspects.append("Camera quality concerns")
                
                if "price" in text_lower or "expensive" in text_lower:
                    negative_aspects.append("Pricing and value concerns")
                
                if "design" in text_lower or "build" in text_lower:
                    if "premium" in text_lower or "quality" in text_lower:
                        positive_aspects.append("Design and build quality")
            
            # Remove duplicates and limit to 5 each
            positive_aspects = list(set(positive_aspects))[:5]
            negative_aspects = list(set(negative_aspects))[:5]
            
            # Generate specific summary
            pos_count = sum(1 for r in reviews if r.sentiment_label == 'POSITIVE')
            neg_count = sum(1 for r in reviews if r.sentiment_label == 'NEGATIVE')
            avg_rating = sum(r.stars for r in reviews) / len(reviews) if reviews else 3.0
            
            summary = f"Analysis of {len(reviews)} reviews reveals {pos_count} positive, {neg_count} negative experiences with an average rating of {avg_rating:.1f}/5 stars. "
            if positive_aspects:
                summary += f"Users specifically praised: {', '.join(positive_aspects[:3])}. "
            if negative_aspects:
                summary += f"Main concerns include: {', '.join(negative_aspects[:3])}. "
            summary += f"The {query} shows {'generally positive' if avg_rating > 3.5 else 'mixed' if avg_rating > 2.5 else 'negative'} sentiment overall."
            
            return {
                "pros": positive_aspects if positive_aspects else ["Quality features", "User experience", "Performance", "Design", "Value"],
                "cons": negative_aspects if negative_aspects else ["Pricing concerns", "Service issues", "Reliability questions", "Limited features", "Support problems"],
                "summary": summary
            }


class FallbackLLMService:
    """Fallback LLM service using simple rules."""
    
    def __init__(self):
        logger.info("Using fallback LLM service")
    
    def analyze_comment(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Simple rule-based analysis."""
        text_lower = text.lower()
        
        # Simple keyword-based sentiment
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "disappointing", "poor"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            compound = 0.3
            label = "POSITIVE"
            stars = 4.0
        elif neg_count > pos_count:
            compound = -0.3
            label = "NEGATIVE"
            stars = 2.0
        else:
            compound = 0.0
            label = "NEUTRAL"
            stars = 3.0
        
        return {
            "compound": compound,
            "label": label,
            "stars": stars
        }
    
    def generate_pros_cons(self, reviews: List[Any], query: str) -> Dict[str, str]:
        """Simple pros/cons extraction."""
        return {
            "pros": f"Users generally appreciate the {query} for its features and quality.",
            "cons": f"Some users have concerns about the {query} pricing and availability."
        }
