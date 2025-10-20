"""Cross-platform review collection and aggregation manager."""

import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import yaml
import os

from ..core.config import settings
from ..core.cross_platform_models import Platform, QueryIntent, WeightedResult
from .reddit_client import RedditService
from .youtube_client import YouTubeService
from .yelp_client import YelpService
# Future platforms (keep for extensibility)
# from .google_client import GoogleService
# from .xiaohongshu_client import XiaohongshuService

logger = logging.getLogger(__name__)

class CrossPlatformManager:
    """Manages multi-platform review collection and aggregation."""
    
    def __init__(self):
        self.platforms = {
            Platform.REDDIT: RedditService(),
            Platform.YOUTUBE: YouTubeService(),
            Platform.YELP: YelpService(),
            # Future platforms (keep for extensibility)
            # Platform.GOOGLE: GoogleService(),
            # Platform.XIAOHONGSHU: XiaohongshuService(),
        }
        self.platform_weights = self._load_platform_weights()
    
    def _load_platform_weights(self) -> Dict[str, Any]:
        """Load platform weighting configuration."""
        try:
            weights_file = os.path.join(os.path.dirname(__file__), "../../config/platform_weights.yaml")
            with open(weights_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load platform weights: {e}. Using defaults.")
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, Any]:
        """Get default platform weights when config file is not available."""
        return {
            "domain_priors": {
                "consumer_electronics": {
                    "reddit": 0.40,
                    "youtube": 0.40,
                    "yelp": 0.20
                },
                "local_business": {
                    "reddit": 0.20,
                    "youtube": 0.10,
                    "yelp": 0.70
                }
            },
            "intent_boosts": {
                "RANKING": {
                    "reddit": 1.15,
                    "youtube": 1.10,
                    "yelp": 1.00
                },
                "SOLUTION": {
                    "reddit": 1.25,
                    "youtube": 1.20,
                    "yelp": 0.90
                },
                "GENERIC": {
                    "reddit": 1.00,
                    "youtube": 1.00,
                    "yelp": 1.00
                }
            }
        }
    
    def detect_domain(self, query: str) -> str:
        """Detect the domain/category of the query."""
        query_lower = query.lower()
        
        # Local business indicators
        business_keywords = ["restaurant", "hotel", "cafe", "bar", "shop", "store", "service"]
        if any(keyword in query_lower for keyword in business_keywords):
            return "local_business"
        
        # Consumer electronics indicators  
        tech_keywords = ["phone", "phone", "laptop", "tablet", "camera", "headphone", "speaker"]
        if any(keyword in query_lower for keyword in tech_keywords):
            return "consumer_electronics"
        
        # Beauty/fashion indicators
        beauty_keywords = ["makeup", "skincare", "fashion", "clothing", "beauty"]
        if any(keyword in query_lower for keyword in beauty_keywords):
            return "beauty_fashion"
        
        # Default to consumer electronics for tech products
        return "consumer_electronics"
    
    def calculate_platform_weights(self, query: str, intent: QueryIntent, domain: str, 
                                 platform_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate dynamic platform weights based on query context and data quality."""
        
        # Start with domain-based priors
        domain_priors = self.platform_weights.get("domain_priors", {}).get(domain, {})
        if not domain_priors:
            # Fallback to equal weights
            platforms = ["reddit", "youtube", "yelp", "google", "xiaohongshu"]
            domain_priors = {p: 1.0/len(platforms) for p in platforms}
        
        # Apply intent-based boosts
        intent_boosts = self.platform_weights.get("intent_boosts", {}).get(intent.value, {})
        if not intent_boosts:
            intent_boosts = {p: 1.0 for p in domain_priors.keys()}
        
        # Calculate final weights
        weights = {}
        total_weight = 0.0
        
        for platform in ["reddit", "youtube", "yelp"]:
            # Base weight from domain prior
            base_weight = domain_priors.get(platform, 0.0)
            
            # Apply intent boost
            boost = intent_boosts.get(platform, 1.0)
            
            # Adjust for data availability (more data = higher weight)
            count = platform_counts.get(platform, 0)
            availability_factor = min(1.0, count / 20.0) if count > 0 else 0.1
            
            final_weight = base_weight * boost * availability_factor
            weights[platform] = final_weight
            total_weight += final_weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def scrape_all_platforms(self, query: str, limit_per_platform: int = 50, 
                           enabled_platforms: Optional[List[Platform]] = None) -> Dict[str, List[dict]]:
        """Scrape all enabled platforms in parallel."""
        
        if enabled_platforms is None:
            enabled_platforms = list(Platform)
        
        results = {}
        platform_limits = {}
        
        # Set platform-specific limits based on API constraints
        for platform in enabled_platforms:
            if platform == Platform.YELP:
                platform_limits[platform] = min(limit_per_platform, 30)  # Yelp has 3 reviews/business limit
            # Future platforms would go here
            # elif platform == Platform.XIAOHONGSHU:
            #     platform_limits[platform] = min(limit_per_platform, 100)  # Apify limits
            else:
                platform_limits[platform] = limit_per_platform
        
        # Execute scraping in parallel
        with ThreadPoolExecutor(max_workers=len(enabled_platforms)) as executor:
            future_to_platform = {}
            
            for platform in enabled_platforms:
                platform_service = self.platforms.get(platform)
                if platform_service:
                    future = executor.submit(
                        platform_service.scrape, 
                        query, 
                        platform_limits[platform]
                    )
                    future_to_platform[future] = platform
                else:
                    logger.warning(f"No service available for platform: {platform}")
            
            # Collect results
            for future in as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    reviews = future.result(timeout=300)  # 5-minute timeout per platform
                    results[platform.value] = reviews
                    logger.info(f"✅ {platform.value}: collected {len(reviews)} reviews")
                except Exception as e:
                    logger.error(f"❌ {platform.value}: failed with error: {e}")
                    results[platform.value] = []
        
        return results
    
    def aggregate_results(self, platform_results: Dict[str, List[dict]], 
                         query: str, intent: QueryIntent) -> WeightedResult:
        """Aggregate results from multiple platforms with dynamic weighting."""
        
        # Count reviews per platform
        platform_counts = {platform: len(reviews) for platform, reviews in platform_results.items()}
        
        # Detect domain
        domain = self.detect_domain(query)
        
        # Calculate platform weights
        platform_weights = self.calculate_platform_weights(query, intent, domain, platform_counts)
        
        # Aggregate all reviews
        all_reviews = []
        for platform, reviews in platform_results.items():
            for review in reviews:
                review["platform"] = platform  # Ensure platform is set
                all_reviews.append(review)
        
        # Calculate platform-specific ratings (placeholder - would use sentiment analysis)
        platform_ratings = {}
        for platform, reviews in platform_results.items():
            if reviews:
                # Simple heuristic: average upvotes as proxy for rating
                avg_upvotes = sum(r.get("upvotes", 0) for r in reviews) / len(reviews)
                # Convert to 1-5 scale (rough heuristic)
                platform_ratings[platform] = min(5.0, max(1.0, avg_upvotes / 10.0))
            else:
                platform_ratings[platform] = 3.0  # Neutral default
        
        # Calculate weighted overall rating
        weighted_rating = 0.0
        total_weight = 0.0
        
        for platform, rating in platform_ratings.items():
            weight = platform_weights.get(platform, 0.0)
            weighted_rating += rating * weight
            total_weight += weight
        
        overall_rating = weighted_rating / total_weight if total_weight > 0 else 3.0
        
        # Calculate confidence based on data availability and consistency
        confidence = min(1.0, len(all_reviews) / 100.0)  # Simple confidence metric
        
        return WeightedResult(
            overall_rating=overall_rating,
            aspect_scores={},  # Would be populated by LLM analysis
            platform_weights=platform_weights,
            confidence=confidence,
            total_reviews=len(all_reviews),
            platform_ratings=platform_ratings,
            platform_counts=platform_counts,
            query_intent=intent,
            domain=domain,
            analysis_timestamp=datetime.now()
        )
    
    def search_cross_platform(self, query: str, intent: QueryIntent, 
                            limit_per_platform: int = 50, 
                            enabled_platforms: Optional[List[Platform]] = None) -> Dict[str, Any]:
        """Main method for cross-platform search and analysis."""
        
        logger.info(f"🚀 Starting cross-platform search for: '{query}' (intent: {intent.value})")
        start_time = time.time()
        
        # Scrape selected platforms
        platform_results = self.scrape_all_platforms(query, limit_per_platform, enabled_platforms)
        
        # Aggregate results with dynamic weighting
        aggregated = self.aggregate_results(platform_results, query, intent)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Cross-platform search completed in {elapsed_time:.1f}s")
        
        return {
            "query": query,
            "intent": intent.value,
            "platform_results": platform_results,
            "aggregated": aggregated.to_dict(),
            "total_reviews": len(sum(platform_results.values(), [])),
            "execution_time": elapsed_time
        }
