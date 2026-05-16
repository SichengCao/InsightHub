"""Cross-platform review collection and aggregation manager."""

import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import yaml
import os
from diskcache import Cache

from ..core.config import settings
from ..core.constants import FileConstants
from ..core.cross_platform_models import Platform, QueryIntent, WeightedResult
from .reddit_client import RedditService
from .youtube_client import YouTubeService
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
            # Future platforms (keep for extensibility)
            # Platform.GOOGLE: GoogleService(),
            # Platform.XIAOHONGSHU: XiaohongshuService(),
        }
        self.platform_weights = self._load_platform_weights()
        self._scrape_cache = Cache(FileConstants.CACHE_DIR)
    
    def _load_platform_weights(self) -> Dict[str, Any]:
        """Load platform weighting configuration."""
        try:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "../../../config/platform_weights.yaml"),
                os.path.join(os.path.dirname(__file__), "../../config/platform_weights.yaml"),
                "config/platform_weights.yaml"
            ]
            
            for weights_file in possible_paths:
                if os.path.exists(weights_file):
                    with open(weights_file, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)
            
            raise FileNotFoundError("platform_weights.yaml not found in any expected location")
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
            platforms = ["reddit", "youtube"]
            domain_priors = {p: 1.0/len(platforms) for p in platforms}
        
        # Apply intent-based boosts
        intent_boosts = self.platform_weights.get("intent_boosts", {}).get(intent.value, {})
        if not intent_boosts:
            intent_boosts = {p: 1.0 for p in domain_priors.keys()}
        
        # Calculate final weights
        weights = {}
        total_weight = 0.0
        
        for platform in ["reddit", "youtube"]:
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
        """Scrape all enabled platforms in parallel, with 1-hour result cache."""

        if enabled_platforms is None:
            enabled_platforms = list(Platform)

        results = {}
        uncached = []

        # Per-platform cache: each platform's results are cached independently.
        # This lets the streamlit layer start annotating one platform while another
        # is still scraping, and avoids re-fetching a platform whose cache is warm.
        for platform in enabled_platforms:
            key = ("scrape_platform_v1", platform.value, query.lower().strip(), limit_per_platform)
            hit = self._scrape_cache.get(key)
            if hit is not None:
                results[platform.value] = hit
                logger.info(f"✅ Cache hit {platform.value}: {len(hit)} reviews")
            else:
                uncached.append(platform)

        if not uncached:
            return results

        # Scrape only the platforms that weren't cached.
        with ThreadPoolExecutor(max_workers=len(uncached)) as executor:
            future_to_platform = {
                executor.submit(self.platforms[p].scrape, query, limit_per_platform): p
                for p in uncached
                if self.platforms.get(p)
            }

            for future in as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    reviews = future.result(timeout=300)
                    results[platform.value] = reviews
                    key = ("scrape_platform_v1", platform.value, query.lower().strip(), limit_per_platform)
                    self._scrape_cache.set(key, reviews, expire=3600)
                    logger.info(f"✅ {platform.value}: {len(reviews)} reviews")
                except Exception as e:
                    logger.error(f"❌ {platform.value}: {e}")
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
        
        # Calculate confidence based on data availability and consistency
        confidence = min(1.0, len(all_reviews) / 100.0)  # Simple confidence metric
        
        # Create platform counts for reference (no ratings since LLM handles analysis)
        platform_ratings = {platform: 0.0 for platform in platform_results.keys()}
        
        return WeightedResult(
            overall_rating=0.0,  # LLM will provide actual rating
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
