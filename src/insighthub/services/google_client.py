"""Google Places/Reviews data collection service for InsightHub."""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests

from ..core.config import settings

logger = logging.getLogger(__name__)

class GoogleService:
    """Google Places data collection service using Google Places API."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'google_places_api_key', '')
        self.base_url = "https://maps.googleapis.com/maps/api/place"
    
    def search_places(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for places using Google Places API."""
        if not self.api_key:
            return self._get_mock_places(query, limit)
        
        try:
            # Text search for places
            params = {
                "query": query,
                "key": self.api_key,
                "fields": "place_id,name,rating,user_ratings_total,formatted_address"
            }
            
            response = requests.get(f"{self.base_url}/textsearch/json", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                places = data.get("results", [])
                logger.info(f"Found {len(places)} places for query: {query}")
                return places[:limit]
            else:
                logger.error(f"Google Places search failed: {response.status_code}")
                return self._get_mock_places(query, limit)
                
        except Exception as e:
            logger.error(f"Google Places search error: {e}")
            return self._get_mock_places(query, limit)
    
    def get_place_reviews(self, place_id: str) -> List[Dict[str, Any]]:
        """Get reviews for a specific place (max 5 reviews per place)."""
        if not self.api_key:
            return self._get_mock_reviews(place_id)
        
        try:
            params = {
                "place_id": place_id,
                "key": self.api_key,
                "fields": "reviews,rating,user_ratings_total"
            }
            
            response = requests.get(f"{self.base_url}/details/json", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                reviews = result.get("reviews", [])
                logger.info(f"Retrieved {len(reviews)} reviews for place {place_id}")
                return reviews
            else:
                logger.error(f"Google Places reviews fetch failed: {response.status_code}")
                return self._get_mock_reviews(place_id)
                
        except Exception as e:
            logger.error(f"Google Places reviews error: {e}")
            return self._get_mock_reviews(place_id)
    
    def scrape(self, query: str, limit: int = 50) -> List[dict]:
        """Scrape Google for reviews related to the query."""
        logger.info(f"Scraping Google for '{query}' with limit {limit}...")
        
        # Search for places
        places = self.search_places(query, limit=20)
        
        all_reviews = []
        for place in places[:10]:  # Limit to top 10 places
            place_reviews = self.get_place_reviews(place["place_id"])
            
            # Add place context to reviews
            for review in place_reviews:
                review["place_name"] = place.get("name", "")
                review["place_id"] = place["place_id"]
                review["place_rating"] = place.get("rating", 0)
                review["place_address"] = place.get("formatted_address", "")
            
            all_reviews.extend(place_reviews)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Convert to dict format matching Reddit client
        reviews = []
        for review_data in all_reviews[:limit]:
            try:
                # Parse timestamp to UTC (Google uses seconds since epoch)
                created_utc = review_data.get("time", int(datetime.now().timestamp()))
                if isinstance(created_utc, str):
                    created_utc = int(datetime.now().timestamp())
                
                review_dict = {
                    "id": f"google_{review_data.get('author_name', 'unknown')}_{created_utc}",
                    "source": "google",
                    "text": review_data.get("text", ""),
                    "created_utc": created_utc,
                    "permalink": f"https://www.google.com/maps/place/?q=place_id:{review_data.get('place_id', '')}",
                    "url": f"https://www.google.com/maps/place/?q=place_id:{review_data.get('place_id', '')}",
                    "author": review_data.get("author_name", "Unknown"),
                    "upvotes": review_data.get("rating", 0) * 2,  # Convert 1-5 rating to upvotes
                    "meta": {
                        "place_name": review_data.get("place_name", ""),
                        "place_rating": review_data.get("place_rating", 0),
                        "place_address": review_data.get("place_address", ""),
                        "review_rating": review_data.get("rating", 0),
                        "relative_time": review_data.get("relative_time_description", "")
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                logger.warning(f"Failed to create review dict from Google review: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} Google reviews for '{query}'")
        return reviews
    
    def _get_mock_places(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock place data for testing."""
        mock_places = [
            {
                "place_id": f"mock_place_{i}",
                "name": f"{query} Location {i+1}",
                "rating": 4.0 + (i % 3) * 0.5,
                "user_ratings_total": (i + 1) * 100,
                "formatted_address": f"{100 + i} Main St, City, State"
            }
            for i in range(min(limit, 5))
        ]
        return mock_places
    
    def _get_mock_reviews(self, place_id: str) -> List[Dict[str, Any]]:
        """Generate mock review data for testing."""
        mock_reviews = [
            {
                "text": f"This is a mock review for place {place_id}. Great experience!",
                "rating": 4 + (i % 2),
                "time": int((datetime.now() - timedelta(days=i*30)).timestamp()),
                "author_name": f"Reviewer{i+1}",
                "relative_time_description": f"{i+1} months ago"
            }
            for i in range(5)  # Google API returns max 5 reviews
        ]
        return mock_reviews
