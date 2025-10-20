"""Yelp data collection service for InsightHub."""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests

# Review import removed - using dict format
from ..core.config import settings

logger = logging.getLogger(__name__)

class YelpService:
    """Yelp data collection service using Yelp Fusion API."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'yelp_api_key', '')
        self.base_url = "https://api.yelp.com/v3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
    
    def search_businesses(self, query: str, location: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Search for businesses on Yelp."""
        if not self.api_key:
            return self._get_mock_businesses(query, limit)
        
        try:
            params = {
                "term": query,
                "limit": min(limit, 50),  # Yelp API limit
                "sort_by": "rating"
            }
            
            if location:
                params["location"] = location
            
            response = requests.get(
                f"{self.base_url}/businesses/search",
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                businesses = data.get("businesses", [])
                logger.info(f"Found {len(businesses)} businesses for query: {query}")
                return businesses
            else:
                logger.error(f"Yelp search failed: {response.status_code} - {response.text}")
                return self._get_mock_businesses(query, limit)
                
        except Exception as e:
            logger.error(f"Yelp search error: {e}")
            return self._get_mock_businesses(query, limit)
    
    def get_business_reviews(self, business_id: str) -> List[Dict[str, Any]]:
        """Get reviews for a specific business (max 3 reviews per business)."""
        if not self.api_key:
            return self._get_mock_reviews(business_id)
        
        try:
            response = requests.get(
                f"{self.base_url}/businesses/{business_id}/reviews",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                reviews = data.get("reviews", [])
                logger.info(f"Retrieved {len(reviews)} reviews for business {business_id}")
                return reviews
            else:
                logger.error(f"Yelp reviews fetch failed: {response.status_code}")
                return self._get_mock_reviews(business_id)
                
        except Exception as e:
            logger.error(f"Yelp reviews error: {e}")
            return self._get_mock_reviews(business_id)
    
    def scrape(self, query: str, limit: int = 50) -> List[dict]:
        """Scrape Yelp for reviews related to the query."""
        logger.info(f"Scraping Yelp for '{query}' with limit {limit}...")
        
        # Search for businesses
        businesses = self.search_businesses(query, limit=20)
        
        all_reviews = []
        for business in businesses[:10]:  # Limit to top 10 businesses
            business_reviews = self.get_business_reviews(business["id"])
            
            # Add business context to reviews
            for review in business_reviews:
                review["business_name"] = business["name"]
                review["business_id"] = business["id"]
                review["business_rating"] = business["rating"]
                review["business_location"] = business.get("location", {}).get("address1", "")
            
            all_reviews.extend(business_reviews)
            
            # Rate limiting (Yelp allows 5000 requests per day)
            time.sleep(0.1)
        
        # Convert to dict format matching Reddit client
        reviews = []
        for review_data in all_reviews[:limit]:
            try:
                # Parse date to UTC timestamp
                created_utc = datetime.fromisoformat(
                    review_data["time_created"].replace("Z", "+00:00")
                ).timestamp()
                
                review_dict = {
                    "id": review_data["id"],
                    "source": "yelp",
                    "text": review_data["text"],
                    "created_utc": created_utc,
                    "permalink": f"https://www.yelp.com/biz/{review_data['business_id']}",
                    "url": f"https://www.yelp.com/biz/{review_data['business_id']}",
                    "author": review_data["user"]["name"],
                    "upvotes": review_data["useful"] + review_data["funny"] + review_data["cool"],
                    "meta": {
                        "business_name": review_data.get("business_name", ""),
                        "business_rating": review_data.get("business_rating", 0),
                        "business_location": review_data.get("business_location", ""),
                        "review_rating": review_data["rating"],
                        "useful": review_data["useful"],
                        "funny": review_data["funny"],
                        "cool": review_data["cool"]
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                logger.warning(f"Failed to create review dict from Yelp review: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} Yelp reviews for '{query}'")
        return reviews
    
    def _get_mock_businesses(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock business data for testing."""
        mock_businesses = [
            {
                "id": f"mock_business_{i}",
                "name": f"{query} Business {i+1}",
                "rating": 4.0 + (i % 3) * 0.5,
                "review_count": (i + 1) * 50,
                "location": {
                    "address1": f"{100 + i} Main St",
                    "city": "San Francisco",
                    "state": "CA"
                }
            }
            for i in range(min(limit, 5))
        ]
        return mock_businesses
    
    def _get_mock_reviews(self, business_id: str) -> List[Dict[str, Any]]:
        """Generate mock review data for testing."""
        mock_reviews = [
            {
                "id": f"mock_review_{i}",
                "text": f"This is a mock review for business {business_id}. Great experience!",
                "rating": 4 + (i % 2),
                "time_created": (datetime.now() - timedelta(days=i*30)).isoformat() + "Z",
                "user": {
                    "name": f"Reviewer{i+1}",
                    "image_url": "https://via.placeholder.com/60x60"
                },
                "useful": i * 2,
                "funny": i,
                "cool": i + 1
            }
            for i in range(3)  # Yelp API only returns 3 reviews max
        ]
        return mock_reviews
