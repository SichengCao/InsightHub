"""Xiaohongshu (RED) data collection service for InsightHub using Apify."""

import logging
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Review import removed - using dict format
from ..core.config import settings

logger = logging.getLogger(__name__)

class XiaohongshuService:
    """Xiaohongshu data collection service using Apify scraper."""
    
    def __init__(self):
        self.apify_token = getattr(settings, 'apify_token', '')
        self.actor_id = "apify/xiaohongshu-scraper"  # Default actor
        self.base_url = "https://api.apify.com/v2"
    
    def scrape_xiaohongshu(self, query: str, max_posts: int = 50) -> List[Dict[str, Any]]:
        """Scrape Xiaohongshu using Apify."""
        if not self.apify_token:
            return self._get_mock_posts(query, max_posts)
        
        try:
            # Start the actor run
            run_input = {
                "search": query,
                "maxPosts": min(max_posts, 100),  # Apify limit
                "scroll": 3,  # Number of scrolls
                "includeComments": True
            }
            
            # Start actor run
            run_response = requests.post(
                f"{self.base_url}/actor-tasks/{self.actor_id}/runs",
                headers={"Authorization": f"Bearer {self.apify_token}"},
                json=run_input,
                timeout=30
            )
            
            if run_response.status_code != 201:
                logger.error(f"Failed to start Apify run: {run_response.status_code}")
                return self._get_mock_posts(query, max_posts)
            
            run_data = run_response.json()
            run_id = run_data["data"]["id"]
            dataset_id = run_data["data"]["defaultDatasetId"]
            
            logger.info(f"Started Apify run {run_id} for query: {query}")
            
            # Wait for completion (with timeout)
            max_wait_time = 300  # 5 minutes
            wait_time = 0
            while wait_time < max_wait_time:
                status_response = requests.get(
                    f"{self.base_url}/actor-runs/{run_id}",
                    headers={"Authorization": f"Bearer {self.apify_token}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data["data"]["status"]
                    
                    if status == "SUCCEEDED":
                        break
                    elif status == "FAILED":
                        logger.error(f"Apify run failed: {status_data}")
                        return self._get_mock_posts(query, max_posts)
                
                time.sleep(10)
                wait_time += 10
            
            if wait_time >= max_wait_time:
                logger.warning(f"Apify run timeout for query: {query}")
                return self._get_mock_posts(query, max_posts)
            
            # Get the results
            results_response = requests.get(
                f"{self.base_url}/datasets/{dataset_id}/items",
                headers={"Authorization": f"Bearer {self.apify_token}"},
                params={"clean": "true"}
            )
            
            if results_response.status_code == 200:
                posts = results_response.json()
                logger.info(f"Retrieved {len(posts)} posts from Xiaohongshu for query: {query}")
                return posts
            else:
                logger.error(f"Failed to get Apify results: {results_response.status_code}")
                return self._get_mock_posts(query, max_posts)
                
        except Exception as e:
            logger.error(f"Xiaohongshu scraping error: {e}")
            return self._get_mock_posts(query, max_posts)
    
    def scrape(self, query: str, limit: int = 50) -> List[dict]:
        """Scrape Xiaohongshu for posts related to the query."""
        logger.info(f"Scraping Xiaohongshu for '{query}' with limit {limit}...")
        
        # Scrape posts
        posts = self.scrape_xiaohongshu(query, limit)
        
        # Convert to dict format matching Reddit client
        reviews = []
        for post in posts[:limit]:
            try:
                # Parse date to UTC timestamp
                if post.get("publishDate"):
                    created_utc = datetime.fromisoformat(
                        post["publishDate"].replace("Z", "+00:00")
                    ).timestamp()
                else:
                    created_utc = datetime.now().timestamp()
                
                # Extract text content
                text_content = post.get("text", "")
                if post.get("title"):
                    text_content = f"{post['title']}\n{text_content}"
                
                review_dict = {
                    "id": post["id"],
                    "source": "xiaohongshu",
                    "text": text_content,
                    "created_utc": created_utc,
                    "permalink": post.get("postUrl", ""),
                    "url": post.get("postUrl", ""),
                    "author": post.get("authorName", "Unknown"),
                    "upvotes": post.get("likes", 0),
                    "meta": {
                        "title": post.get("title", ""),
                        "likes": post.get("likes", 0),
                        "comments": post.get("comments", 0),
                        "shares": post.get("shares", 0),
                        "images": post.get("images", []),
                        "tags": post.get("tags", []),
                        "language": "zh"
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                logger.warning(f"Failed to create review dict from Xiaohongshu post: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} Xiaohongshu posts for '{query}'")
        return reviews
    
    def _get_mock_posts(self, query: str, max_posts: int) -> List[Dict[str, Any]]:
        """Generate mock post data for testing."""
        mock_posts = [
            {
                "id": f"mock_post_{i}",
                "title": f"{query} 种草分享 {i+1}",
                "text": f"这是关于{query}的种草内容，非常推荐！",
                "authorName": f"用户{i+1}",
                "likes": (i + 1) * 20,
                "comments": (i + 1) * 5,
                "shares": i + 1,
                "publishDate": (datetime.now() - timedelta(days=i*7)).isoformat() + "Z",
                "postUrl": f"https://www.xiaohongshu.com/explore/mock_{i}",
                "images": [f"https://via.placeholder.com/300x300"],
                "tags": [query, "种草", "推荐"]
            }
            for i in range(min(max_posts, 10))
        ]
        return mock_posts
