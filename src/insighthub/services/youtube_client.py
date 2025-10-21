"""YouTube data collection service for InsightHub."""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Review import removed - using dict format
from ..core.config import settings

logger = logging.getLogger(__name__)

class YouTubeService:
    """YouTube data collection service using YouTube Data API v3."""
    
    def __init__(self):
        self.youtube = None
        self._init_youtube()
    
    def _init_youtube(self):
        """Initialize YouTube client."""
        youtube_key = getattr(settings, 'youtube_api_key', '')
        if youtube_key:
            try:
                self.youtube = build("youtube", "v3", developerKey=youtube_key)
                logger.info("YouTube client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube client: {e}")
                self.youtube = None
        else:
            logger.warning("YouTube API key not provided, using mock data")
    
    def search_videos(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for videos related to the query."""
        if not self.youtube:
            return self._get_mock_videos(query, max_results)
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=max_results,
                type="video",
                order="relevance",
                publishedAfter=(datetime.now() - timedelta(days=365)).isoformat() + "Z"
            ).execute()
            
            videos = []
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                
                videos.append({
                    "video_id": video_id,
                    "title": snippet["title"],
                    "description": snippet["description"],
                    "channel_title": snippet["channelTitle"],
                    "published_at": snippet["publishedAt"],
                    "thumbnail": snippet["thumbnails"]["default"]["url"]
                })
            
            logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube search failed: {e}")
            return self._get_mock_videos(query, max_results)
    
    def get_video_comments(self, video_id: str, max_comments: int = 100) -> List[Dict[str, Any]]:
        """Get comments for a specific video."""
        if not self.youtube:
            return self._get_mock_comments(video_id, max_comments)
        
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_comments:
                # Get comment threads
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page_token,
                    order="relevance"
                )
                
                response = request.execute()
                
                for item in response.get("items", []):
                    # Top-level comment
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "comment_id": item["id"],
                        "author": top_comment["authorDisplayName"],
                        "text": top_comment["textDisplay"],
                        "likes": top_comment["likeCount"],
                        "published_at": top_comment["publishedAt"],
                        "updated_at": top_comment["updatedAt"],
                        "reply_count": item["snippet"]["totalReplyCount"]
                    })
                    
                    # Include replies if available
                    if "replies" in item and len(comments) < max_comments:
                        for reply in item["replies"]["comments"][:5]:  # Limit replies
                            reply_snippet = reply["snippet"]
                            comments.append({
                                "comment_id": reply["id"],
                                "author": reply_snippet["authorDisplayName"],
                                "text": reply_snippet["textDisplay"],
                                "likes": reply_snippet["likeCount"],
                                "published_at": reply_snippet["publishedAt"],
                                "updated_at": reply_snippet["updatedAt"],
                                "reply_count": 0,
                                "is_reply": True
                            })
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            logger.info(f"Retrieved {len(comments)} comments for video {video_id}")
            return comments[:max_comments]
            
        except HttpError as e:
            logger.error(f"YouTube comments fetch failed: {e}")
            return self._get_mock_comments(video_id, max_comments)
    
    def _is_relevant_video(self, video: dict, query: str) -> bool:
        """Check if video is relevant to the query."""
        query_words = set(query.lower().split())
        title_words = set(video.get("title", "").lower().split())
        desc_words = set(video.get("description", "").lower().split()[:100])  # First 100 words of description
        
        # Check if query words appear in title or description
        title_match = len(query_words.intersection(title_words)) > 0
        desc_match = len(query_words.intersection(desc_words)) > 0
        
        return title_match or desc_match

    def scrape(self, query: str, limit: int = 50) -> List[dict]:
        """Scrape YouTube for reviews related to the query."""
        logger.info(f"Scraping YouTube for '{query}' with limit {limit}...")
        
        # Search for relevant videos
        videos = self.search_videos(query, max_results=20)
        
        # Filter for relevant videos only
        relevant_videos = [v for v in videos if self._is_relevant_video(v, query)]
        logger.info(f"Found {len(relevant_videos)} relevant videos out of {len(videos)} total")
        
        all_comments = []
        for video in relevant_videos[:10]:  # Limit to top 10 relevant videos
            video_comments = self.get_video_comments(
                video["video_id"], 
                max_comments=limit // 10
            )
            
            # Add video context to comments
            for comment in video_comments:
                comment["video_title"] = video["title"]
                comment["video_id"] = video["video_id"]
                comment["channel_title"] = video["channel_title"]
            
            all_comments.extend(video_comments)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Convert to dict format matching Reddit client
        reviews = []
        for comment in all_comments[:limit]:
            # Basic relevance filter for comments
            comment_text = comment.get("text", "").lower()
            query_words = set(query.lower().split())
            comment_words = set(comment_text.split())
            
            # Skip if comment doesn't mention any query words and is too short
            if len(comment_text) < 20 or (len(query_words.intersection(comment_words)) == 0 and len(comment_text) < 50):
                continue
                
            try:
                # Parse timestamp to UTC
                created_utc = datetime.fromisoformat(
                    comment["published_at"].replace("Z", "+00:00")
                ).timestamp()
                
                review_dict = {
                    "id": comment["comment_id"],
                    "source": "youtube",
                    "text": comment["text"],
                    "created_utc": created_utc,
                    "permalink": f"https://youtube.com/watch?v={comment['video_id']}#comment-{comment['comment_id']}",
                    "url": f"https://youtube.com/watch?v={comment['video_id']}",
                    "author": comment["author"],
                    "upvotes": comment["likes"],
                    "meta": {
                        "video_title": comment.get("video_title", ""),
                        "channel_title": comment.get("channel_title", ""),
                        "reply_count": comment.get("reply_count", 0),
                        "is_reply": comment.get("is_reply", False)
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                logger.warning(f"Failed to create review dict from YouTube comment: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} YouTube reviews for '{query}'")
        return reviews
    
    def _get_mock_videos(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock video data for testing."""
        mock_videos = [
            {
                "video_id": f"mock_video_{i}",
                "title": f"{query} Review - Video {i+1}",
                "description": f"This is a mock video about {query}",
                "channel_title": f"Review Channel {i+1}",
                "published_at": (datetime.now() - timedelta(days=i*7)).isoformat() + "Z",
                "thumbnail": "https://via.placeholder.com/120x90"
            }
            for i in range(min(max_results, 5))
        ]
        return mock_videos
    
    def _get_mock_comments(self, video_id: str, max_comments: int) -> List[Dict[str, Any]]:
        """Generate mock comment data for testing."""
        mock_comments = [
            {
                "comment_id": f"mock_comment_{i}",
                "author": f"User{i+1}",
                "text": f"This is a mock comment about the video {video_id}. Great content!",
                "likes": (i + 1) * 5,
                "published_at": (datetime.now() - timedelta(days=i)).isoformat() + "Z",
                "updated_at": (datetime.now() - timedelta(days=i)).isoformat() + "Z",
                "reply_count": i % 3
            }
            for i in range(min(max_comments, 20))
        ]
        return mock_comments
