"""YouTube data collection service for InsightHub."""

import logging
import re as _re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..core.config import settings

logger = logging.getLogger(__name__)


def thumbnail_from_id(video_id: str) -> str:
    """Construct a thumbnail URL from a bare video ID (API-free fallback)."""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def _fmt_duration(iso: str) -> str:
    """ISO-8601 duration (PT1H12M34S) → display string (1:12:34 / 12:34)."""
    import re as _re
    m = _re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return ""
    h, mi, s = (int(x or 0) for x in m.groups())
    return f"{h}:{mi:02d}:{s:02d}" if h else f"{mi}:{s:02d}"


def _best_thumbnail(thumbnails: dict, video_id: str) -> str:
    """Pick the highest-resolution thumbnail the API returned, else construct one."""
    for size in ("maxres", "standard", "high", "medium", "default"):
        url = (thumbnails.get(size) or {}).get("url")
        if url:
            return url
    return thumbnail_from_id(video_id)


def _strip_html(text: str) -> str:
    """Remove HTML tags from YouTube comment text and normalise whitespace."""
    if not text:
        return ""
    cleaned = _re.sub(r"<[^>]+>", " ", text)
    cleaned = _re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

class YouTubeService:
    """YouTube data collection service using YouTube Data API v3."""
    
    def __init__(self):
        self._api_key = getattr(settings, 'youtube_api_key', '')
        self.youtube = None
        self._init_youtube()

    def _build_client(self):
        """Build a fresh YouTube client (thread-safe: each call gets its own http connection)."""
        if not self._api_key:
            return None
        try:
            return build("youtube", "v3", developerKey=self._api_key, cache_discovery=False)
        except Exception as e:
            logger.error(f"Failed to build YouTube client: {e}")
            return None

    def _init_youtube(self):
        """Initialize the shared YouTube client used for non-parallel calls."""
        if self._api_key:
            self.youtube = self._build_client()
            if self.youtube:
                logger.info("YouTube client initialized successfully")
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
                    "thumbnail": _best_thumbnail(snippet.get("thumbnails", {}), video_id),
                })

            self._enrich_video_stats(videos)
            logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube search failed: {e}")
            return self._get_mock_videos(query, max_results)
    
    def _enrich_video_stats(self, videos: List[Dict[str, Any]]) -> None:
        """Attach view_count, duration, and the FULL description to each video
        via one batched videos().list call (1 quota unit). search().list only
        returns a ~160-char description stub, but full food/travel video
        descriptions typically name every place featured — a strong entity
        signal, and the main YouTube evidence when transcripts are blocked.
        Fail-open: missing stats leave view_count=0 / duration=""."""
        if not videos or not self.youtube:
            return
        try:
            ids = ",".join(v["video_id"] for v in videos if v.get("video_id"))
            resp = self.youtube.videos().list(part="snippet,statistics,contentDetails", id=ids).execute()
            by_id = {it["id"]: it for it in resp.get("items", [])}
            for v in videos:
                it = by_id.get(v.get("video_id"), {})
                v["view_count"] = int(it.get("statistics", {}).get("viewCount", 0) or 0)
                v["duration"] = _fmt_duration(it.get("contentDetails", {}).get("duration", ""))
                full_desc = (it.get("snippet", {}).get("description") or "").strip()
                if len(full_desc) > len(v.get("description") or ""):
                    v["description"] = full_desc
        except Exception as e:
            logger.warning(f"Video stats enrichment skipped: {e}")

    def get_video_comments(self, video_id: str, max_comments: int = 100,
                           _youtube_client=None) -> List[Dict[str, Any]]:
        """Get comments for a specific video."""
        yt = _youtube_client or self.youtube
        if not yt:
            return self._get_mock_comments(video_id, max_comments)

        try:
            comments = []
            next_page_token = None

            while len(comments) < max_comments:
                # Get comment threads
                request = yt.commentThreads().list(
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
                        "text": _strip_html(top_comment["textDisplay"]),
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
                                "text": _strip_html(reply_snippet["textDisplay"]),
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

    def _select_videos_with_gpt(self, query: str, videos: List[dict], top_n: int = 10) -> List[dict]:
        """Pick the most review-relevant videos for the query using GPT.

        There are no hardcoded title keywords: the model judges which titles are
        genuine, useful reviews (hands-on, long-term, comparisons) versus
        unboxings / leaks / reaction-clickbait. Falls back to the native search
        order (already relevance-ranked by YouTube) if GPT is unavailable.
        """
        if len(videos) <= top_n:
            return videos
        try:
            from .llm import LLMServiceFactory, _safe_json_loads
            svc = LLMServiceFactory.create()
            listing = "\n".join(f'{i}: {v.get("title", "")}' for i, v in enumerate(videos))
            system = (
                "You select the YouTube videos that work best as genuine, useful "
                "reviews for a user's query. Prefer hands-on reviews, long-term "
                "experience, and head-to-head comparisons; avoid unboxings, leaks, "
                "specs-only news, and reaction/clickbait. Reply with JSON only."
            )
            user = (
                f'Query: "{query}"\n\nVideos (index: title):\n{listing}\n\n'
                f'Return the indices of the up to {top_n} most useful videos, best '
                f'first, as JSON: {{"indices": [int, ...]}}'
            )
            resp = svc.chat(system, user, temperature=0.0, max_tokens=150)
            data = _safe_json_loads(resp) if resp else {}
            idxs = data.get("indices", []) if isinstance(data, dict) else []
            picked = [videos[i] for i in idxs if isinstance(i, int) and 0 <= i < len(videos)]
            if picked:
                return picked[:top_n]
        except Exception as e:
            logger.warning(f"GPT video selection failed ({e}); using native order")
        return videos[:top_n]

    def scrape(self, query: str, limit: int = 50, query_category: str = None) -> List[dict]:
        """Scrape YouTube for reviews related to the query."""
        logger.info(f"Scraping YouTube for '{query}' with limit {limit}...")

        # Search for relevant videos
        videos = self.search_videos(query, max_results=20)

        # Keep videos whose title/description overlap the query (neutral text match),
        # then let GPT pick the most review-relevant ones — no hardcoded title keywords.
        relevant_videos = [v for v in videos if self._is_relevant_video(v, query)]
        if not relevant_videos:
            relevant_videos = videos
        target_videos = self._select_videos_with_gpt(query, relevant_videos, top_n=10)
        logger.info(f"Selected {len(target_videos)} videos out of {len(videos)} total")
        per_video_limit = max(1, limit // max(1, len(target_videos)))

        def _fetch_video(video: dict) -> List[dict]:
            # Build a fresh client per thread — httplib2 is not thread-safe.
            local_yt = self._build_client()
            if local_yt is None:
                return self._get_mock_comments(video["video_id"], per_video_limit)
            comments = self.get_video_comments(video["video_id"], max_comments=per_video_limit,
                                               _youtube_client=local_yt)
            for c in comments:
                c["video_title"] = video["title"]
                c["video_id"] = video["video_id"]
                c["channel_title"] = video["channel_title"]
                c["thumbnail_url"] = video.get("thumbnail", "")
                c["view_count"] = video.get("view_count", 0)
                c["duration"] = video.get("duration", "")
            return comments

        all_comments = []
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(target_videos)))) as executor:
            futures = [executor.submit(_fetch_video, v) for v in target_videos]
            for future in as_completed(futures):
                try:
                    all_comments.extend(future.result())
                except Exception as e:
                    logger.warning(f"Failed to fetch video comments: {e}")
        
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
                    "unit_type": "comment",
                    "thread_id": comment["video_id"],
                    "post_title": comment.get("video_title", ""),
                    "meta": {
                        "video_title": comment.get("video_title", ""),
                        "channel_title": comment.get("channel_title", ""),
                        "reply_count": comment.get("reply_count", 0),
                        "is_reply": comment.get("is_reply", False),
                        # API thumbnail; falls back to ID-constructed URL if absent.
                        "thumbnail_url": comment.get("thumbnail_url") or thumbnail_from_id(comment["video_id"]),
                        "view_count": comment.get("view_count", 0),
                        "duration": comment.get("duration", ""),
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                logger.warning(f"Failed to create review dict from YouTube comment: {e}")
                continue
        
        # Add the video itself as analyzable content: its title+description (the
        # "post") and, when available, its transcript. These let recommendations
        # made by the creator — not just commenters — feed the ranking, and give
        # the scorer post↔comment corroboration signal.
        for v in target_videos:
            vid = v.get("video_id", "")
            if not vid:
                continue
            title = v.get("title", "") or ""
            desc = v.get("description", "") or ""
            post_body = (title + "\n\n" + desc).strip()
            if len(post_body) > 40:
                reviews.append({
                    "id": f"ytpost_{vid}",
                    "source": "youtube",
                    "text": post_body,
                    "created_utc": None,
                    "permalink": f"https://youtube.com/watch?v={vid}",
                    "url": f"https://youtube.com/watch?v={vid}",
                    "author": v.get("channel_title", "") or "",
                    "upvotes": 0,
                    "unit_type": "post",
                    "thread_id": vid,
                    "post_title": title,
                    "meta": {"video_title": title, "channel_title": v.get("channel_title", ""),
                             "thumbnail_url": v.get("thumbnail") or thumbnail_from_id(vid),
                             "view_count": v.get("view_count", 0), "duration": v.get("duration", "")},
                })
            transcript = self._fetch_transcript(vid)
            if transcript:
                reviews.append({
                    "id": f"yttrans_{vid}",
                    "source": "youtube",
                    "text": transcript,
                    "created_utc": None,
                    "permalink": f"https://youtube.com/watch?v={vid}",
                    "url": f"https://youtube.com/watch?v={vid}",
                    "author": v.get("channel_title", "") or "",
                    "upvotes": 0,
                    "unit_type": "transcript",
                    "thread_id": vid,
                    "post_title": title,
                    "meta": {"video_title": title, "channel_title": v.get("channel_title", ""),
                             "thumbnail_url": v.get("thumbnail") or thumbnail_from_id(vid),
                             "view_count": v.get("view_count", 0), "duration": v.get("duration", "")},
                })

        n_transcripts = sum(1 for r in reviews if r.get("unit_type") == "transcript")
        if target_videos and not n_transcripts:
            logger.info(
                "No transcripts retrieved for any video — YouTube is likely "
                "blocking this IP (VPN/datacenter). Set TRANSCRIPT_PROXY to route "
                "around it; falling back to full descriptions + comments.")
        logger.info(f"Scraped {len(reviews)} YouTube reviews for '{query}'")
        return reviews

    def _transcript_api(self):
        """Build a YouTubeTranscriptApi honoring the optional TRANSCRIPT_PROXY
        env var (http(s) proxy URL) — the documented workaround for YouTube
        IP-blocking transcript requests (VPNs/datacenter IPs)."""
        from youtube_transcript_api import YouTubeTranscriptApi
        import os
        proxy = os.environ.get("TRANSCRIPT_PROXY", "").strip()
        if proxy:
            try:
                from youtube_transcript_api.proxies import GenericProxyConfig
                return YouTubeTranscriptApi(
                    proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy))
            except Exception as e:
                logger.warning(f"TRANSCRIPT_PROXY ignored ({e}); using direct connection")
        return YouTubeTranscriptApi()

    def _fetch_transcript(self, video_id: str, max_chars: int = 4000) -> str:
        """Fetch a video's transcript as plain text, truncated to max_chars.

        Returns "" when transcripts are unavailable, disabled, or the optional
        youtube-transcript-api dependency is not installed — never raises.
        """
        if not video_id or str(video_id).startswith("mock"):
            return ""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except Exception:
            logger.debug("youtube-transcript-api not installed; skipping transcripts")
            return ""
        try:
            # Support both the v1.x instance API (.fetch) and the legacy
            # classmethod (.get_transcript), which return objects/dicts respectively.
            if hasattr(YouTubeTranscriptApi, "fetch"):
                fetched = self._transcript_api().fetch(video_id)
                parts = [getattr(s, "text", "") for s in fetched]
            else:  # legacy <1.0
                segments = YouTubeTranscriptApi.get_transcript(video_id)
                parts = [seg.get("text", "") for seg in segments]
            text = " ".join(p for p in parts if p)
            text = " ".join(text.split())  # collapse whitespace/newlines
            return text[:max_chars]
        except Exception as e:
            self._transcript_failures = getattr(self, "_transcript_failures", 0) + 1
            logger.debug(f"No transcript for {video_id}: {e}")
            return ""
    
    def _get_mock_videos(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock video data for testing."""
        mock_videos = [
            {
                "video_id": f"mock_video_{i}",
                "title": f"{query} Review - Video {i+1}",
                "description": f"This is a mock video about {query}",
                "channel_title": f"Review Channel {i+1}",
                "published_at": (datetime.now() - timedelta(days=i*7)).isoformat() + "Z",
                "thumbnail": "https://via.placeholder.com/120x90",
                "view_count": (i + 1) * 12000,
                "duration": f"{8 + i}:{15 + i:02d}",
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
