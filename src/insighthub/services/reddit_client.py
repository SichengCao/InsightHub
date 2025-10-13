"""Reddit data collection service."""

import logging
import time
import random
import unicodedata
import re
from typing import List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import praw
from ..core.models import Review
from ..core.config import settings

# --- Search planning dataclass and helpers ---
@dataclass
class SearchPlan:
    terms: list
    subreddits: list
    strategies: list
    time_filter: str
    min_comment_score: int
    per_post_top_n: int
    comment_must_patterns: list

def _norm_q(s: str) -> str:
    """Normalize query string for search planning."""
    return re.sub(r"\s+"," ", unicodedata.normalize("NFKC", (s or "")).strip())

# --- Anti-bot heuristics & text quality gates ---
import hashlib, math, time
from collections import Counter
from ..core.constants import SearchConstants, ErrorConstants

STOP_USERS = {"AutoModerator"}
WORD_RE = re.compile(r"[A-Za-z]{2,}")
URL_RE  = re.compile(r"https?://\S+")
CONTACT_RE = re.compile(r"(whats?app|telegram|wechat|kik|line|viber|dm\s+me|contact\s+me|text\s+me)", re.I)
REF_RE = re.compile(r"(ref=|utm_|affid=|affiliate|promo\s*code)", re.I)
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]")  # wide range
CODEBLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

def _norm_text(t: str) -> str:
    t = URL_RE.sub("", t or "")
    t = CODEBLOCK_RE.sub("", t)
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return t

def _text_hash(t: str) -> str:
    return hashlib.sha1(_norm_text(t).encode("utf-8")).hexdigest()

def _alpha_ratio(s: str) -> float:
    if not s: return 0.0
    alpha = sum(ch.isalpha() for ch in s)
    return alpha / max(1, len(s))

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((cnt/n) * math.log2(cnt/n) for cnt in c.values())

def _looks_english(s: str, min_ratio: float = 0.75) -> bool:
    if not s: return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in s)
    total_letters = sum(ch.isalpha() for ch in s)
    if total_letters == 0: return False
    return ascii_letters / total_letters >= min_ratio

# Legacy function removed - subreddit selection now handled by LLM planning

def _author_meta(comment):
    """
    Extract author metadata WITHOUT triggering lazy API calls.
    Returns conservative defaults if data is not already loaded.
    """
    try:
        # Check if author is already loaded (avoid lazy API call)
        a = comment.__dict__.get("author", None)
        if not a:
            return 0, 0, 9999  # karma, link_karma, age_days (pass by default)
        
        # Only access attributes if they're already cached
        karma = getattr(a, "comment_karma", 0) or 0
        link_karma = getattr(a, "link_karma", 0) or 0
        created_utc = getattr(a, "created_utc", None)
        
        now = time.time()
        age_days = (now - created_utc) / 86400 if created_utc else 9999
        return karma, link_karma, age_days
    except Exception:
        # If ANY attribute access fails, return permissive defaults
        return 0, 0, 9999

BOT_PHRASES = (
    "check my profile", "click my bio", "free giveaway", "investment manager",
    "message me on", "join our channel", "vip signal", "forex", "crypto signal",
    "reach me via", "whatsapp", "telegram", "weekly returns", "guaranteed profit"
)

def _botlike_score(comment) -> float:
    body = getattr(comment, "body", "") or ""
    bnorm = _norm_text(body)
    up = getattr(comment, "score", 0) or 0
    n_urls = len(URL_RE.findall(body))
    n_emojis = len(EMOJI_RE.findall(body))
    caps_ratio = sum(ch.isupper() for ch in body) / max(1, sum(ch.isalpha() for ch in body))
    ent = _shannon_entropy(bnorm)
    karma, link_karma, age_days = _author_meta(comment)

    score = 0.0
    if any(p in bnorm for p in BOT_PHRASES): score += 2.0
    if CONTACT_RE.search(bnorm): score += 2.0
    if REF_RE.search(bnorm): score += 1.0
    if n_urls >= 2: score += 1.2
    if n_emojis >= 8: score += 0.8
    if caps_ratio > 0.35: score += 0.6
    if ent < 2.2: score += 0.8                     # very repetitive / templated
    if up <= 0: score += 0.4
    if karma < 20 and age_days < 14: score += 1.5  # very new + low karma account
    if getattr(comment, "distinguished", None) in {"moderator", "admin"}:
        score -= 1.0                               # trusted

    return score

def _shingle_sig(s: str, n=8) -> str:
    """Create shingle-based signature for detecting copypasta."""
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    shingles = [" ".join(toks[i:i+n]) for i in range(max(1, len(toks)-n+1))]
    h = hashlib.md5("||".join(shingles[:40]).encode()).hexdigest()
    return h

def _passes_quality(comment, query, settings, body: str = None) -> bool:
    """
    Check if comment passes quality filters WITHOUT triggering lazy API calls.
    
    Args:
        comment: Reddit comment object
        query: Search query string
        settings: Search settings
        body: Comment body text (if already loaded, to avoid re-fetching)
    """
    # Get body if not provided, without triggering lazy load
    if body is None:
        body = comment.__dict__.get("body", "") or ""
    if not body:
        return False
    
    # Check author without triggering lazy load
    try:
        author = comment.__dict__.get("author", None)
        if author and str(author) in STOP_USERS:
            return False
    except Exception:
        pass  # If author check fails, proceed with other filters
    
    # Skip brand-new accounts
    karma, link_karma, age_days = _author_meta(comment)
    if age_days < SearchConstants.MIN_ACCOUNT_AGE_DAYS and karma < SearchConstants.MIN_ACCOUNT_KARMA:
        return False
    
    # Use __dict__ to avoid lazy loading score
    score = comment.__dict__.get("score", 0) or 0
    if score < SearchConstants.MIN_COMMENT_SCORE:
        return False
    if len(body) < getattr(settings, "min_comment_len", SearchConstants.MIN_COMMENT_LENGTH): 
        return False
    if _alpha_ratio(body) < getattr(settings, "min_alpha_ratio", SearchConstants.MIN_ALPHA_RATIO):
        return False
    if body.strip().endswith("?") and len(body) < 140:
        return False
    if len(WORD_RE.findall(body)) < SearchConstants.MIN_WORD_COUNT:
        return False
    if not _looks_english(body, 0.75):
        return False
    # topical relevance (reuse your existing relevance function if present)
    try:
        if hasattr(comment, "_is_relevant_comment") and not comment._is_relevant_comment(body, query):
            return False
    except Exception:
        pass
    # botlike composite - lowered threshold
    if _botlike_score(comment) >= 2.5:
        return False
    return True

def passes_quality(comment, settings) -> bool:
    """Legacy wrapper for backward compatibility"""
    return _passes_quality(comment, "", settings)

logger = logging.getLogger(__name__)


class RedditService:
    """Reddit data collection service."""
    
    def __init__(self):
        self.reddit = None
        self._init_reddit()
    
    def _init_reddit(self):
        """Initialize Reddit client."""
        if (settings.reddit_client_id and 
            settings.reddit_client_secret and 
            settings.reddit_user_agent):
            try:
                self.reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit credentials not provided, using mock data")
    
    def _fallback_plan(self, query: str) -> SearchPlan:
        """Generate fallback search plan without hard-coded aliases."""
        q = _norm_q(query)
        terms = {q, f"{q} review", f"{q} owner", f"{q} experience", f"{q} recommendations", f"{q} best"}
        m = re.match(r"([A-Za-z]+)\s*([0-9]+)$", q)
        if m:
            terms.add(f"{m.group(1)}{m.group(2)}")
            terms.add(f"{m.group(1)} {m.group(2)}")
        terms = [t for t in terms if t][:8]

        # Discover subreddits via API only (no static lists)
        subs = []
        try:
            if self.reddit:
                for sr in self.reddit.subreddits.search(q, limit=25):
                    name = str(getattr(sr,"display_name","") or "")
                    subs_count = int(getattr(sr,"subscribers",0) or 0)
                    if name and subs_count >= 2000 and not getattr(sr,"over18",False):
                        subs.append(name)
        except Exception:
            pass
        subs = list(dict.fromkeys(subs))[:10] + ["all"]

        # Create more flexible comment patterns that require ANY of the main query terms
        query_words = [w.lower() for w in re.findall(r'\b\w+\b', q) if len(w) > 2]
        main_patterns = [r"\b(love|hate|recommend|avoid|worth|issue|problem|help|good|bad|great|terrible|best|worst)\b"]
        if query_words:
            # Require at least one main query word to appear (more flexible)
            main_patterns.append(rf"\b({'|'.join(query_words[:4])})\b")

        return SearchPlan(
            terms=terms,
            subreddits=subs,
            strategies=["relevance","top"],
            time_filter="year",
            min_comment_score=1,
            per_post_top_n=8,
            comment_must_patterns=main_patterns
        )

    def _compile_comment_filter(self, patterns: list):
        """Compile regex patterns for comment filtering."""
        em = [re.compile(p, re.I) for p in (patterns or [])]
        def _ok(text: str) -> bool:
            if not em: return True
            return any(rx.search(text or "") for rx in em)
        return _ok
    
    def _execute_single_search(self, bucket: str, term: str, strategy: str, plan: SearchPlan, seen_posts: set) -> tuple:
        """
        Execute a single Reddit search and return comments.
        
        Args:
            bucket: Subreddit combination (e.g., "AskNYC+NewYorkCity")
            term: Search term
            strategy: Search strategy ("top", "relevance", "new")
            plan: Search plan with parameters
            seen_posts: Set of already processed post IDs
            
        Returns:
            Tuple of (comments, processed_post_count, search_identifier)
        """
        comments = []
        search_id = f"r/{bucket} '{term}' ({strategy})"
        
        try:
            sr = self.reddit.subreddit(bucket)
            submissions = sr.search(term, sort=strategy, time_filter=plan.time_filter, syntax="lucene", limit=SearchConstants.REDDIT_SEARCH_LIMIT)
            
            submissions_processed = 0
            for sub in submissions:
                sid = getattr(sub, "id", None)
                if not sid or sid in seen_posts:
                    continue
                seen_posts.add(sid)  # Thread-safe with GIL
                submissions_processed += 1
                
                # Get top-level comments without expanding nested threads (faster!)
                # Skip replace_more() as it's expensive and we only need top comments
                flat = []
                for c in sub.comments:
                    # Skip "MoreComments" objects
                    if not hasattr(c, "body"):
                        continue
                    
                    # Force load body and score NOW (during parallel execution)
                    # This way lazy loading happens in parallel, not sequentially later
                    try:
                        body = c.body  # Force load
                        score = c.score  # Force load
                    except Exception:
                        continue
                    
                    # Quick quality check
                    if score >= plan.min_comment_score and len(body) > SearchConstants.MIN_COMMENT_LENGTH:
                        flat.append((c, score))
                
                # Sort by score and take top N per post
                flat.sort(key=lambda x: x[1], reverse=True)
                comments.extend([c for c, _ in flat[:plan.per_post_top_n]])
            
            return (comments, submissions_processed, search_id)
            
        except Exception as e:
            logger.debug(f"Search failed {search_id}: {e}")
            return ([], 0, search_id)

    def _plan_search(self, query: str, max_subreddits: int = 4) -> SearchPlan:
        """Plan search using LLM with fallback to API-only discovery."""
        try:
            # Import LLM service
            from .llm import LLMServiceFactory
            llm = LLMServiceFactory.create()
            plan = llm.plan_reddit_search(query, max_subreddits)
            return SearchPlan(**plan)
        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
            return self._fallback_plan(query)
    
    def _get_search_terms(self, query: str) -> List[str]:
        """Generate multiple search terms for better coverage."""
        query_lower = query.lower()
        terms = [query]  # Original query
        
        # Add review-specific terms
        review_terms = ['review', 'opinion', 'experience', 'thoughts', 'impression']
        for term in review_terms:
            terms.append(f"{query} {term}")
        
        # Add product-specific terms
        if any(word in query_lower for word in ['iphone', 'samsung', 'google pixel', 'oneplus']):
            terms.extend([f"{query} camera", f"{query} battery", f"{query} performance"])
        elif any(word in query_lower for word in ['tesla', 'model y', 'model 3', 'ford', 'bmw']):
            terms.extend([f"{query} range", f"{query} charging", f"{query} autopilot"])
        elif any(word in query_lower for word in ['macbook', 'laptop', 'dell', 'hp']):
            terms.extend([f"{query} performance", f"{query} battery", f"{query} keyboard"])
        
        # Add ownership terms
        ownership_terms = ['own', 'owned', 'using', 'bought', 'purchased']
        for term in ownership_terms:
            terms.append(f"{term} {query}")
        
        return terms[:5]  # Limit to 5 terms to avoid rate limiting
    
    def _is_review_relevant(self, submission, query: str) -> bool:
        """Check if a Reddit post is review-relevant."""
        title = getattr(submission, 'title', '').lower()
        selftext = getattr(submission, 'selftext', '').lower()
        query_lower = query.lower()
        
        # Must contain the main query term
        if query_lower not in title and query_lower not in selftext:
            return False
        
        # Look for review indicators
        review_indicators = [
            'review', 'opinion', 'experience', 'thoughts', 'impression',
            'own', 'owned', 'using', 'bought', 'purchased', 'have had',
            'pros and cons', 'good and bad', 'likes and dislikes',
            'worth it', 'recommend', 'avoid', 'love', 'hate'
        ]
        
        content = f"{title} {selftext}"
        return any(indicator in content for indicator in review_indicators)
    
    def _is_comment_review_relevant(self, comment, query: str) -> bool:
        """Check if a comment contains review-relevant content."""
        body = getattr(comment, 'body', '').lower()
        query_lower = query.lower()
        
        # Must mention the product
        if query_lower not in body:
            return False
        
        # Look for review language patterns (more lenient)
        review_patterns = [
            # Ownership indicators
            r'\b(own|owned|using|bought|purchased|have had|got|received|have|had)\b',
            # Experience indicators  
            r'\b(experience|experienced|tried|tested|used|using|use)\b',
            # Opinion indicators (expanded)
            r'\b(love|hate|like|dislike|recommend|avoid|worth|not worth|amazing|terrible|awesome|sucks|perfect|awful)\b',
            # Quality indicators (expanded)
            r'\b(good|bad|great|terrible|excellent|awful|amazing|horrible|nice|poor|fantastic|disappointing|impressive|mediocre)\b',
            # Specific feature mentions (expanded)
            r'\b(camera|battery|performance|design|price|quality|build|screen|display|audio|sound|speed|fast|slow|lag|bug|issue|problem)\b',
            # Comparison indicators
            r'\b(compared to|vs|versus|better than|worse than|instead of|rather than)\b',
            # General sentiment words
            r'\b(yes|no|definitely|absolutely|never|always|sometimes|often|rarely|usually)\b'
        ]
        
        import re
        return any(re.search(pattern, body) for pattern in review_patterns)
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def scrape(self, query: str, limit: int = 50, max_subreddits: int = 4) -> List[Review]:
        """Scrape Reddit for reviews."""
        if self.reddit:
            return self._scrape_real(query, limit, max_subreddits)
        else:
            return self._scrape_mock(query, limit)
    
    def _scrape_real(self, query: str, limit: int, max_subreddits: int = 4) -> List[Review]:
        """
        Scrape real Reddit data using universal search planning.
        
        This method implements a sophisticated multi-stage search and filtering process:
        1. Generate search plan via LLM (subreddits, terms, strategies)
        2. Execute Reddit API searches with rate limiting
        3. Extract and filter comments using quality gates
        4. Apply pattern matching and deduplication
        5. Return structured Review objects
        
        Args:
            query: User search query (e.g., "best iPhone 15")
            limit: Target number of comments to return
            max_subreddits: Maximum number of subreddits to search
            
        Returns:
            List of Review objects with filtered, high-quality comments
        """
        import time
        start_time = time.time()
        
        # Step 1: Generate intelligent search plan using LLM
        plan = self._plan_search(query, max_subreddits)
        raw_comments = []
        seen_posts = set()  # Track processed posts to avoid duplicates
        
        logger.info(f"Searching for '{query}' with limit={limit}, max_raw={limit*5}, subreddits={plan.subreddits}, terms={plan.terms}")

        try:
            # Step 2: Prepare subreddit search buckets
            # Combine multiple subreddits into search buckets for efficient API usage
            subs = [s for s in plan.subreddits if s] or ["all"]
            combined = "+".join([s for s in subs if s.lower() != "all"][:10])  # Reddit API limit
            buckets = [combined] if combined else []
            # Removed "all" fallback for faster, more focused searches

            # Step 3: Execute parallel multi-dimensional search strategy
            # Build all search tasks (bucket × term × strategy combinations)
            search_tasks = []
            for bucket in buckets:
                for term in plan.terms:
                    for strategy in plan.strategies:
                        search_tasks.append((bucket, term, strategy))
            
            total_searches = len(search_tasks)
            search_count = 0
            start_time = time.time()
            logger.info(f"🚀 Starting {total_searches} parallel Reddit API searches...")
            
            # Execute searches in parallel using ThreadPoolExecutor
            # Max 8 workers for maximum speed
            max_workers = min(8, total_searches)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all search tasks at once
                future_to_task = {
                    executor.submit(self._execute_single_search, bucket, term, strategy, plan, seen_posts): (bucket, term, strategy)
                    for bucket, term, strategy in search_tasks
                }
                
                # Process completed searches as they finish
                for future in as_completed(future_to_task):
                    bucket, term, strategy = future_to_task[future]
                    search_count += 1
                    elapsed = time.time() - start_time
                    
                    try:
                        comments, submissions_processed, search_id = future.result()
                        raw_comments.extend(comments)
                        
                        logger.info(f"🔍 [{search_count}/{total_searches}] {search_id} | Posts: {submissions_processed} | Comments: {len(raw_comments)} | {elapsed:.1f}s")
                            
                    except Exception as e:
                        logger.debug(f"Search task failed for r/{bucket} '{term}' ({strategy}': {e}")
                        continue
                
                # Log completion after all searches finish
                elapsed_final = time.time() - start_time
                logger.info(f"✅ All {search_count} searches completed in {elapsed_final:.1f}s")
        except Exception as e:
            logger.error(f"Reddit scraping failed: {e}")
            return self._scrape_mock(query, limit)

        # Step 5: Advanced multi-stage filtering pipeline
        # This implements sophisticated comment filtering with multiple quality gates
        filter_start = time.time()
        logger.info(f"🔄 Starting filtering of {len(raw_comments)} comments...")
        
        seen, shingles, filtered = set(), set(), []
        ok = self._compile_comment_filter(plan.comment_must_patterns)
        
        # Prepare query relevance patterns for fallback filtering
        # Extract meaningful words from user query for pattern matching
        query_lower = query.lower()
        query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) > 2]
        query_patterns = [re.compile(rf'\b{re.escape(w)}\b', re.I) for w in query_words[:3]]  # Top 3 words
        
        # Filtering statistics for transparency
        quality_filtered = 0
        pattern_filtered = 0
        dedupe_filtered = 0
        
        # Apply multi-stage filtering to each comment
        for c in raw_comments:
            try:
                # Get body without triggering lazy load
                body = c.__dict__.get("body", "") or ""
                if not body:
                    quality_filtered += 1
                    continue
                
                # Stage 1: Quality filtering (account age, karma, length, language)
                # Pass body to avoid re-fetching
                if not _passes_quality(c, query, settings, body): 
                    quality_filtered += 1
                    continue
                
                # Stage 2: Pattern filtering (flexible relevance matching)
                # Comments must match either LLM-generated patterns OR query words
                if not ok(body) and not any(p.search(body) for p in query_patterns):
                    pattern_filtered += 1
                    continue
                
                # Stage 3: Deduplication (exact and fuzzy matching)
                h = _text_hash(body)
                if h in seen: 
                    dedupe_filtered += 1
                    continue
                seen.add(h)
                
                # Stage 4: Shingle-based similarity detection
                sig = _shingle_sig(body)
                if sig in shingles: 
                    dedupe_filtered += 1
                    continue
                shingles.add(sig)
                
                # Comment passed all filters
                filtered.append(c)
            except Exception:
                quality_filtered += 1
                continue

        # Canonical mapping - avoid ALL lazy loading
        reviews = []
        for c in filtered[:limit]:
            # Use __dict__ for ALL attributes to prevent lazy API calls
            rid = str(c.__dict__.get("id", ""))
            
            # Get author without lazy load
            author = c.__dict__.get("author", None)
            try:
                author_name = getattr(author, "name", None) or (str(author) if author else "u/[deleted]")
            except Exception:
                author_name = "u/[deleted]"
            
            # Get all other attributes from __dict__
            score = int(c.__dict__.get("score", 0) or c.__dict__.get("upvotes", 0) or 0)
            permalink = c.__dict__.get("permalink", "") or ""
            url = c.__dict__.get("url", "") or (f"https://reddit.com{permalink}" if permalink else "")
            text = c.__dict__.get("body", "") or ""
            created_utc = c.__dict__.get("created_utc", None)
            
            reviews.append({
                "id": rid,
                "source": "reddit",
                "text": text,
                "created_utc": created_utc,
                "permalink": permalink,
                "url": url,
                "author": author_name,
                "upvotes": score,
            })
        
        # Log detailed timing breakdown
        filter_time = time.time() - filter_start
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Filtering completed in {filter_time:.1f}s | Mapping to Review objects completed")
        logger.info(f"🎉 Search completed in {elapsed_time:.1f}s: {len(raw_comments)} raw -> quality_filtered={quality_filtered} -> pattern_filtered={pattern_filtered} -> dedupe_filtered={dedupe_filtered} -> {len(filtered)} filtered -> {len(reviews)} final")
        logger.info(f"📊 API calls made: {search_count}/{total_searches} searches | Avg: {elapsed_time/search_count:.2f}s per search")
        return reviews
    
    def _scrape_mock(self, query: str, limit: int) -> List[Review]:
        """Generate comprehensive mock reviews for testing with quality filtering."""
        # Create diverse mock comments with different sentiments and aspects
        mock_comments = []
        
        # Positive reviews (40% of total)
        positive_templates = [
            f"I've been using the {query} for 6 months now and I'm absolutely blown away by its performance. The build quality is exceptional and it handles everything I throw at it without breaking a sweat. The battery life is outstanding - I can go a full day of heavy usage without worrying about charging. The camera quality is phenomenal, especially in low light conditions. The user interface is intuitive and responsive. The only minor issue I've noticed is that it can get a bit warm during intensive tasks, but it's never been a problem. Overall, this is one of the best purchases I've made in years. Highly recommend to anyone looking for a premium experience.",
            
            f"After extensive research and comparison, I decided to go with the {query}. The initial setup was smooth and the learning curve wasn't too steep. The design is sleek and modern, definitely stands out from the competition. Performance-wise, it's been solid for my daily tasks. The display quality is crisp and vibrant, perfect for both work and entertainment. However, I've encountered some software bugs that occasionally affect the user experience. The customer support has been helpful but the response time could be better. The price point is on the higher side, but considering the features and build quality, it's justified. Would recommend with some reservations about the software stability.",
            
            f"The {query} has been a game-changer for my workflow. The processing power is incredible - I can run multiple demanding applications simultaneously without any performance issues. The storage capacity is generous and the transfer speeds are lightning fast. The connectivity options are comprehensive, making it easy to integrate with my existing setup. The audio quality is surprisingly good for built-in speakers. The keyboard and trackpad are comfortable for long typing sessions. The only downside is the weight - it's heavier than I expected, which affects portability. The price is steep, but the productivity gains have more than justified the investment. This is definitely a professional-grade device.",
            
            f"After months of deliberation, I finally purchased the {query} and I couldn't be happier with my decision. The attention to detail in the design is remarkable - every aspect feels carefully considered. The performance exceeds my expectations, handling complex tasks with ease. The battery life is exceptional, easily lasting through multiple days of moderate usage. The camera system is outstanding, producing professional-quality photos and videos. The user experience is smooth and intuitive, with thoughtful features that enhance productivity. The build quality is top-notch, feeling premium and durable. The customer support has been excellent, with quick responses and helpful solutions. While the price is high, the quality and features justify the investment. This is a premium product that delivers on its promises.",
            
            f"The {query} has exceeded all my expectations. The build quality is exceptional, with attention to detail that's immediately apparent. The performance is outstanding - it handles everything I throw at it without breaking a sweat. The battery life is impressive, easily lasting through a full day of heavy usage. The camera system is phenomenal, producing stunning photos and videos in various conditions. The user interface is intuitive and responsive, making daily tasks enjoyable. The connectivity options are comprehensive and reliable. The audio quality is excellent for both music and calls. The software updates are regular and bring meaningful improvements. While the price is high, the quality and features make it worth every penny. This is a premium product that delivers exceptional value.",
            
            f"My experience with the {query} has been overwhelmingly positive. The attention to detail in both hardware and software is remarkable. The performance is consistently excellent, handling demanding tasks with ease. The battery life is outstanding, often lasting multiple days with moderate usage. The camera system is exceptional, producing professional-quality results. The user experience is smooth and intuitive, with thoughtful design choices throughout. The build quality is premium, feeling solid and well-crafted. The connectivity is reliable and fast. The audio quality is impressive for built-in speakers. The customer support has been responsive and helpful. While the price is high, the quality and features justify the investment. This is a premium product that delivers on every front."
        ]
        
        # Negative reviews (30% of total) - Diverse negative templates
        negative_templates = [
            f"I absolutely hate the {query}. This is the worst product I've ever bought. The build quality is terrible and it feels like cheap garbage. The performance is awful - it's constantly freezing, crashing, and lagging. The battery life is pathetic and dies within hours. The camera is completely useless and produces horrible photos. The user interface is confusing and frustrating. I've had nothing but problems since day one. The customer service is terrible and never helps. This is a complete waste of money. I regret buying this piece of junk. Do not buy this garbage product under any circumstances. It's the biggest mistake I've ever made.",
            
            f"Worst purchase ever! The {query} is a complete disaster. After using it for months, I can confidently say this is garbage. The design looks cheap and feels flimsy. Performance is consistently terrible - apps crash constantly and everything runs slow. Battery drains in just a few hours of normal use. Camera quality is laughably bad, photos look like they were taken with a potato. The software is buggy and frustrating to navigate. Customer support is non-existent. Save your money and buy literally anything else. This thing belongs in the trash.",
            
            f"I'm so disappointed with the {query}. What a waste of money! The build quality is shockingly bad for the price they're charging. It feels like it's made of cheap plastic and will break any day. Performance is sluggish and unreliable. Battery life is terrible - I have to charge it multiple times per day. The camera produces blurry, washed-out photos that look terrible. The interface is clunky and unintuitive. I've contacted support multiple times but they never respond. This is the biggest regret of my life. Don't make the same mistake I did.",
            
            f"The {query} is hands down the worst product I've ever owned. I can't believe I wasted my money on this piece of junk. The construction feels cheap and poorly made. It's constantly slow, freezes up, and crashes on me. Battery dies way too fast - barely lasts half a day. Camera is completely useless, takes awful photos. The software is buggy and hard to use. Customer service is terrible and unhelpful. I wish I could get my money back. This thing is pure garbage and I hate it.",
            
            f"Terrible experience with the {query}. This product is a complete failure. The quality is awful - it feels cheap and poorly constructed. Performance is consistently bad, everything runs slow and crashes frequently. Battery life is pathetic, dies within hours. Camera quality is terrible, photos are blurry and dark. The user interface is confusing and frustrating. Support is useless and never helps. This was the worst investment I've ever made. I strongly advise against buying this garbage product.",
            
            f"I regret buying the {query} every single day. This is the worst product I've ever used. The build quality is terrible and feels like it will break any moment. Performance is awful - constantly slow, laggy, and crashes all the time. Battery life is pathetic and dies way too quickly. The camera is completely useless and produces horrible photos. The user interface is confusing and frustrating to navigate. I've had nothing but problems since day one. Customer service is terrible and never responds. This is a complete waste of money. Avoid this terrible product at all costs."
        ]
        
        # Neutral reviews (30% of total) - Truly neutral language
        neutral_templates = [
            f"I have been using the {query} for several months. The product has some features that work and some that don't work as well. The build quality is standard. The performance is what you would expect. The battery life is normal. The camera functions as described. The user interface is basic. The price is what it is. This product does what it says it will do.",
            
            f"I purchased the {query} a while ago. It has both advantages and disadvantages. Some features work while others need improvement. The build quality is standard. The performance meets basic requirements. The battery life is typical. The camera produces standard results. The user interface is basic. The price is market standard. It's a product that exists.",
            
            f"I own the {query} and have used it for some time. It has features that work and features that could be better. The build quality is what you would expect. The performance is standard. The battery life is normal for this type of product. The camera works as advertised. The user interface is basic. The price is typical for the market. It's a standard product.",
            
            f"The {query} is a product that I have been using. It has some features that function well and some that function less well. The build quality is standard. The performance is what one would expect. The battery life is normal. The camera works. The user interface is basic. The price is market standard. It's a product that does what it's supposed to do."
        ]
        
        # Generate diverse mock comments with proper distribution
        comment_id = 0
        target_total = limit * 2  # Generate 2x to account for filtering
        
        # Calculate distribution: 40% positive, 35% negative, 25% neutral
        pos_count = int(target_total * 0.40)
        neg_count = int(target_total * 0.35)
        neu_count = target_total - pos_count - neg_count
        
        # Generate positive comments
        for i in range(pos_count):
            if comment_id >= target_total:
                break
            template = random.choice(positive_templates)
            variations = [
                template,
                f"Update: {template}",
                f"After 2 years of use: {template}",
                f"Long-term review: {template}",
                f"Final verdict: {template}",
                f"Quick update: {template}",
                f"6 months later: {template}",
                f"One year review: {template}",
                f"Honest review: {template}",
                f"Real user experience: {template}"
            ]
            variation = random.choice(variations)
            
            class MockComment:
                def __init__(self, text, author, score):
                    self.body = text
                    self.text = text  # alias for compatibility
                    self.author = author
                    self.score = score
                    self.upvotes = score  # alias for compatibility
                    self.id = f"mock_{comment_id}"
                    self.created_utc = time.time() - random.randint(0, 86400 * 365)
                    self.subreddit = random.choice(['technology', 'cars', 'iphone', 'android', 'gaming'])
                    self.permalink = f"/r/{self.subreddit}/comments/{self.id}/"
                    self.url = f"https://reddit.com{self.permalink}"
            
            mock_comment = MockComment(
                text=variation,
                author=f"user_{comment_id}",
                score=random.randint(5, 50)  # Higher scores for positive
            )
            mock_comments.append(mock_comment)
            comment_id += 1
        
        # Generate negative comments
        for i in range(neg_count):
            if comment_id >= target_total:
                break
            template = random.choice(negative_templates)
            variations = [
                template,
                f"Update: {template}",
                f"After 2 years of use: {template}",
                f"Long-term review: {template}",
                f"Final verdict: {template}",
                f"Quick update: {template}",
                f"6 months later: {template}",
                f"One year review: {template}",
                f"Honest review: {template}",
                f"Real user experience: {template}"
            ]
            variation = random.choice(variations)
            
            mock_comment = MockComment(
                text=variation,
                author=f"user_{comment_id}",
                score=random.randint(1, 5)  # Lower scores for negative
            )
            mock_comments.append(mock_comment)
            comment_id += 1
        
        # Generate neutral comments
        for i in range(neu_count):
            if comment_id >= target_total:
                break
            template = random.choice(neutral_templates)
            variations = [
                template,
                f"Update: {template}",
                f"After 2 years of use: {template}",
                f"Long-term review: {template}",
                f"Final verdict: {template}",
                f"Quick update: {template}",
                f"6 months later: {template}",
                f"One year review: {template}",
                f"Honest review: {template}",
                f"Real user experience: {template}"
            ]
            variation = random.choice(variations)
            
            mock_comment = MockComment(
                text=variation,
                author=f"user_{comment_id}",
                score=random.randint(2, 15)  # Medium scores for neutral
            )
            mock_comments.append(mock_comment)
            comment_id += 1
        
        # Apply quality filtering and deduplication to mock data
        seen = set()
        filtered = []
        for c in mock_comments:
            try:
                if not passes_quality(c, settings): 
                    continue
                h = _text_hash(c.body)
                if h in seen: 
                    continue
                seen.add(h)
                filtered.append(c)
            except Exception:
                continue
        
        # Convert filtered comments to dict objects
        reviews = []
        for comment in filtered[:limit]:  # Limit to requested amount
            reviews.append({
                "id": comment.id,
                "source": "mock",
                "text": comment.body,
                "created_utc": comment.created_utc,
                "permalink": comment.permalink,  # keep relative
                "url": comment.url,              # absolute
                "author": comment.author,
                "upvotes": int(comment.score),
            })
        
        logger.info(f"Generated {len(mock_comments)} raw mock comments, filtered to {len(filtered)} quality comments, returning {len(reviews)} reviews")
        return reviews
