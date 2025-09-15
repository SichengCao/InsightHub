"""Reddit data collection service."""

import logging
import time
import random
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import praw
from ..models import Review
from ..config import settings

# --- Anti-bot heuristics & text quality gates ---
import re, hashlib, math, time
from collections import Counter

STOP_USERS = {"AutoModerator"}
WORD_RE = re.compile(r"[A-Za-z]{2,}")
URL_RE  = re.compile(r"https?://\S+")
CONTACT_RE = re.compile(r"(whats?app|telegram|wechat|kik|line|viber|dm\s+me|contact\s+me|text\s+me)", re.I)
REF_RE = re.compile(r"(ref=|utm_|affid=|affiliate|promo\s*code)", re.I)
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]")  # wide range
CODEBLOCK_RE = re.compile(r"```[\s\S]*?```", re.M)

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

def _author_meta(comment):
    # Safe access without extra API calls if rate limited
    a = getattr(comment, "author", None)
    if not a:
        return 0, 0, 9999  # karma, link_karma, age_days
    karma = (getattr(a, "comment_karma", 0) or 0) + (getattr(a, "link_karma", 0) or 0)
    link_karma = getattr(a, "link_karma", 0) or 0
    created_utc = getattr(a, "created_utc", None)
    now = time.time()
    age_days = (now - created_utc) / 86400 if created_utc else 9999
    return karma, link_karma, age_days

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

def _passes_quality(comment, query, settings) -> bool:
    body = getattr(comment, "body", "") or ""
    if not body: return False
    if comment.author and str(comment.author) in STOP_USERS: 
        return False
    
    # Skip brand-new accounts
    karma, link_karma, age_days = _author_meta(comment)
    if age_days < 3 and karma < 5:
        return False
    
    if getattr(comment, "score", 0) < 1: 
        return False
    if len(body) < getattr(settings, "min_comment_len", 80): 
        return False
    if _alpha_ratio(body) < getattr(settings, "min_alpha_ratio", 0.6): 
        return False
    if body.strip().endswith("?") and len(body) < 140:
        return False
    if len(WORD_RE.findall(body)) < 8:
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
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def scrape(self, query: str, limit: int = 50) -> List[Review]:
        """Scrape Reddit for reviews."""
        if self.reddit:
            return self._scrape_real(query, limit)
        else:
            return self._scrape_mock(query, limit)
    
    def _scrape_real(self, query: str, limit: int) -> List[Review]:
        """Scrape real Reddit data."""
        raw_comments = []
        
        try:
            # Search across multiple subreddits
            subreddits = ['technology', 'cars', 'iphone', 'android', 'gaming']
            
            for subreddit_name in subreddits:
                if len(raw_comments) >= limit * 2:  # Collect more to account for filtering
                    break
                
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts
                    for submission in subreddit.search(query, limit=10):
                        if len(raw_comments) >= limit * 2:
                            break
                        
                        # Get top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:5]:  # Top 5 comments per post
                            if len(raw_comments) >= limit * 2:
                                break
                            
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                raw_comments.append(comment)
                        
                        # Polite sleep
                        time.sleep(0.2)
                
                except Exception as e:
                    logger.warning(f"Failed to scrape subreddit {subreddit_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Reddit scraping failed: {e}")
            return self._scrape_mock(query, limit)
        
        # Apply enhanced quality filtering and deduplication
        seen = set()
        shingle_seen = set()
        filtered = []
        for c in raw_comments:
            try:
                if not _passes_quality(c, query, settings):
                    continue
                # de-dupe by normalized text
                h = _text_hash(c.body)
                if h in seen:
                    continue
                seen.add(h)
                
                # Drop mass-posted copypasta using shingle hashing
                sig = _shingle_sig(c.body)
                if sig in shingle_seen:
                    continue
                shingle_seen.add(sig)
                
                filtered.append(c)
            except Exception:
                continue
        
        logger.info(f"Generated {len(raw_comments)} raw mock comments, filtered to {len(filtered)} quality comments, returning {min(limit, len(filtered))} reviews")
        
        # Convert filtered comments to Review objects with real IDs/links
        def _to_review_obj(comment):
            rid = str(getattr(comment, "id", ""))               # "k3abcd"
            author = getattr(comment, "author", None)
            author_name = getattr(author, "name", None) or (str(author) if author else "u/[deleted]")
            score = int(getattr(comment, "score", getattr(comment, "upvotes", 0)) or 0)
            permalink = getattr(comment, "permalink", "") or "" # e.g. "/r/teslamotors/comments/xyz/..."
            url = getattr(comment, "url", "") or (f"https://reddit.com{permalink}" if permalink else "")
            body = getattr(comment, "text", getattr(comment, "body", "")) or ""

            # Return a canonical dict (or hydrate your Review dataclass the same way)
            return {
                "id": rid,
                "source": "reddit",
                "text": body,
                "created_utc": getattr(comment, "created_utc", None),
                "permalink": permalink,   # keep both; UI prefers absolute url, LLM can use either
                "url": url,
                "author": author_name,
                "upvotes": score,
            }
        
        # after filtering & dedupe
        reviews = [_to_review_obj(c) for c in filtered[:limit]]
        
        logger.info(f"Scraped {len(raw_comments)} raw comments, filtered to {len(filtered)} quality comments, returning {len(reviews)} reviews")
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
            review = {
                "id": comment.id,
                "source": "mock",
                "text": comment.body,
                "created_utc": comment.created_utc,
                "permalink": f"https://reddit.com/r/reviews/comments/{comment.id}",
                "author": comment.author,
                "upvotes": comment.score,
            }
            reviews.append(review)
        
        logger.info(f"Generated {len(mock_comments)} raw mock comments, filtered to {len(filtered)} quality comments, returning {len(reviews)} reviews")
        return reviews
