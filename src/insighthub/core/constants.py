"""Constants and configuration values for InsightHub."""

# Search and Filtering Constants
class SearchConstants:
    """Constants related to Reddit search and filtering."""
    
    # API Rate Limiting (Optimized for speed)
    REDDIT_API_DELAY = 0.02  # seconds between API calls (reduced from 0.05)
    REDDIT_SEARCH_LIMIT = 30  # posts per search call — recall first; the GPT relevance filter prunes downstream
    REDDIT_MAX_RAW_COMMENTS = 3  # multiplier for comment collection (reduced from 4 for early termination)
    
    # Quality Filtering
    MIN_COMMENT_LENGTH = 30  # minimum characters in comment
    MIN_COMMENT_SCORE = 1  # minimum Reddit score
    MIN_ALPHA_RATIO = 0.6  # minimum alphabetic character ratio
    MIN_WORD_COUNT = 8  # minimum words in comment
    MIN_ACCOUNT_AGE_DAYS = 3  # minimum account age
    MIN_ACCOUNT_KARMA = 5  # minimum account karma
    
    # Text Processing
    MAX_COMMENT_LENGTH_FOR_ANNOTATION = 300  # chars for GPT processing
    MAX_QUOTE_LENGTH = 150  # chars for quote display
    MAX_TEXT_HASH_LENGTH = 50  # chars for text hashing
    
    # Batch Processing
    LLM_BATCH_SIZE = 12  # comments per batch for GPT
    LLM_MAX_TOKENS = 800  # max tokens per GPT response
    LLM_TEMPERATURE = 0.2  # GPT temperature for consistency
    
    # Platform engagement normalization caps.
    # Upvotes/likes are normalized per-platform so Reddit (typical: 1–500)
    # and YouTube (typical: 0–15) land on the same 1–10 weight scale before
    # being merged.  Values are the 95th-percentile engagement for review-type
    # comments; anything above the cap gets the maximum weight of 10.
    PLATFORM_ENGAGEMENT_CAPS: dict = {
        "reddit":  50,
        "youtube": 15,
        "default": 50,
    }

    # Entity Ranking
    DEFAULT_MIN_MENTIONS = 2  # minimum mentions for entity ranking
    # When the confident ranking is sparser than this, surface the long tail
    # under a low-confidence "also mentioned" treatment instead of hiding it.
    # Tunable while we evaluate across queries.
    MIN_RANKED_RESULTS = 5
    MAX_ENTITIES_TO_DISPLAY = 10  # top entities to show (increased from 5)
    MAX_ENTITIES_FOR_SUMMARY = 10  # entities to include in ranking summary
    MAX_ENTITIES_FOR_QUOTES = 3  # top entities for detailed quotes in summary
    MAX_QUOTES_PER_ENTITY = 3  # quotes per entity
    
    # UI Limits
    MAX_COMMENTS_UI = 500  # maximum comments user can request
    MIN_COMMENTS_UI = 10   # minimum comments user can request
    DEFAULT_COMMENTS_UI = 100  # default comment count
    
    MAX_SUBREDDITS_UI = 8  # maximum subreddits user can request (reduced from 12)
    MIN_SUBREDDITS_UI = 2  # minimum subreddits user can request (reduced from 3)
    DEFAULT_SUBREDDITS_UI = 6  # default subreddit count — broader community coverage by default
    
    # Search Strategy Limits (Optimized for speed)
    MAX_SEARCH_TERMS = 6  # maximum search terms per query — broader coverage, filtered downstream
    MAX_SEARCH_STRATEGIES = 2  # maximum search strategies (reduced from 3)
    MAX_COMMENT_PATTERNS = 2  # maximum regex patterns (reduced from 3)
    MAX_POSTS_PER_SUBREDDIT = 15  # posts per subreddit per search (increased for better coverage)

# Prompt Constants
class PromptConstants:
    """Constants for LLM prompts and templates."""
    
    # Prompt Versions (for cache invalidation)
    PLANNER_PROMPT_VERSION = "v2.2"  # recall-first: 3-6 subreddits, min_comment_score 0-2, more terms
    
    # Response Limits
    MAX_ASPECTS_PER_QUERY = 6  # maximum aspects to extract
    MAX_SOLUTION_CLUSTERS = 5  # maximum solution clusters
    
    # Summary Limits
    MAX_SUMMARY_TOKENS = 600  # max tokens for summary generation
    SUMMARY_TEMPERATURE = 0.4  # temperature for creative summaries

# Cache Constants
class CacheConstants:
    """Constants for caching behavior."""
    
    CACHE_TTL_HOURS = 24  # cache time-to-live in hours
    CACHE_KEY_LENGTH = 8  # length of cache key for logging

# Mock Data Constants
class MockDataConstants:
    """Constants for mock data generation."""
    
    MOCK_COMMENT_COUNT = 10  # number of mock comments to generate
    POSITIVE_MOCK_RATIO = 0.4  # ratio of positive mock comments
    NEGATIVE_MOCK_RATIO = 0.3  # ratio of negative mock comments
    NEUTRAL_MOCK_RATIO = 0.3  # ratio of neutral mock comments

# Error Handling Constants
class ErrorConstants:
    """Constants for error handling and retries."""
    
    MAX_RETRY_ATTEMPTS = 3  # maximum retry attempts
    RETRY_BASE_DELAY = 2  # base delay for exponential backoff
    REQUEST_TIMEOUT = 60  # timeout for API requests

# File and Path Constants
class FileConstants:
    """Constants for file operations."""
    
    CACHE_DIR = "cache/llm_cache"  # cache directory
    IMAGE_CACHE_DIR = "cache/image_cache"  # entity image_url enrichment cache
    CONFIG_FILE = ".env.example"  # configuration template file
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ---------------------------------------------------------------------------
# Query Category Routing
# ---------------------------------------------------------------------------
class QueryCategory:
    """
    Taxonomy of query types and their platform routing rules.

    Each category defines:
      platforms        - ordered list of platforms to query
      reddit_weight    - fraction of total weight given to Reddit signal
      youtube_weight   - fraction of total weight given to YouTube signal
      reddit_subreddit_policy - "review_only" | "local" | "dedicated" | "any"
      youtube_video_policy    - "review" | "comparison" | "any" | "none"
      min_corpus       - minimum filtered comments before showing results
    """

    # NOTE: query→category is decided by GPT (OpenAIService.classify_query_category).
    # No keyword detection lists live here. The table below is neutral numeric
    # routing config keyed by the GPT-chosen category.

    # Routing table: category_name → config dict
    ROUTING_TABLE: dict = {
        "local_discovery": {
            "platforms":               ["reddit", "youtube"],
            "reddit_weight":           0.90,
            "youtube_weight":          0.10,
            "reddit_subreddit_policy": "local",
            "youtube_video_policy":    "any",
            "min_corpus":              10,
        },
        "consumer_electronics": {
            "platforms":               ["youtube", "reddit"],
            "reddit_weight":           0.20,
            "youtube_weight":          0.80,
            "reddit_subreddit_policy": "review_only",
            "youtube_video_policy":    "review",
            "min_corpus":              8,
        },
        "product_ranking": {
            "platforms":               ["reddit", "youtube"],
            "reddit_weight":           0.50,
            "youtube_weight":          0.50,
            "reddit_subreddit_policy": "dedicated",
            "youtube_video_policy":    "comparison",
            "min_corpus":              8,
        },
        "tool_comparison": {
            "platforms":               ["reddit"],
            "reddit_weight":           1.00,
            "youtube_weight":          0.00,
            "reddit_subreddit_policy": "any",
            "youtube_video_policy":    "none",
            "min_corpus":              8,
        },
        "troubleshooting": {
            "platforms":               ["reddit"],
            "reddit_weight":           1.00,
            "youtube_weight":          0.00,
            "reddit_subreddit_policy": "any",
            "youtube_video_policy":    "none",
            "min_corpus":              5,
        },
        "unsupported": {
            "platforms":               [],
            "reddit_weight":           0.00,
            "youtube_weight":          0.00,
            "reddit_subreddit_policy": "none",
            "youtube_video_policy":    "none",
            "min_corpus":              0,
        },
        "service_review": {
            "platforms":               ["reddit", "youtube"],
            "reddit_weight":           0.60,
            "youtube_weight":          0.40,
            "reddit_subreddit_policy": "any",
            "youtube_video_policy":    "review",
            "min_corpus":              8,
        },
    }

    # Subreddit selection is decided by GPT in the search-planner prompt — no
    # hardcoded block/prefer lists here.


# ---------------------------------------------------------------------------
# Source Quality Multipliers (SQM)
# ---------------------------------------------------------------------------
class SourceQualityMultipliers:
    """
    Per-category affinity score for each platform.
    Applied as a multiplier on top of the platform-normalised engagement weight.
    Values are fractions in [0, 1]; higher = stronger signal for that category.
    """

    TABLE: dict = {
        #                          reddit   youtube
        "local_discovery":       {"reddit": 0.90, "youtube": 0.20},
        "consumer_electronics":  {"reddit": 0.20, "youtube": 0.90},
        "product_ranking":       {"reddit": 0.60, "youtube": 0.70},
        "tool_comparison":       {"reddit": 0.60, "youtube": 0.30},
        "troubleshooting":       {"reddit": 0.90, "youtube": 0.10},
        "service_review":        {"reddit": 0.70, "youtube": 0.50},
        "unsupported":           {"reddit": 0.00, "youtube": 0.00},
    }

    DEFAULT: dict = {"reddit": 0.60, "youtube": 0.50}

    @classmethod
    def get(cls, category: str, platform: str) -> float:
        row = cls.TABLE.get(category, cls.DEFAULT)
        return row.get(platform.lower(), 0.5)


# ---------------------------------------------------------------------------
# Confidence Scoring Framework
# ---------------------------------------------------------------------------
class ConfidenceConfig:
    """
    Thresholds and weights for the four-factor entity confidence score.

    confidence = Volume × Diversity × Consistency × SourceFit

    Tiers control how a RankingItem is displayed in the UI:
      ESTABLISHED  → full stars + aspect breakdown + quotes
      EMERGING     → stars + "limited data" note + quotes
      MENTIONED    → name + top quote only, no stars
      INSUFFICIENT → suppressed entirely (not shown)
    """

    # Volume factor: V = n / (n + VOLUME_PRIOR)
    # VOLUME_PRIOR is the "equivalent neutral observations" needed before
    # volume contributes meaningfully.  At n=VOLUME_PRIOR the factor is 0.5.
    VOLUME_PRIOR: int = 5

    # Consistency factor: C = max(0, 1 - std_dev / CONSISTENCY_SCALE)
    # A std_dev equal to CONSISTENCY_SCALE maps to C=0 (maximally inconsistent).
    CONSISTENCY_SCALE: float = 2.0

    # Corroboration lift: final = base + (1-base) * corroboration * WEIGHT.
    # Controls how much post↔comment and multi-thread agreement can raise an
    # entity's confidence above what its raw volume alone would give.
    CORROBORATION_WEIGHT: float = 0.6

    # Evidence prior: a high-quality piece of evidence (a detailed review or a
    # credible ranking) counts like up to this many extra observations when
    # shrinking the star score toward neutral. Lets a single authoritative
    # mention resist Bayesian shrinkage instead of collapsing to ~3.0.
    EVIDENCE_PRIOR: float = 3.0

    # Confidence tier thresholds (inclusive lower bounds)
    TIER_ESTABLISHED:  float = 0.60
    TIER_EMERGING:     float = 0.35
    TIER_MENTIONED:    float = 0.15
    # Below TIER_MENTIONED → INSUFFICIENT → suppressed

    TIER_LABELS: dict = {
        "established":  "established",
        "emerging":     "emerging",
        "mentioned":    "mentioned",
        "insufficient": "insufficient",
    }


# ---------------------------------------------------------------------------
# Query Relevance Pre-Filter
# ---------------------------------------------------------------------------
class RelevanceFilterConfig:
    """
    Configuration for the fast token-overlap pre-filter that runs before
    the expensive GPT relevance call.

    The pre-filter scores each comment by how much of the query's meaningful
    vocabulary appears in the comment text.  Comments below PRE_FILTER_THRESHOLD
    are dropped without calling GPT — saving cost and latency.

    Score formula:
        score = matched_query_terms / total_query_terms
    where query terms have stop-words removed and are stemmed to root form.

    A score of 0.15 means at least 15 % of query terms must appear somewhere
    in the comment.  For a 4-term query like "best ramen in NYC" (after stop
    words: ["ramen", "nyc"]) this means at least 1 of the 2 meaningful terms
    must appear.
    """

    PRE_FILTER_THRESHOLD: float = 0.15
    # Require at least this many query terms to be present (absolute floor)
    PRE_FILTER_MIN_TERMS_MATCHED: int = 1

    # Words that are stripped from the query before computing overlap.
    # Keeping this list in constants (not code) makes it easy to extend.
    STOP_WORDS: frozenset = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "best", "top", "good", "great", "better", "worst", "worst",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "what", "which", "who", "how", "when",
        "where", "why", "this", "that", "these", "those",
    })

    # Minimum comment length (chars) before pre-filter is applied.
    # Very short comments skip the filter and go straight to GPT.
    MIN_LENGTH_FOR_PRE_FILTER: int = 40


# Domain-Specific Constants
class DomainConstants:
    """Constants for domain-specific logic."""

    # Generic article-led phrases that are never valid entity names.
    # Location-specific filtering is handled dynamically via _extract_query_location().
    GEO_REGION_FILTER = {
        "the area", "the region", "the city", "the bay", "the valley",
        "the town", "the neighborhood", "the district", "the zone",
    }
