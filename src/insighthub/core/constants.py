"""Constants and configuration values for InsightHub."""

# Search and Filtering Constants
class SearchConstants:
    """Constants related to Reddit search and filtering."""
    
    # API Rate Limiting (Optimized for speed)
    REDDIT_API_DELAY = 0.02  # seconds between API calls (reduced from 0.05)
    REDDIT_SEARCH_LIMIT = 20  # posts per search call (reduced from 30 for faster searches)
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
    DEFAULT_SUBREDDITS_UI = 4  # default subreddit count (reduced from 6)
    
    # Search Strategy Limits (Optimized for speed)
    MAX_SEARCH_TERMS = 4  # maximum search terms per query (reduced from 8)
    MAX_SEARCH_STRATEGIES = 2  # maximum search strategies (reduced from 3)
    MAX_COMMENT_PATTERNS = 2  # maximum regex patterns (reduced from 3)
    MAX_POSTS_PER_SUBREDDIT = 15  # posts per subreddit per search (increased for better coverage)

# Prompt Constants
class PromptConstants:
    """Constants for LLM prompts and templates."""
    
    # Prompt Versions (for cache invalidation)
    PLANNER_PROMPT_VERSION = "v2.1"
    
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

    # Detection keyword patterns (used by the router, not hardcoded in prompts)
    LOCATION_PREPOSITIONS = {"in", "near", "around", "at", "of"}

    ELECTRONICS_BRANDS = {
        "iphone", "samsung", "galaxy", "pixel", "airpods", "sony", "bose",
        "ipad", "macbook", "surface", "oneplus", "xiaomi", "oppo", "apple",
        "nvidia", "amd", "intel", "rtx", "rx", "gtx",
    }

    TOOL_VS_SIGNALS = {"vs", "versus", "or", "compared", "compare", "vs."}

    SOLUTION_SIGNALS = {
        "how to", "how do i", "fix", "repair", "problem", "issue",
        "not working", "broken", "error", "crashed", "wont", "won't",
    }

    NEWS_SIGNALS = {
        "rumor", "rumours", "leak", "leaked", "upcoming", "announced",
        "release date", "launch date", "specs", "price revealed",
    }

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

    # Subreddits that should NEVER be used for review queries on consumer electronics.
    # Brand fan forums are dominated by support posts, news, and fan content.
    REVIEW_BLOCKED_SUBREDDITS: set = {
        "iPhone", "Apple", "Samsung", "SamsungGalaxy", "GooglePixel",
        "sony", "bose", "nvidia", "AMD",
    }

    # Subreddits preferred for consumer electronics review queries.
    REVIEW_PREFERRED_SUBREDDITS: list = [
        "gadgets", "technology", "hardware", "pcmasterrace",
        "headphones", "HeadphoneAdvice", "Laptops", "Laptop",
    ]


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
# YouTube Video Type Filters
# ---------------------------------------------------------------------------
class YouTubeVideoFilters:
    """
    Title keyword lists for scoring YouTube videos by content quality.
    Applied when selecting which videos to pull comments from.

    Prefer titles → score +1 each match
    Avoid titles  → score -2 each match (weighted higher because avoiding
                    clickbait / unboxing has more value than preferring reviews)

    Per-category overrides allow product rankings to favour comparison videos
    while electronics reviews favour long-form experience write-ups.
    """

    PREFER_ALWAYS: list = [
        "review", "honest", "worth it", "should you buy",
        "after", "months", "weeks", "tested", "daily driver",
        "long term", "real world", "hands on experience",
    ]

    AVOID_ALWAYS: list = [
        "unboxing", "leaked", "leaked specs", "price revealed",
        "reaction", "reacts to", "meme", "viral",
    ]

    # Category-specific prefer/avoid lists extend the base lists above.
    CATEGORY_PREFER: dict = {
        "consumer_electronics": [
            "review", "worth it", "upgrade", "compared to",
            "vs", "impressions", "after 3 months", "after 6 months",
        ],
        "product_ranking": [
            "i tested", "best", "ranked", "compared",
            "which is best", "ultimate guide", "buyer's guide",
        ],
        "tool_comparison": [
            "switched", "switching to", "i use", "daily", "workflow",
            "productivity", "coding", "developer", "honest review",
        ],
    }

    CATEGORY_AVOID: dict = {
        "consumer_electronics": ["first look", "unboxing", "setup guide"],
        "product_ranking":      ["sponsored", "gifted", "ad", "kickstarter"],
        "tool_comparison":      ["tutorial", "how to", "beginner guide"],
    }

    MIN_VIDEO_SCORE: int = 0  # videos scoring below this are excluded


# ---------------------------------------------------------------------------
# Aspect Intelligence v2
# ---------------------------------------------------------------------------
class AspectTaxonomy:
    """
    Precisely-defined aspect taxonomy for consumer electronics (phones).

    Each aspect entry has:
      description       — what this aspect covers, injected into the GPT prompt
      inclusion_signals — topic words that SHOULD map here
      exclusion_signals — topic words that should NOT map here despite surface similarity

    The descriptions replace keyword matching with semantic definitions,
    which dramatically reduces cross-contamination between aspects.
    """

    PHONE_ASPECTS: dict = {
        "Battery": {
            "description": (
                "How long the battery lasts, how fast it charges, wireless charging support, "
                "battery degradation over time, and MagSafe / charging accessories. "
                "Does NOT include screen-on time attributed to display quality."
            ),
            "inclusion_signals": [
                "battery", "battery life", "mah", "charging", "fast charge",
                "magsafe", "wireless charge", "charge speed", "overnight",
                "wired", "usb-c", "drain", "endurance", "plug in",
            ],
            "exclusion_signals": ["camera", "photo", "screen", "display"],
        },
        "Camera": {
            "description": (
                "Photo and video quality, camera hardware (main, ultra-wide, telephoto), "
                "computational photography, night mode, video stabilisation, selfie camera. "
                "Does NOT include storage for photos or app performance."
            ),
            "inclusion_signals": [
                "camera", "photo", "picture", "video", "shot", "photography",
                "zoom", "portrait", "night mode", "cinematic", "4k",
                "selfie", "lens", "megapixel", "sensor",
            ],
            "exclusion_signals": ["storage", "app", "performance", "battery"],
        },
        "Performance": {
            "description": (
                "CPU and GPU speed, app launch times, gaming performance, multitasking, "
                "thermal throttling, benchmarks, and perceived snappiness. "
                "Does NOT include camera processing speed."
            ),
            "inclusion_signals": [
                "performance", "speed", "fast", "slow", "lag", "chip",
                "processor", "a18", "a17", "bionic", "ram", "memory",
                "benchmark", "gaming", "smooth", "stutter", "throttle",
            ],
            "exclusion_signals": ["camera", "battery", "display"],
        },
        "Display": {
            "description": (
                "Screen quality, resolution, refresh rate (ProMotion), brightness, "
                "colour accuracy, OLED vs LCD, notch or Dynamic Island. "
                "Does NOT include screen size as a form-factor preference."
            ),
            "inclusion_signals": [
                "screen", "display", "oled", "promotion", "refresh rate",
                "120hz", "brightness", "nits", "resolution", "colours",
                "dynamic island", "notch", "hdr", "aod", "always on",
            ],
            "exclusion_signals": ["camera", "performance"],
        },
        "Software": {
            "description": (
                "iOS features, software updates, Siri, Apple Intelligence, "
                "app quality, bugs, UI design, notification system, privacy features. "
                "Does NOT include hardware reliability."
            ),
            "inclusion_signals": [
                "ios", "software", "update", "siri", "ai", "intelligence",
                "feature", "app", "widget", "notification", "ui", "interface",
                "privacy", "bug", "crash", "glitch",
            ],
            "exclusion_signals": ["hardware", "build", "battery"],
        },
        "AI": {
            "description": (
                "Apple Intelligence features specifically: writing tools, image generation, "
                "priority notifications, ChatGPT integration, on-device AI processing. "
                "Does NOT include general Siri performance."
            ),
            "inclusion_signals": [
                "apple intelligence", "ai feature", "writing tools",
                "image playground", "genmoji", "chatgpt", "on-device ai",
                "priority notification", "summarise", "clean up photo",
            ],
            "exclusion_signals": ["siri", "general performance"],
        },
        "Heat": {
            "description": (
                "Whether the phone gets warm or hot during use, charging, or gaming. "
                "Thermal management, heat dissipation, and comfort during extended use."
            ),
            "inclusion_signals": [
                "heat", "hot", "warm", "temperature", "thermal", "overheat",
                "burns", "uncomfortable", "cool down",
            ],
            "exclusion_signals": [],
        },
        "Durability": {
            "description": (
                "Physical toughness: drop resistance, scratch resistance, water/dust protection "
                "(IP rating), Ceramic Shield, titanium/aluminium frame longevity. "
                "Does NOT include software reliability or battery degradation."
            ),
            "inclusion_signals": [
                "durable", "drop", "scratch", "ip68", "waterproof", "water resistant",
                "ceramic shield", "titanium", "aluminium", "case", "crack",
                "break", "repair", "shatter",
            ],
            "exclusion_signals": ["battery", "software", "bug"],
        },
        "Charging": {
            "description": (
                "Charging ecosystem: cable types, adapters, charging bricks, "
                "USB-C transition, MagSafe accessories, AirPower alternatives. "
                "Separate from battery capacity — this is the charging infrastructure."
            ),
            "inclusion_signals": [
                "cable", "usb-c", "lightning", "adapter", "brick", "plug",
                "magsafe", "qi", "wireless pad", "charger", "included",
            ],
            "exclusion_signals": ["battery life", "endurance"],
        },
        "Price": {
            "description": (
                "Device purchase price, carrier pricing, trade-in value, financing, "
                "whether the model is worth the money at its price tier, "
                "comparison to cheaper Android alternatives at the same price."
            ),
            "inclusion_signals": [
                "price", "cost", "expensive", "cheap", "affordable", "worth",
                "value", "dollar", "$", "trade in", "financing", "monthly",
                "£", "aud",
            ],
            "exclusion_signals": [],
        },
        "Ecosystem": {
            "description": (
                "Apple ecosystem integration: AirDrop, Handoff, iMessage, FaceTime, "
                "Apple Watch pairing, MacBook continuity, iCloud, Family Sharing. "
                "Switching costs to/from Android."
            ),
            "inclusion_signals": [
                "ecosystem", "airdrop", "handoff", "imessage", "facetime",
                "apple watch", "macbook", "icloud", "continuity", "family sharing",
                "lock-in", "switching", "android",
            ],
            "exclusion_signals": [],
        },
        "Build Quality": {
            "description": (
                "Physical feel, materials (titanium, glass, aluminium), weight, "
                "button feel, port quality, form factor, size preferences."
            ),
            "inclusion_signals": [
                "build", "quality", "feel", "solid", "premium", "plastic",
                "glass", "titanium", "aluminium", "weight", "heavy", "light",
                "thin", "thick", "button", "haptic", "action button",
            ],
            "exclusion_signals": ["performance", "software", "camera"],
        },
        "Value": {
            "description": (
                "Overall value judgement: does the phone deliver enough improvement "
                "over the previous model to justify upgrading? Is the full package "
                "worth the price compared to alternatives? "
                "Distinct from Price (which is the raw cost)."
            ),
            "inclusion_signals": [
                "worth it", "upgrade", "worth upgrading", "should i buy",
                "value for money", "not worth", "compelling", "impressive",
                "disappointed", "underwhelming", "recommend",
            ],
            "exclusion_signals": [],
        },
    }

    # Non-phone query types fall through to the existing GENERAL_ASPECTS in aspect.py.
    # Only override for consumer_electronics category.
    APPLIES_TO_CATEGORIES: frozenset = frozenset({"consumer_electronics"})


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
