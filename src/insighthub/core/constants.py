"""Constants and configuration values for InsightHub."""

# Search and Filtering Constants
class SearchConstants:
    """Constants related to Reddit search and filtering."""
    
    # API Rate Limiting (Optimized for speed)
    REDDIT_API_DELAY = 0.02  # seconds between API calls (reduced from 0.05)
    REDDIT_SEARCH_LIMIT = 30  # posts per search call (increased from 25)
    REDDIT_MAX_RAW_COMMENTS = 4  # multiplier for comment collection (reduced from 5)
    
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
    LLM_BATCH_SIZE = 15  # comments per batch for GPT
    LLM_MAX_TOKENS = 800  # max tokens per GPT response
    LLM_TEMPERATURE = 0.2  # GPT temperature for consistency
    
    # Entity Ranking
    DEFAULT_MIN_MENTIONS = 2  # minimum mentions for entity ranking
    MAX_ENTITIES_TO_DISPLAY = 10  # top entities to show (increased from 5)
    MAX_ENTITIES_FOR_SUMMARY = 10  # entities to include in ranking summary
    MAX_ENTITIES_FOR_QUOTES = 3  # top entities for detailed quotes in summary
    MAX_QUOTES_PER_ENTITY = 3  # quotes per entity
    
    # UI Limits
    MAX_COMMENTS_UI = 200  # maximum comments user can request
    MIN_COMMENTS_UI = 10  # minimum comments user can request
    DEFAULT_COMMENTS_UI = 30  # default comment count (reduced for faster results)
    
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

# Domain-Specific Constants
class DomainConstants:
    """Constants for domain-specific logic."""
    
    # Entity Type Mappings
    ENTITY_TYPE_MAPPINGS = {
        'character': 'media',
        'person': 'media', 
        'actor': 'media',
        'director': 'media',
        'movie': 'media',
        'film': 'media',
        'restaurant': 'locations',
        'location': 'locations',
        'place': 'locations'
    }
    
    # Geographic Keywords
    BAY_AREA_KEYWORDS = [
        "bay area", "san francisco", "sf", "east bay", "peninsula", 
        "south bay", "san jose", "oakland", "marin", "berkeley", 
        "palo alto", "menlo park", "silicon valley", "monterey"
    ]
    
    NYC_KEYWORDS = [
        "nyc", "new york", "manhattan", "brooklyn", "queens", 
        "bronx", "staten island", "new york city"
    ]
