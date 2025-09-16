"""InsightHub - AI-powered Reddit review analysis platform."""

__version__ = "2.0.0"
__author__ = "InsightHub Team"

from .core.models import *
from .core.config import settings
from .services.llm import LLMServiceFactory
from .services.reddit_client import RedditService

__all__ = [
    "settings",
    "LLMServiceFactory", 
    "RedditService",
]
