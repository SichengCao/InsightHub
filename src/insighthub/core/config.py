"""Configuration management for InsightHub."""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Reddit API
    reddit_client_id: str = Field("", description="Reddit client ID")
    reddit_client_secret: str = Field("", description="Reddit client secret")
    reddit_user_agent: str = Field("InsightHub/1.0", description="Reddit user agent")
    
    # OpenAI API
    openai_api_key: str = Field("", description="OpenAI API key")
    OPENAI_API_KEY: str = Field("", description="OpenAI API key (alternative naming)")
    
    @property
    def effective_openai_key(self) -> str:
        """Get the effective OpenAI API key from either field."""
        return self.openai_api_key or self.OPENAI_API_KEY
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    
    # Analysis settings
    default_limit: int = Field(50, description="Default number of reviews to analyze")
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, description="Base retry delay in seconds")
    retry_backoff: float = Field(2.0, description="Retry backoff multiplier")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
