"""Configuration management for InsightHub."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
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


def ensure_config_files():
    """Ensure config files exist with sensible defaults."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    aspects_dir = config_dir / "aspects"
    aspects_dir.mkdir(exist_ok=True)
    
    # Create tech_products.yaml if missing
    tech_aspects_file = aspects_dir / "tech_products.yaml"
    if not tech_aspects_file.exists():
        tech_aspects = {
            "battery": ["battery", "battery life", "charge", "charging", "power"],
            "camera": ["camera", "photo", "photo quality", "video"],
            "durability": ["durable", "build", "scratch", "drop", "water", "IP68"],
            "price": ["price", "expensive", "cheap", "value"],
            "heat": ["hot", "overheat", "warm", "heat"],
            "ai": ["ai", "siri", "assistant", "generative", "model", "ml"],
            "other": ["design", "screen", "display", "audio", "speaker", "microphone"]
        }
        with open(tech_aspects_file, 'w') as f:
            yaml.dump(tech_aspects, f, default_flow_style=False)
    
    # Create weights.yaml if missing
    weights_file = config_dir / "weights.yaml"
    if not weights_file.exists():
        weights = {
            "global": {
                "sentiment_weight": 1.0,
                "aspect_weight": 1.0
            }
        }
        with open(weights_file, 'w') as f:
            yaml.dump(weights, f, default_flow_style=False)


# Global settings instance
settings = Settings()
