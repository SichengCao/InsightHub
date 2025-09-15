"""Aspect detection modules."""

import yaml
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class YAMLAspectDetector:
    """YAML-based aspect detector."""
    
    def __init__(self):
        self.aspects = self._load_aspects()
    
    def _load_aspects(self) -> Dict[str, List[str]]:
        """Load aspects from YAML file."""
        try:
            aspects_file = Path("config/aspects/tech_products.yaml")
            if aspects_file.exists():
                with open(aspects_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Aspects file not found, using defaults")
                return self._get_default_aspects()
        except Exception as e:
            logger.error(f"Failed to load aspects: {e}")
            return self._get_default_aspects()
    
    def _get_default_aspects(self) -> Dict[str, List[str]]:
        """Get default aspects if file loading fails."""
        return {
            "battery": ["battery", "battery life", "charge", "charging", "power"],
            "camera": ["camera", "photo", "photo quality", "video"],
            "durability": ["durable", "build", "scratch", "drop", "water", "IP68"],
            "price": ["price", "expensive", "cheap", "value"],
            "heat": ["hot", "overheat", "warm", "heat"],
            "ai": ["ai", "siri", "assistant", "generative", "model", "ml"],
            "other": ["design", "screen", "display", "audio", "speaker", "microphone"]
        }
    
    def detect_aspects(self, text: str, category: str = "tech") -> List[str]:
        """Detect aspects in text."""
        text_lower = text.lower()
        detected_aspects = []
        
        for aspect_name, keywords in self.aspects.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_aspects.append(aspect_name)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(detected_aspects))
