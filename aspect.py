"""Aspect detection modules."""

import yaml
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def _norm(text: str) -> str:
    """Normalize text: NFKC, lower, collapse spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class YAMLAspectDetector:
    """YAML-based aspect detector."""
    
    def __init__(self):
        self.aspects = self._load_aspects()
    
    def _load_aspects(self) -> Dict[str, List[str]]:
        """Load aspects from YAML file."""
        try:
            # Try user config first
            user_config = Path.home() / ".insighthub" / "aspects.yaml"
            if user_config.exists():
                with open(user_config, 'r') as f:
                    return yaml.safe_load(f)
            
            # Try project config
            aspects_file = Path("config/aspects/tech_products.yaml")
            if aspects_file.exists():
                with open(aspects_file, 'r') as f:
                    return yaml.safe_load(f)
            
            logger.warning("Aspects file not found, using defaults")
            return self._get_default_aspects()
        except Exception as e:
            logger.error(f"Failed to load aspects: {e}")
            return self._get_default_aspects()
    
    def _get_default_aspects(self) -> Dict[str, List[str]]:
        """Get comprehensive default aspects."""
        return {
            "battery": ["battery", "battery life", "charge", "charging", "power", "endurance", "drain", "runtime"],
            "camera": ["camera", "photo", "photo quality", "video", "picture", "photography", "zoom", "portrait", "night mode"],
            "display": ["screen", "display", "resolution", "brightness", "colors", "oled", "lcd", "quality"],
            "performance": ["performance", "speed", "fast", "slow", "lag", "processor", "chip", "ram", "cpu", "gpu"],
            "price/value": ["price", "expensive", "cheap", "value", "worth", "affordable", "cost", "budget"],
            "durability": ["durable", "build", "scratch", "drop", "water", "IP68", "protection", "case", "sturdy"],
            "software": ["software", "ios", "android", "updates", "bugs", "interface", "apps", "system"],
            "design": ["design", "looks", "appearance", "size", "weight", "thickness", "bezels", "style", "aesthetic"]
        }
    
    def detect_aspects(self, text: str, category: str = "tech") -> List[Dict[str, any]]:
        """Detect aspects in text with word-boundary matching."""
        text_norm = _norm(text)
        detected_aspects = []
        
        for aspect_name, keywords in self.aspects.items():
            hits = []
            for keyword in keywords:
                # Use word boundary matching for better precision
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_norm):
                    hits.append(keyword)
            
            if hits:
                detected_aspects.append({
                    "aspect": aspect_name,
                    "hits": hits,
                    "count": len(hits)
                })
        
        return detected_aspects


# Domain-aware aspect taxonomies
CAR_ASPECTS = {
    "Build Quality": ["build quality", "construction", "materials", "fit", "finish", "panel gaps", "interior quality"],
    "Ride/Handling": ["ride", "handling", "steering", "suspension", "comfort", "smooth", "bumpy", "responsive"],
    "Range/Battery": ["range", "battery", "charging", "miles", "km", "efficiency", "mpge", "kwh"],
    "Performance": ["acceleration", "speed", "power", "torque", "0-60", "quarter mile", "fast", "slow"],
    "Autopilot/FSD": ["autopilot", "fsd", "self driving", "autonomous", "lane keeping", "adaptive cruise"],
    "Interior/Infotainment": ["interior", "seats", "dashboard", "screen", "infotainment", "navigation", "audio"],
    "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth", "affordable"],
    "Service/Support": ["service", "support", "warranty", "repair", "maintenance", "customer service"]
}

PHONE_ASPECTS = {
    "Camera": ["camera", "photo", "video", "picture", "photography", "zoom", "portrait", "night mode"],
    "Battery": ["battery", "battery life", "charge", "charging", "power", "endurance", "drain"],
    "Performance": ["performance", "speed", "fast", "slow", "lag", "processor", "chip", "ram"],
    "Display": ["screen", "display", "resolution", "brightness", "colors", "oled", "lcd"],
    "Design": ["design", "looks", "appearance", "size", "weight", "thickness", "bezels"],
    "Software": ["software", "ios", "android", "updates", "bugs", "interface", "apps"],
    "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth", "affordable"],
    "Durability": ["durable", "waterproof", "drop", "scratch", "protection", "case"]
}

GENERAL_ASPECTS = {
    "Quality": ["quality", "build", "construction", "materials", "durable", "reliable"],
    "Performance": ["performance", "speed", "fast", "slow", "power", "efficiency"],
    "Design": ["design", "looks", "appearance", "style", "aesthetic", "beautiful"],
    "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth", "affordable"],
    "User Experience": ["easy", "difficult", "interface", "usability", "comfortable", "convenient"],
    "Features": ["features", "functionality", "capabilities", "options", "specs"],
    "Support": ["support", "service", "warranty", "help", "customer service"],
    "Overall": ["overall", "general", "summary", "verdict", "recommendation"]
}


def get_domain_aspects(query: str) -> Dict[str, List[str]]:
    """Get domain-specific aspects based on the query."""
    query_lower = query.lower()
    
    # Automotive
    if any(term in query_lower for term in ['tesla', 'model y', 'model 3', 'model s', 'model x', 'cybertruck', 'ev', 'electric vehicle', 'car', 'automotive', 'ford', 'bmw', 'mercedes', 'audi', 'honda', 'toyota', 'nissan', 'hyundai', 'kia']):
        return CAR_ASPECTS
    
    # Phones
    elif any(term in query_lower for term in ['iphone', 'android', 'samsung', 'google pixel', 'oneplus', 'xiaomi', 'huawei', 'phone', 'smartphone', 'mobile']):
        return PHONE_ASPECTS
    
    # Default to general aspects
    else:
        return GENERAL_ASPECTS


def aspect_hint_for_query(query: str) -> str:
    """Generate a hint string for the LLM based on the query domain."""
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['tesla', 'model y', 'model 3', 'model s', 'model x', 'cybertruck', 'ev', 'electric vehicle', 'car', 'automotive', 'ford', 'bmw', 'mercedes', 'audi', 'honda', 'toyota', 'nissan', 'hyundai', 'kia']):
        return (
            "DOMAIN=Automotive. Prefer these aspects when relevant: "
            "Build Quality; Ride/Handling; Range/Battery; Performance; Autopilot/FSD; "
            "Interior/Infotainment; Price/Value; Service/Support. "
            "DO NOT mention storage capacity, transfer speeds, mobile app processing power, or phone-camera quality "
            "unless they appear verbatim in the comments as vehicle features."
        )
    elif any(term in query_lower for term in ['iphone', 'android', 'samsung', 'google pixel', 'oneplus', 'xiaomi', 'huawei', 'phone', 'smartphone', 'mobile']):
        return (
            "DOMAIN=Mobile/Phone. Prefer these aspects when relevant: "
            "Camera; Battery; Performance; Display; Design; Software; Price/Value; Durability. "
            "Focus on mobile-specific features like camera quality, battery life, and mobile performance."
        )
    else:
        return (
            "DOMAIN=General. Use appropriate aspects based on the product type: "
            "Quality; Performance; Design; Price/Value; User Experience; Features; Support; Overall. "
            "Adapt the aspects to match the specific product category being discussed."
        )
