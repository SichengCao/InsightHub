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
    """LLM-powered aspect detector with fallback to predefined aspects."""
    
    def __init__(self):
        self.aspects = self._load_aspects()
        self.llm_service = None
        self._aspect_cache = {}  # Cache generated aspects per query
    
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
    
    def detect_aspects(self, text: str, query: str = "", sample_comments: list = None, category: str = "tech") -> List[Dict[str, any]]:
        """Detect aspects in text using LLM-generated aspects with fallback."""
        text_norm = _norm(text)
        detected_aspects = []
        
        # Get aspects (LLM-generated or fallback)
        if query and sample_comments:
            domain_aspects = self._get_generated_aspects(query, sample_comments)
        elif query:
            domain_aspects = get_domain_aspects(query)
        else:
            domain_aspects = self.aspects
        
        for aspect_name, keywords in domain_aspects.items():
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
    
    def _get_llm_service(self):
        """Get LLM service instance."""
        if self.llm_service is None:
            try:
                from llm import LLMServiceFactory
                self.llm_service = LLMServiceFactory.create()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM service: {e}")
                return None
        return self.llm_service
    
    def _get_generated_aspects(self, query: str, sample_comments: list) -> Dict[str, List[str]]:
        """Get LLM-generated aspects for a query."""
        # Check cache first
        cache_key = f"{query.lower().strip()}"
        if cache_key in self._aspect_cache:
            return self._aspect_cache[cache_key]
        
        # Try LLM generation
        llm_service = self._get_llm_service()
        if llm_service:
            try:
                generated_aspects = llm_service.generate_aspects_for_query(query, sample_comments)
                if generated_aspects:
                    self._aspect_cache[cache_key] = generated_aspects
                    logger.info(f"Generated {len(generated_aspects)} aspects for query: {query}")
                    return generated_aspects
            except Exception as e:
                logger.warning(f"LLM aspect generation failed: {e}")
        
        # Fallback to predefined aspects
        fallback_aspects = get_domain_aspects(query)
        self._aspect_cache[cache_key] = fallback_aspects
        logger.info(f"Using fallback aspects for query: {query}")
        return fallback_aspects


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

GOLF_ASPECTS = {
    "Course Quality": ["course", "greens", "fairways", "rough", "condition", "maintenance", "quality", "well-maintained"],
    "Difficulty/Challenge": ["difficult", "challenging", "easy", "hard", "slope", "rating", "handicap", "skill level"],
    "Pace of Play": ["pace", "slow", "fast", "wait", "backup", "crowded", "busy", "time", "speed"],
    "Scenery/Views": ["scenic", "beautiful", "views", "landscape", "nature", "mountains", "ocean", "picturesque"],
    "Facilities": ["clubhouse", "pro shop", "restaurant", "bar", "locker room", "practice", "driving range", "putting green"],
    "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth", "affordable", "green fees", "membership"],
    "Staff/Service": ["staff", "service", "friendly", "helpful", "professional", "rude", "attitude", "customer service"],
    "Accessibility": ["public", "private", "accessible", "location", "parking", "easy to find", "convenient"]
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
    
    # Golf courses
    if any(term in query_lower for term in ['golf', 'golf course', 'golfing', 'course', 'greens', 'fairway', 'tee', 'putting', 'golf club', 'country club']):
        return GOLF_ASPECTS
    
    # Automotive
    elif any(term in query_lower for term in ['tesla', 'model y', 'model 3', 'model s', 'model x', 'cybertruck', 'ev', 'electric vehicle', 'car', 'automotive', 'ford', 'bmw', 'mercedes', 'audi', 'honda', 'toyota', 'nissan', 'hyundai', 'kia']):
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
    
    if any(term in query_lower for term in ['golf', 'golf course', 'golfing', 'course', 'greens', 'fairway', 'tee', 'putting', 'golf club', 'country club']):
        return (
            "DOMAIN=Golf Course. Prefer these aspects when relevant: "
            "Course Quality; Difficulty/Challenge; Pace of Play; Scenery/Views; Facilities; "
            "Price/Value; Staff/Service; Accessibility. "
            "Focus on golf-specific features like course condition, greens quality, pace of play, and facilities."
        )
    elif any(term in query_lower for term in ['tesla', 'model y', 'model 3', 'model s', 'model x', 'cybertruck', 'ev', 'electric vehicle', 'car', 'automotive', 'ford', 'bmw', 'mercedes', 'audi', 'honda', 'toyota', 'nissan', 'hyundai', 'kia']):
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
