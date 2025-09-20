"""Domain-specific aspect taxonomies for GPT hints."""

from typing import Dict, List


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