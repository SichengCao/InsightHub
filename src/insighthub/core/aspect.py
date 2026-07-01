"""Neutral aspect fallbacks.

Domain-specific aspect selection is performed by GPT (see
``OpenAIService.detect_intent_and_schema`` / ``generate_dynamic_aspects``).
This module intentionally contains NO keyword-based domain routing — the
helpers below are neutral last-resort fallbacks used only when a GPT call
is unavailable or fails. All real domain decisions go through the model.
"""

from typing import Dict, List


# Neutral, domain-agnostic aspect set used only as a fallback when GPT
# aspect generation is unavailable. These names apply to almost any product
# or service and carry no domain assumptions.
GENERAL_ASPECTS = {
    "Quality": ["quality", "build", "construction", "materials", "durable", "reliable"],
    "Performance": ["performance", "speed", "fast", "slow", "power", "efficiency"],
    "Design": ["design", "looks", "appearance", "style", "aesthetic", "beautiful"],
    "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth", "affordable"],
    "User Experience": ["easy", "difficult", "interface", "usability", "comfortable", "convenient"],
    "Features": ["features", "functionality", "capabilities", "options", "specs"],
    "Support": ["support", "service", "warranty", "help", "customer service"],
    "Overall": ["overall", "general", "summary", "verdict", "recommendation"],
}


def get_domain_aspects(query: str) -> Dict[str, List[str]]:
    """Neutral fallback aspect set.

    GPT decides the real aspects for a query; this is only reached when the
    model path fails. It returns a generic, domain-agnostic set rather than
    routing on hardcoded keywords.
    """
    return GENERAL_ASPECTS


def aspect_hint_for_query(query: str) -> str:
    """A neutral, domain-agnostic instruction for the summarizer.

    Instead of matching keywords to pick a domain, we ask the model to infer
    the domain from the query and the comments themselves and to choose
    aspects that fit that domain. This keeps all domain reasoning inside GPT.
    """
    return (
        "Infer the product, service, or venue category from the query and the "
        "comments themselves, then organize insights around the aspects that "
        "actually matter for that category. Use only aspects that the comments "
        "genuinely discuss; do not import aspects from unrelated categories."
    )
