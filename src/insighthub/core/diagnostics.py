"""Pipeline observability helpers.

`entity_flow_table` traces each entity through the funnel so you can tell WHY a
place is missing from the ranking: it was never retrieved, retrieved but the
relevance filter cut its threads, extracted but suppressed by scoring, etc.

Stages (per entity):
  retrieved  — raw scraped units whose text mentions the entity (pre-filter)
  filtered   — surviving units (post relevance filter) whose text mentions it
  extracted  — annotations where GPT actually pulled it out as an entity (conf>=0.5)
  primary    — of those, how many as the primary/focus entity
  outcome    — ranked | also-mentioned | dropped (suppressed by scoring)
"""
from __future__ import annotations
import re
from typing import List, Dict, Any

from .scoring import _normalize_entity_name


def _text_of(u) -> str:
    t = u.get("text", "") if isinstance(u, dict) else getattr(u, "text", "")
    return (t or "").lower()


def _mention_hits(units, needle: str) -> int:
    """How many units' text contains the (normalized) entity name."""
    if not needle:
        return 0
    pat = re.compile(re.escape(needle), re.IGNORECASE)
    return sum(1 for u in units if pat.search(_text_of(u)))


def entity_flow_table(reviews, comments, annos, ranked=None, also_mentioned=None) -> List[Dict[str, Any]]:
    """Return per-entity funnel rows, sorted by extracted count (desc).

    Args:
        reviews:        raw scraped units (pre relevance filter) — dicts or objects
        comments:       units that survived the relevance filter
        annos:          List[GPTCommentAnno] produced from `comments`
        ranked:         final ranked RankingItems (confident tier)
        also_mentioned: low-confidence "also mentioned" RankingItems
    """
    ranked = ranked or []
    also_mentioned = also_mentioned or []

    # Vocabulary + extraction counts come from what GPT actually extracted.
    vocab: Dict[str, str] = {}          # normalized -> best display name
    extracted: Dict[str, int] = {}
    primary: Dict[str, int] = {}
    for a in annos:
        for e in getattr(a, "entities", []) or []:
            if getattr(e, "confidence", 0.0) < 0.5:
                continue
            norm = _normalize_entity_name(e.name)
            if not norm:
                continue
            if norm not in vocab or len(e.name) > len(vocab[norm]):
                vocab[norm] = e.name
            extracted[norm] = extracted.get(norm, 0) + 1
            if getattr(e, "is_primary", True):
                primary[norm] = primary.get(norm, 0) + 1

    outcome: Dict[str, tuple] = {}
    for it in ranked:
        outcome[_normalize_entity_name(it.name)] = ("ranked", it.mentions, it.overall_stars)
    for it in also_mentioned:
        outcome.setdefault(_normalize_entity_name(it.name), ("also-mentioned", it.mentions, it.overall_stars))

    rows = []
    for norm, disp in vocab.items():
        oc, om, ostars = outcome.get(norm, ("dropped", 0, 0.0))
        rows.append({
            "entity": disp,
            "retrieved": _mention_hits(reviews, norm),
            "filtered": _mention_hits(comments, norm),
            "extracted": extracted.get(norm, 0),
            "primary": primary.get(norm, 0),
            "outcome": oc,
            "ranked_mentions": om,
            "stars": round(ostars, 2) if ostars else 0.0,
        })
    rows.sort(key=lambda r: (r["outcome"] == "dropped", -r["extracted"], -r["retrieved"]))
    return rows
