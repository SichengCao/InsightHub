"""Scoring and analysis modules."""

import logging
import math, time
from typing import List, Dict, Any
from collections import defaultdict
from .models import Review, AspectScore, AnalysisSummary, GPTCommentAnno, RankingItem
from .constants import SearchConstants, DomainConstants

logger = logging.getLogger(__name__)

def _winsorize(x: float, lo=1.0, hi=5.0) -> float:
    """Clamp value to range [lo, hi]."""
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 3.0

def _winsor_weight(w: int, cap: int = 50) -> int:
    """Winsorize weight to prevent extreme values from dominating."""
    try:
        return max(1, min(cap, int(w)))
    except Exception:
        return 1

def _weighted_mean(pairs):  # [(score, weight)]
    """Compute weighted average of score-weight pairs."""
    num = sum(s*w for s, w in pairs)
    den = sum(w for _, w in pairs) or 1.0
    return _winsorize(num / den)

def aspect_aggregate(reviews):
    """Aggregate aspect scores with winsorized weighted averaging."""
    agg = defaultdict(list)
    for r in reviews:
        w = _winsor_weight(int(getattr(r, "upvotes", getattr(r, "score", 0)) or 0))
        for a in getattr(r, "aspect_scores", []) or []:
            agg[a.aspect].append((_winsorize(getattr(a, "score", 3.0)), w))
    return [{"name": k, "score": round(_weighted_mean(v),1), "count": len(v)} for k, v in agg.items()]


def compute_global(reviews_with_stars: List[Dict[str, Any]]) -> float:
    """Compute global average stars via arithmetic mean of per-review stars."""
    if not reviews_with_stars:
        return 3.0
    
    stars_list = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    for review_data in reviews_with_stars:
        stars = review_data.get('stars', 3.0)
        label = review_data.get('label', 'NEUTRAL')
        
        # Ensure stars is valid
        if stars is None or stars < 1.0 or stars > 5.0:
            stars = 3.0
        
        stars_list.append(stars)
        
        # Count sentiment labels
        if label == 'POSITIVE':
            pos_count += 1
        elif label == 'NEGATIVE':
            neg_count += 1
        else:
            neu_count += 1
    
    total_reviews = len(stars_list)
    
    # Calculate arithmetic mean of per-comment stars
    average_rating = sum(stars_list) / total_reviews
    
    # Calculate baseline for guardrail
    baseline = 1.0 + 4.0 * ((pos_count + 0.5 * neu_count) / total_reviews)
    
    # Apply guardrail if needed
    if abs(average_rating - baseline) > 1.0:
        # Blend toward baseline to damp anomalies
        average_rating = 0.6 * average_rating + 0.4 * baseline
    
    # Clamp to [1, 5]
    average_rating = max(1.0, min(5.0, average_rating))
    
    return average_rating


def compute_aspect_scores(reviews_by_aspect: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """Compute per-aspect stars."""
    aspect_scores = {}
    
    for aspect, reviews in reviews_by_aspect.items():
        if not reviews:
            continue
        
        aspect_stars = []
        for review_data in reviews:
            stars = review_data.get('stars', 3.0)
            if stars is not None:
                aspect_stars.append(max(1.0, min(5.0, stars)))
        
        if aspect_stars:
            aspect_scores[aspect] = sum(aspect_stars) / len(aspect_stars)
    
    return aspect_scores


def create_analysis_summary(
    query: str,
    reviews: List[Review],
    sentiment_results: List[Dict[str, Any]],
    aspect_scores: Dict[str, float]
) -> AnalysisSummary:
    """Create analysis summary."""
    total = len(reviews)
    pos = sum(1 for r in sentiment_results if r.get('label') == 'POSITIVE')
    neg = sum(1 for r in sentiment_results if r.get('label') == 'NEGATIVE')
    neu = total - pos - neg
    
    # Compute global average
    reviews_with_stars = [
        {'stars': r.get('stars', 3.0), 'label': r.get('label', 'NEUTRAL')}
        for r in sentiment_results
    ]
    avg_stars = compute_global(reviews_with_stars)
    
    return AnalysisSummary(
        query=query,
        total=total,
        pos=pos,
        neg=neg,
        neu=neu,
        avg_stars=avg_stars,
        aspect_averages=aspect_scores
    )

# --- Trust weight for each review/comment ---
def _comment_weight(review, now_ts=None) -> float:
    """
    Weight = sqrt(upvotes+1) * recency_decay
    Recency halves every ~365 days: exp(-age_days/365)
    Clamped to [0.5, 10] to avoid whale domination.
    """
    up = max(0, int(getattr(review, "upvotes", getattr(review, "score", 0)) or 0))
    w_votes = math.sqrt(up + 1)
    created_utc = getattr(review, "created_utc", None)
    if now_ts is None:
        now_ts = time.time()
    age_days = (now_ts - created_utc) / 86400 if created_utc else 30
    w_time = math.exp(-age_days / 365.0)
    w = w_votes * w_time
    return max(0.5, min(10.0, w))

# --- From a 1-5 star into pos/neu/neg labels ---
def _label_from_stars(stars: float, pos=3.6, neg=2.4) -> str:
    if stars is None:
        return "NEU"
    s = float(stars)
    if s >= pos: return "POS"
    if s <= neg: return "NEG"
    return "NEU"

# --- Wilson lower bound for a positive-rate estimate (with weights) ---
def _wilson_lower_bound(pos_w: float, n_eff: float, z: float = 1.96) -> float:
    """
    pos_w: positive "weight" (sum of weights of positive samples)
    n_eff: total effective weight (pos + neg + optionally a slice of neu)
    Returns lower bound of the proportion in [0,1].
    """
    if n_eff <= 0:
        return 0.0
    phat = pos_w / n_eff
    denom = 1 + z*z/n_eff
    center = phat + z*z/(2*n_eff)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n_eff))/n_eff)
    return max(0.0, (center - margin) / denom)

# --- Convert a wilson positive rate to a 1-5 star estimate ---
def _posrate_to_stars(p: float) -> float:
    return 1.0 + 4.0 * max(0.0, min(1.0, p))

def overall_rating_wilson(reviews) -> float:
    """
    Compute overall star rating using trust-weighted Wilson LB of positive rate.
    Treat NEU as 0.5 positive (weak evidence).
    """
    pos_w = neg_w = neu_w = 0.0
    for r in reviews:
        stars = getattr(r, "stars", None)
        if stars is None:
            continue
        w = _comment_weight(r)
        label = _label_from_stars(stars)
        if label == "POS": pos_w += w
        elif label == "NEG": neg_w += w
        else: neu_w += w
    n_eff = pos_w + neg_w + 0.5 * neu_w
    p_lb = _wilson_lower_bound(pos_w + 0.5 * neu_w, n_eff)  # slight prior from NEU
    return round(_posrate_to_stars(p_lb), 1)

def aspect_rating_wilson(reviews) -> list[dict]:
    """
    For each aspect across reviews, aggregate trust-weighted Wilson LB and output stars.
    Each review is expected to have `aspect_scores: List[AspectScore(aspect, score)]`.
    """
    bucket = defaultdict(lambda: {"pos":0.0, "neg":0.0, "neu":0.0})
    for r in reviews:
        w = _comment_weight(r)
        for a in getattr(r, "aspect_scores", []) or []:
            label = _label_from_stars(getattr(a, "stars", None))
            if label == "POS": bucket[a.name]["pos"] += w
            elif label == "NEG": bucket[a.name]["neg"] += w
            else: bucket[a.name]["neu"] += w
    out = []
    for aspect, d in bucket.items():
        n_eff = d["pos"] + d["neg"] + 0.5 * d["neu"]
        p_lb = _wilson_lower_bound(d["pos"] + 0.5 * d["neu"], n_eff) if n_eff > 0 else 0.0
        stars = round(_posrate_to_stars(p_lb), 1)
        count = int(d["pos"] + d["neg"] + d["neu"])  # effective weight (approx)
        out.append({"name": aspect, "score": stars, "count": count})
    # Sort by count desc then score desc
    out.sort(key=lambda x: (-(x["count"]), -x["score"]))
    return out

def crowd_trust_stars(reviews):
    """Trust-weighted crowd score using Wilson lower bound."""
    pos_w = neg_w = neu_w = 0.0
    for r in reviews:
        s = float(r.get("overall_rating", 0) or 0)
        if s <= 0: continue
        w = _comment_weight(r)
        if s >= 3.6: pos_w += w
        elif s <= 2.4: neg_w += w
        else: neu_w += w
    n_eff = pos_w + neg_w + 0.5*neu_w
    p_lb = _wilson_lower_bound(pos_w + 0.5*neu_w, n_eff)
    return round(1 + 4*p_lb, 1)


def aggregate_generic(aspect_schema: List[str], annos: List[GPTCommentAnno], upvote_map: Dict[str, int]) -> tuple[float, Dict[str, float]]:
    """Aggregate generic analysis results."""
    if not annos:
        return 3.0, {}
    
    # Calculate overall rating
    overall_scores = []
    aspect_scores = defaultdict(list)
    
    for anno in annos:
        # Weight by upvotes
        weight = _winsor_weight(upvote_map.get(anno.comment_id, 1))
        
        # Overall score
        overall_scores.append((anno.overall_score, weight))
        
        # Aspect scores
        for aspect_name in aspect_schema:
            if aspect_name in anno.aspect_scores:
                aspect_scores[aspect_name].append((anno.aspect_scores[aspect_name], weight))
    
    # Calculate weighted averages
    overall = _weighted_mean(overall_scores) if overall_scores else 3.0
    
    aspect_averages = {}
    for aspect_name in aspect_schema:
        if aspect_scores[aspect_name]:
            aspect_averages[aspect_name] = _weighted_mean(aspect_scores[aspect_name])
        else:
            aspect_averages[aspect_name] = 3.0
    
    return overall, aspect_averages


_GEOGRAPHIC_ENTITY_TYPES = {
    "location", "city", "region", "area", "country", "state",
    "neighborhood", "district", "metro", "borough", "county",
    "province", "territory", "continent",
}

def _normalize_entity_name(name: str) -> str:
    """Normalize entity name for deduplication: lowercase, strip punctuation."""
    import re
    name = name.strip().lower()
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()

def _is_valid_entity_name(name: str, entity_type: str) -> bool:
    """Validate entity name: filter obvious non-entities before scoring."""
    name = name.strip()
    if len(name) < 2:
        return False
    if name.lower() in DomainConstants.GEO_REGION_FILTER:
        logger.debug(f"Filtered generic placeholder name: '{name}'")
        return False
    return True


def _extract_query_location(query: str) -> str:
    """
    Pull the location phrase out of the query so we can filter entities
    that belong to a different place — without maintaining a hardcoded city list.

    Examples:
      "top 10 golf course in bay area"  → "bay area"
      "best restaurants in NYC"         → "nyc"
      "Tesla Model Y review"            → ""   (no location)
    """
    import re
    # Match everything after a location preposition
    m = re.search(
        r"\b(?:in|near|around|at|for)\s+(.+)$",
        query.strip(),
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().lower()
    return ""



def rank_entities(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = 3, query: str = "") -> List[RankingItem]:
    """Rank entities based on mentions and scores."""
    query_location = _extract_query_location(query)
    entity_stats = defaultdict(lambda: {
        'mentions': 0,
        'primary_mentions': 0,
        'confidence_sum': 0.0,
        'overall_scores': [],
        'aspect_scores': defaultdict(list),
        'comment_ids': [],
        'original_names': set(),
    })
    
    # Aggregate entity data
    for anno in annos:
        base_weight = _winsor_weight(upvote_map.get(anno.comment_id, 1))

        for entity in anno.entities:
            # Drop low-confidence extractions
            if entity.confidence < 0.5:
                continue

            # Drop entities GPT itself classified as geographic types (domain-agnostic: uses GPT's own type field)
            if (entity.entity_type or "").lower() in _GEOGRAPHIC_ENTITY_TYPES:
                continue

            # Drop the entity if its name IS the query location (e.g. "Bay Area" in a bay area query)
            if query_location and entity.name.strip().lower() == query_location:
                continue

            # Drop generic placeholder names (e.g. "the area", "the city")
            if not _is_valid_entity_name(entity.name, entity_type):
                continue

            # ── Scoring ──────────────────────────────────────────────────────
            # Use entity-specific sentiment_score (new field).
            # Fall back to anno.overall_score only for primary entities; non-primary
            # entities without an explicit score get a neutral 3.0 to avoid polluting
            # rankings with misattributed comment-level sentiment.
            is_primary = getattr(entity, 'is_primary', True)
            entity_sentiment = getattr(entity, 'sentiment_score', None)
            if entity_sentiment is None or entity_sentiment == 3.0:
                entity_sentiment = anno.overall_score if is_primary else 3.0

            # 4. Non-primary entities (comparison/context mentions) get 0.75× weight
            # so they still contribute but don't dominate the ranking.
            effective_weight = base_weight if is_primary else base_weight * 0.75

            normalized_name = _normalize_entity_name(entity.name)
            entity_stats[normalized_name]['mentions'] += 1
            entity_stats[normalized_name]['confidence_sum'] += entity.confidence
            entity_stats[normalized_name]['overall_scores'].append((entity_sentiment, effective_weight))
            entity_stats[normalized_name]['comment_ids'].append(anno.comment_id)
            entity_stats[normalized_name]['original_names'].add(entity.name)
            if is_primary:
                entity_stats[normalized_name]['primary_mentions'] += 1

            # Use entity-specific aspect scores when available; fall back to
            # comment-level aspect scores only for primary entities.
            entity_aspects = getattr(entity, 'aspect_scores', {}) or {}
            aspect_src = entity_aspects if entity_aspects else (anno.aspect_scores if is_primary else {})
            for aspect_name, score in aspect_src.items():
                entity_stats[normalized_name]['aspect_scores'][aspect_name].append((score, effective_weight))
    
    # Merge entities where one name's words are a strict subset of another's
    # (e.g. "half moon" collapses into "half moon bay"). Requires ≥2 matching words
    # so single-word overlaps like "park" don't cause spurious merges.
    _names = sorted(entity_stats.keys(), key=lambda n: len(set(n.split())))
    _absorbed = {}
    for _short in _names:
        _sw = set(_short.split())
        if len(_sw) < 2:
            continue
        for _long in _names:
            if _long == _short or _long in _absorbed:
                continue
            if _sw < set(_long.split()):
                _absorbed[_short] = _long
                break
    for _short, _long in _absorbed.items():
        if _short in entity_stats and _long in entity_stats:
            s, t = entity_stats[_short], entity_stats[_long]
            t['mentions'] += s['mentions']
            t['primary_mentions'] += s['primary_mentions']
            t['confidence_sum'] += s['confidence_sum']
            t['overall_scores'].extend(s['overall_scores'])
            t['comment_ids'].extend(s['comment_ids'])
            t['original_names'].update(s['original_names'])
            for _asp, _scores in s['aspect_scores'].items():
                t['aspect_scores'][_asp].extend(_scores)
            del entity_stats[_short]

    # Create ranking items
    ranking_items = []
    for normalized_name, stats in entity_stats.items():
        if stats['mentions'] >= min_mentions:
            overall_stars = _weighted_mean(stats['overall_scores']) if stats['overall_scores'] else 3.0

            # Entities that were NEVER a primary mention (only appeared in
            # comparison lists or passing context) get blended toward neutral.
            # This prevents "Course A, B, C are all great" from inflating B and C
            # to the same 5★ as the comment's primary focus (Course A).
            if stats['primary_mentions'] == 0:
                overall_stars = 0.6 * overall_stars + 0.4 * 3.0

            aspect_scores = {}
            for aspect_name, scores in stats['aspect_scores'].items():
                aspect_scores[aspect_name] = _weighted_mean(scores) if scores else 3.0

            confidence = stats['confidence_sum'] / stats['mentions'] if stats['mentions'] > 0 else 0.0

            original_names = list(stats['original_names'])
            display_name = max(original_names, key=len) if original_names else normalized_name

            ranking_items.append(RankingItem(
                name=display_name,
                overall_stars=round(overall_stars, 2),
                aspect_scores=aspect_scores,
                mentions=stats['mentions'],
                confidence=confidence,
                quotes=[],
            ))
    
    # Sort by overall score descending, then by mentions descending
    ranking_items.sort(key=lambda x: (-x.overall_stars, -x.mentions))
    
    return ranking_items

def rank_entities_with_relaxation(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = SearchConstants.DEFAULT_MIN_MENTIONS, query: str = "") -> List[RankingItem]:
    """
    Rank entities with automatic relaxation if too few results.
    
    This method implements progressive relaxation to ensure we get sufficient results:
    1. Start with strict evidence requirements (min_mentions=2)
    2. If < 3 entities found, relax to min_mentions=1
    3. If still insufficient, return whatever was found
    
    This adaptive approach balances quality (high evidence) with quantity (sufficient results).
    
    Args:
        annos: List of GPT annotations with extracted entities
        upvote_map: Mapping of comment IDs to upvote counts for weighting
        entity_type: Expected entity type for filtering
        min_mentions: Starting minimum mentions per entity
        
    Returns:
        List of RankingItem objects sorted by overall score and mentions
    """
    max_retries = 3
    current_min_mentions = min_mentions
    
    for attempt in range(max_retries):
        logger.info(f"Entity ranking attempt {attempt + 1}: min_mentions={current_min_mentions}")
        
        ranked = rank_entities(annos, upvote_map, entity_type, current_min_mentions, query=query)
        
        if len(ranked) >= 3:
            logger.info(f"Entity ranking: {len(ranked)} entities ranked with min_mentions={current_min_mentions}")
            return ranked
        
        if attempt < max_retries - 1:
            # Relax constraints for next attempt
            current_min_mentions = max(1, current_min_mentions - 1)
            logger.info(f"Auto-relaxing: reducing min_mentions to {current_min_mentions}")
    
    logger.info(f"Entity ranking: {len(ranked)} entities ranked after {max_retries} attempts")
    return ranked
