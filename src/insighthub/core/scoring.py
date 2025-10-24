"""Scoring and analysis modules."""

import logging
import math, time
from typing import List, Dict, Any
from collections import defaultdict
from .models import Review, AspectScore, AnalysisSummary, GPTCommentAnno, RankingItem
from .constants import SearchConstants, DomainConstants

logger = logging.getLogger(__name__)

# --- Platform bias calibration (configurable) ---
from typing import Dict
import math

# 可放到 constants 或 YAML；这里先内置，后续再抽取
PLATFORM_BIAS: Dict[str, float] = {
    "reddit": +2.00,    # 大幅增加Reddit的正偏差
    "youtube": +2.50,   # 大幅增加YouTube的正偏差
}

_BIAS_ENABLED: bool = True
_SHRINK_K: int = 40  # alpha = n/(n+k)

def set_bias_enabled(enabled: bool) -> None:
    global _BIAS_ENABLED
    _BIAS_ENABLED = bool(enabled)

def set_shrinkage_k(k: int) -> None:
    global _SHRINK_K
    _SHRINK_K = max(1, int(k))

def _shrink_factor(n: int, k: int = None) -> float:
    if n <= 0:
        return 0.0
    kk = _SHRINK_K if k is None else max(1, int(k))
    return n / (n + kk)

def apply_platform_bias(score: float, platform: str, n_reviews: int, *, k: int = None) -> float:
    """校准平台均值：s' = clip(s + alpha(n)*bias, 1..5)。仅在平台聚合后、跨平台加权前调用一次。"""
    if not _BIAS_ENABLED:
        return float(score)
    bias = PLATFORM_BIAS.get((platform or "").lower(), 0.0)
    alpha = _shrink_factor(int(n_reviews), k=k)
    adjusted = float(score) + alpha * float(bias)
    return min(5.0, max(1.0, adjusted))

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


def aggregate_generic(aspect_schema: List[str], annos: List[GPTCommentAnno], upvote_map: Dict[str, int], platform_counts: Dict[str, int] = None) -> tuple[float, Dict[str, float]]:
    """Aggregate generic analysis results with platform bias calibration."""
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
    
    # Apply platform bias calibration (new method)
    if platform_counts:
        # Calculate weighted platform bias
        total_reviews = sum(platform_counts.values())
        if total_reviews > 0:
            weighted_bias = 0.0
            for platform, count in platform_counts.items():
                weight = count / total_reviews
                bias = PLATFORM_BIAS.get(platform.lower(), 0.0)
                alpha = _shrink_factor(count)
                weighted_bias += weight * alpha * bias
            overall = min(5.0, max(1.0, overall + weighted_bias))
    
    aspect_averages = {}
    for aspect_name in aspect_schema:
        if aspect_scores[aspect_name]:
            aspect_averages[aspect_name] = _weighted_mean(aspect_scores[aspect_name])
        else:
            aspect_averages[aspect_name] = 3.0
    
    return overall, aspect_averages


def _normalize_entity_name(name: str) -> str:
    """Normalize entity names for better deduplication."""
    import re
    
    name = name.strip().lower()
    
    # Remove common punctuation and normalize apostrophes
    name = re.sub(r"[''`]", "'", name)
    name = re.sub(r"[']", "", name)  # Remove apostrophes for comparison
    
    # Common variations - specific restaurant mappings
    variations = {
        "joe's pizza": "joe's",
        "joe's": "joe's",
        "joes": "joe's",
        "joes pizza": "joe's",
        "le bernardin": "le bernardin",
        "bernardin": "le bernardin",
        "tatiana": "tatiana",
        "tatiana's": "tatiana",
        "l&b": "l&b",
        "l and b": "l&b",
        # Korean restaurant variations
        "midos": "midos",
        "mido's": "midos",
        "mido cart": "midos",
        "mido": "midos",
    }
    
    # Check exact matches first
    if name in variations:
        return variations[name]
    
    # For similar names like "Midos" vs "Mido's", normalize by removing apostrophes
    normalized = re.sub(r"'", "", name)
    
    # If we have a normalized version in variations, use it
    if normalized in variations:
        return variations[normalized]
    
    return normalized

def _is_valid_entity_name(name: str, entity_type: str) -> bool:
    """
    Basic validation for entity names - let GPT handle the intelligent filtering.
    
    Only filters out obviously invalid names (too short, empty, etc.).
    The real filtering for specific vs generic entities happens in GPT analysis.
    """
    name = name.strip()
    
    # Only filter out obviously invalid names
    if len(name) < 2:
        return False
    
    # Let GPT handle the intelligent distinction between specific entities 
    # (like "iPhone 15 Pro", "Le Bernardin") and generic descriptions
    # (like "a good camera", "michelin restaurant")
    return True

def rank_entities(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = 3) -> List[RankingItem]:
    """Rank entities based on mentions and scores."""
    entity_stats = defaultdict(lambda: {
        'mentions': 0,
        'confidence_sum': 0.0,
        'overall_scores': [],
        'aspect_scores': defaultdict(list),
        'comment_ids': [],
        'original_names': set()
    })
    
    # Aggregate entity data
    for anno in annos:
        weight = _winsor_weight(upvote_map.get(anno.comment_id, 1))
        
        for entity in anno.entities:
            # Flexible entity type matching
            if (entity.entity_type == entity_type or 
                entity.entity_type == entity_type.rstrip('s') or  # singular/plural
                entity.entity_type == entity_type + 's' or  # plural
                entity_type in entity.entity_type or  # substring match
                entity.entity_type in entity_type or  # reverse substring match
                # Special cases for location-based queries
                (entity_type == "locations" and entity.entity_type in ["golf course", "golf", "course", "restaurant", "hotel", "store", "shop", "cafe", "bar", "club", "park", "beach", "museum", "theater", "venue"]) or
                (entity_type == "restaurant" and entity.entity_type in ["restaurant", "cafe", "bar", "diner", "eatery"]) or
                (entity_type == "hotel" and entity.entity_type in ["hotel", "resort", "inn", "lodge"])):
                
                # Validate entity name (filter out generic descriptors)
                if not _is_valid_entity_name(entity.name, entity_type):
                    logger.debug(f"Filtered out generic entity name: '{entity.name}'")
                    continue
                
                # Normalize entity name for deduplication
                normalized_name = _normalize_entity_name(entity.name)
                entity_stats[normalized_name]['mentions'] += 1
                entity_stats[normalized_name]['confidence_sum'] += entity.confidence
                entity_stats[normalized_name]['overall_scores'].append((anno.overall_score, weight))
                entity_stats[normalized_name]['comment_ids'].append(anno.comment_id)
                entity_stats[normalized_name]['original_names'].add(entity.name)
                
                # Add aspect scores
                for aspect_name, score in anno.aspect_scores.items():
                    entity_stats[normalized_name]['aspect_scores'][aspect_name].append((score, weight))
    
    # Create ranking items
    ranking_items = []
    for normalized_name, stats in entity_stats.items():
        if stats['mentions'] >= min_mentions:
            # Calculate overall score
            overall_stars = _weighted_mean(stats['overall_scores']) if stats['overall_scores'] else 3.0
            
            # Calculate aspect scores
            aspect_scores = {}
            for aspect_name, scores in stats['aspect_scores'].items():
                if scores:
                    aspect_scores[aspect_name] = _weighted_mean(scores)
                else:
                    aspect_scores[aspect_name] = 3.0
            
            # Calculate confidence
            confidence = stats['confidence_sum'] / stats['mentions'] if stats['mentions'] > 0 else 0.0
            
            # Choose the best original name (prefer longer, more complete names)
            original_names = list(stats['original_names'])
            display_name = max(original_names, key=len) if original_names else normalized_name
            
            ranking_item = RankingItem(
                name=display_name,
                overall_stars=overall_stars,
                aspect_scores=aspect_scores,
                mentions=stats['mentions'],
                confidence=confidence,
                quotes=[]  # Will be filled by caller
            )
            ranking_items.append(ranking_item)
    
    # Sort by overall score descending, then by mentions descending
    ranking_items.sort(key=lambda x: (-x.overall_stars, -x.mentions))
    
    return ranking_items

def rank_entities_with_relaxation(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = SearchConstants.DEFAULT_MIN_MENTIONS) -> List[RankingItem]:
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
        
        ranked = rank_entities(annos, upvote_map, entity_type, current_min_mentions)
        
        if len(ranked) >= 3:
            logger.info(f"Entity ranking: {len(ranked)} entities ranked with min_mentions={current_min_mentions}")
            return ranked
        
        if attempt < max_retries - 1:
            # Relax constraints for next attempt
            current_min_mentions = max(1, current_min_mentions - 1)
            logger.info(f"Auto-relaxing: reducing min_mentions to {current_min_mentions}")
    
    logger.info(f"Entity ranking: {len(ranked)} entities ranked after {max_retries} attempts")
    return ranked
