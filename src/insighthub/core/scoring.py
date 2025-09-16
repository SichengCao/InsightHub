"""Scoring and analysis modules."""

import logging
import math, time
from typing import List, Dict, Any
from collections import defaultdict
from .models import Review, AspectScore, AnalysisSummary, GPTCommentAnno, RankingItem

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


def rank_entities(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = 3) -> List[RankingItem]:
    """Rank entities based on mentions and scores."""
    entity_stats = defaultdict(lambda: {
        'mentions': 0,
        'confidence_sum': 0.0,
        'overall_scores': [],
        'aspect_scores': defaultdict(list),
        'comment_ids': []
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
                entity.entity_type in entity_type):  # reverse substring match
                entity_stats[entity.name]['mentions'] += 1
                entity_stats[entity.name]['confidence_sum'] += entity.confidence
                entity_stats[entity.name]['overall_scores'].append((anno.overall_score, weight))
                entity_stats[entity.name]['comment_ids'].append(anno.comment_id)
                
                # Add aspect scores
                for aspect_name, score in anno.aspect_scores.items():
                    entity_stats[entity.name]['aspect_scores'][aspect_name].append((score, weight))
    
    # Create ranking items
    ranking_items = []
    for entity_name, stats in entity_stats.items():
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
            
            ranking_item = RankingItem(
                name=entity_name,
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
