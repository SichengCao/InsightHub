"""Scoring and analysis modules."""

import logging
import math, time
from typing import List, Dict, Any
from collections import defaultdict, Counter
from .models import Review, AspectScore, AnalysisSummary, GPTCommentAnno, RankingItem
from .constants import SearchConstants, DomainConstants, SourceQualityMultipliers, ConfidenceConfig

logger = logging.getLogger(__name__)

def _bayesian_stars(raw: float, n: int, k: float = 2.0, mu: float = 3.0) -> float:
    """Credibility-adjusted star rating (IMDb-style Bayesian average).

    Pulls low-evidence scores toward neutral (3.0) so a single glowing mention
    doesn't equal a consistently-praised entity with many data points.

    k=2 means "equivalent to 2 neutral prior observations":
      n=1, raw=5.0 → 3.67   n=3, raw=5.0 → 4.20   n=10, raw=5.0 → 4.67
      n=1, raw=1.0 → 1.67   n=3, raw=1.0 → 1.80    n=10, raw=1.0 → 1.33
    """
    return _winsorize((n * raw + k * mu) / (n + k))


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

# --- Platform-normalised engagement weight ---
def _platform_weight(upvotes: int, source: str = "reddit") -> float:
    """
    Map raw upvote/like counts to a 1–10 scale anchored to each platform's
    own typical engagement range, so Reddit and YouTube comments are
    comparable when mixed in a combined-source ranking.

    Formula: 1 + 9 * clamp(upvotes / platform_cap, 0, 1)
      Reddit  cap=50: upvotes 0→1.0, 5→1.9, 25→5.5, 50+→10.0
      YouTube cap=15: likes   0→1.0, 3→2.8, 8→5.8,  15+→10.0
    """
    caps = SearchConstants.PLATFORM_ENGAGEMENT_CAPS
    cap = caps.get(source.lower(), caps["default"])
    return 1.0 + 9.0 * min(1.0, max(0.0, upvotes / cap))


# --- Trust weight for each review/comment (legacy: operates on Review objects) ---
def _comment_weight(review, now_ts=None) -> float:
    """
    Weight = platform_normalised(upvotes) * recency_decay
    Recency halves every ~365 days: exp(-age_days/365)
    Clamped to [0.5, 10] to avoid whale domination.
    """
    up = max(0, int(getattr(review, "upvotes", getattr(review, "score", 0)) or 0))
    source = getattr(review, "source", "reddit") or "reddit"
    w_votes = _platform_weight(up, source)
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


def aggregate_generic(aspect_schema: List[str], annos: List[GPTCommentAnno], upvote_map: Dict[str, int], source_map: Dict[str, str] = None) -> tuple[float, Dict[str, float]]:
    """Aggregate generic analysis results."""
    if not annos:
        return 3.0, {}

    _source_map = source_map or {}

    # Calculate overall rating
    overall_scores = []
    aspect_scores = defaultdict(list)

    for anno in annos:
        upvotes = upvote_map.get(anno.comment_id, 1)
        source  = _source_map.get(anno.comment_id, "reddit")
        weight  = _platform_weight(upvotes, source)
        
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



def _corroboration_strength(distinct_threads: int, cross_unit: bool) -> float:
    """How strongly independent sources back an entity, in [0, 1].

    Two signals, each worth 0.6 (capped at 1.0):
      • multi-thread  — the entity surfaced in ≥2 separate discussions
      • cross-unit    — within one thread it appears in BOTH the post/transcript
                        (the OP/creator) AND at least one comment (the crowd),
                        i.e. the post's claim is echoed by responders.
    """
    corro = 0.0
    if distinct_threads >= 2:
        corro += 0.6
    if cross_unit:
        corro += 0.6
    return min(1.0, corro)


def _four_factor_confidence(
    n_mentions: int,
    sentiment_scores: list,
    sources_contributing: set,
    total_platforms_queried: int,
    sqm_values: list,
    corroboration: float = 0.0,
) -> float:
    """
    Composite confidence score = Volume × Diversity × Consistency × SourceFit,
    then lifted toward 1.0 by cross-source corroboration.
    All factors are in [0, 1]; result is in [0, 1].

    Args:
        n_mentions:             how many units mentioned this entity
        sentiment_scores:       list of raw sentiment floats for variance calc
        sources_contributing:   set of platform names (e.g. {"reddit", "youtube"})
        total_platforms_queried: how many platforms were actually searched
        sqm_values:             list of SQM values for contributing units
        corroboration:          [0,1] strength of post/comment & multi-thread agreement
    """
    cfg = ConfidenceConfig

    # Factor 1: Volume — asymptotes toward 1 as mentions grow
    volume = n_mentions / (n_mentions + cfg.VOLUME_PRIOR)

    # Factor 2: Diversity — fraction of queried platforms that contributed
    denom = max(1, total_platforms_queried)
    diversity = 0.5 + 0.5 * (len(sources_contributing) / denom)

    # Factor 3: Consistency — penalise high sentiment variance
    if len(sentiment_scores) >= 2:
        mean = sum(sentiment_scores) / len(sentiment_scores)
        variance = sum((s - mean) ** 2 for s in sentiment_scores) / len(sentiment_scores)
        std_dev = variance ** 0.5
        consistency = max(0.0, 1.0 - std_dev / cfg.CONSISTENCY_SCALE)
    else:
        consistency = 0.7  # single data point — modest default

    # Factor 4: Source fit — average SQM of contributing comments
    source_fit = sum(sqm_values) / len(sqm_values) if sqm_values else 0.5

    base = volume * diversity * consistency * source_fit

    # Corroboration lift: an entity backed by the OP *and* the crowd, or seen
    # across multiple independent threads, is more trustworthy than its raw
    # volume implies. Lifts toward 1.0 without ever exceeding it.
    corro = max(0.0, min(1.0, corroboration))
    final = base + (1.0 - base) * corro * cfg.CORROBORATION_WEIGHT
    return round(final, 4)


def _confidence_tier(score: float) -> str:
    """Map a composite confidence score to a display tier label."""
    cfg = ConfidenceConfig
    if score >= cfg.TIER_ESTABLISHED:
        return cfg.TIER_LABELS["established"]
    if score >= cfg.TIER_EMERGING:
        return cfg.TIER_LABELS["emerging"]
    if score >= cfg.TIER_MENTIONED:
        return cfg.TIER_LABELS["mentioned"]
    return cfg.TIER_LABELS["insufficient"]


def _extract_quote(text: str, entity_name: str, max_len: int = 200) -> str:
    """Extract the most relevant sentence mentioning the entity from a comment."""
    import re as _re
    # Build name variants: strip possessives, try partial (last word only for multi-word names)
    name_lower = entity_name.lower().replace("'s", "").replace("'s", "")
    variants = {name_lower}
    words = name_lower.split()
    if len(words) >= 2:
        variants.add(words[-1])          # last word  (e.g. "Nakamura" from "Ivan Nakamura")
        variants.add(words[0])           # first word (e.g. "Deskhaus" from "Deskhaus Apex Pro")
    pattern = _re.compile(
        r'|'.join(_re.escape(v) for v in sorted(variants, key=len, reverse=True)),
        _re.IGNORECASE
    )
    # Split into sentences and pick the best match
    sentences = _re.split(r'(?<=[.!?])\s+', text.strip())
    best = None
    for sent in sentences:
        if pattern.search(sent):
            # Prefer longer sentences (more context) up to max_len
            if best is None or (len(sent) > len(best) and len(sent) <= max_len * 1.5):
                best = sent
    if best:
        return (best[:max_len] + "…") if len(best) > max_len else best
    # Fallback: return the first max_len chars if entity found anywhere in text
    if pattern.search(text):
        return (text[:max_len] + "…") if len(text) > max_len else text
    return ""


def rank_entities(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = 3, query: str = "", comments: List[Dict] = None, source_map: Dict[str, str] = None, query_category: str = None, suppress_insufficient: bool = True) -> List[RankingItem]:
    """Rank entities based on mentions and scores."""
    from insighthub.core.constants import SourceQualityMultipliers
    from insighthub.services.llm import classify_query as _classify_query
    _category   = query_category or (_classify_query(query) if query else "product_ranking")
    _source_map = source_map or {}
    query_location = _extract_query_location(query)
    # How many distinct platforms were searched (for diversity factor)
    _total_platforms = len(set(_source_map.values())) if _source_map else 1

    # Build comment text lookup for quote extraction, plus thread/unit maps so
    # the scorer can detect post↔comment and multi-thread corroboration.
    comment_text_map: Dict[str, str] = {}
    thread_map: Dict[str, str] = {}
    unit_map: Dict[str, str] = {}
    if comments:
        for c in comments:
            cid = c.get("id", "") if isinstance(c, dict) else getattr(c, "id", "")
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            if cid and text:
                comment_text_map[cid] = text
            if cid:
                tid = c.get("thread_id", "") if isinstance(c, dict) else getattr(c, "thread_id", "")
                ut = c.get("unit_type", "comment") if isinstance(c, dict) else getattr(c, "unit_type", "comment")
                # Unknown thread → empty bucket, so units without thread info collapse
                # together and never masquerade as independent corroborating threads.
                thread_map[cid] = tid or ""
                unit_map[cid] = ut or "comment"

    entity_stats = defaultdict(lambda: {
        'mentions': 0,
        'primary_mentions': 0,
        'confidence_sum': 0.0,
        'overall_scores': [],
        'aspect_scores': defaultdict(list),
        'comment_ids': [],
        'original_names': set(),
        'candidate_quotes': [],  # (upvotes, quote_text)
        'sources': set(),        # which platforms contributed mentions
        'raw_sentiments': [],    # raw sentiment floats for consistency calc
        'sqm_values': [],        # SQM per contributing comment for source-fit calc
        'thread_units': defaultdict(set),  # thread_id -> set of unit_types mentioning entity
        'evidence_values': [],   # per-mention GPT evidence_strength (0-1)
    })

    # Aggregate entity data
    for anno in annos:
        upvotes    = upvote_map.get(anno.comment_id, 1)
        src        = _source_map.get(anno.comment_id, "reddit")
        sqm        = SourceQualityMultipliers.get(_category, src)
        base_weight = _platform_weight(upvotes, src) * sqm

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

            # 5. Evidence quality: a detailed review or a credible ranking pulls the
            # score more than a casual name-drop. Scales weight in [0.4, 1.2].
            ev_strength = getattr(entity, 'evidence_strength', 0.5)
            try:
                ev_strength = max(0.0, min(1.0, float(ev_strength)))
            except (TypeError, ValueError):
                ev_strength = 0.5
            effective_weight *= (0.4 + 0.8 * ev_strength)

            normalized_name = _normalize_entity_name(entity.name)
            entity_stats[normalized_name]['mentions'] += 1
            entity_stats[normalized_name]['confidence_sum'] += entity.confidence
            entity_stats[normalized_name]['overall_scores'].append((entity_sentiment, effective_weight))
            entity_stats[normalized_name]['comment_ids'].append(anno.comment_id)
            entity_stats[normalized_name]['original_names'].add(entity.name)
            entity_stats[normalized_name]['sources'].add(src)
            entity_stats[normalized_name]['raw_sentiments'].append(entity_sentiment)
            entity_stats[normalized_name]['sqm_values'].append(sqm)
            _tid = thread_map.get(anno.comment_id, "")
            _ut = unit_map.get(anno.comment_id, "comment")
            entity_stats[normalized_name]['thread_units'][_tid].add(_ut)
            entity_stats[normalized_name]['evidence_values'].append(ev_strength)
            # Collect quote candidate from the source comment
            if comment_text_map and anno.comment_id in comment_text_map:
                raw_text = comment_text_map[anno.comment_id]
                quote = _extract_quote(raw_text, entity.name)
                if quote:
                    upvotes = upvote_map.get(anno.comment_id, 0)
                    entity_stats[normalized_name]['candidate_quotes'].append((upvotes, quote))
            if is_primary:
                entity_stats[normalized_name]['primary_mentions'] += 1

            # Use entity-specific aspect scores when available; fall back to
            # comment-level aspect scores only for primary entities.
            entity_aspects = getattr(entity, 'aspect_scores', {}) or {}
            aspect_src = entity_aspects if entity_aspects else (anno.aspect_scores if is_primary else {})
            for aspect_name, score in aspect_src.items():
                entity_stats[normalized_name]['aspect_scores'][aspect_name].append((score, effective_weight))
    
    # Merge entities where one name is a prefix/subset of a longer, more specific name.
    # Two passes:
    #   Pass A — multi-word subset (e.g. "half moon" ⊂ "half moon bay golf course")
    #   Pass B — single-word brand prefix (e.g. "deskhaus" ⊂ "deskhaus apex pro")
    #            only merges when the brand word appears at the START of the longer name
    #            and the longer name has ≥ 2 words, to avoid spurious collapses.
    _names = sorted(entity_stats.keys(), key=lambda n: len(set(n.split())))
    _absorbed: dict = {}

    # How many multi-word names start with each first word — used to only merge a
    # bare single word into a longer name when that longer name is UNAMBIGUOUS
    # (exactly one candidate). This merges "cote" → "cote korean steakhouse" but
    # refuses to guess when several names share a first word.
    _first_word_counts = Counter(n.split()[0] for n in _names if len(n.split()) >= 2)

    for _short in _names:
        _sw = set(_short.split())
        _sw_list = _short.split()
        for _long in _names:
            if _long == _short or _long in _absorbed or _short in _absorbed:
                continue
            _lw = set(_long.split())
            _lw_list = _long.split()
            # Pass A: strict multi-word subset (original logic, ≥2 matching words)
            if len(_sw) >= 2 and _sw < _lw:
                _absorbed[_short] = _long
                break
            # Pass B: single-word prefix match — only when the longer name is the
            # sole candidate starting with that word (avoids "park" → guessing
            # among "park slope ramen"/"park ave diner"). ≥4 chars skips generic
            # short words like "bar"/"the"/"inn".
            if (len(_sw) == 1 and len(_lw_list) >= 2
                    and _lw_list[0] == _sw_list[0]
                    and len(_sw_list[0]) >= 4
                    and _first_word_counts[_sw_list[0]] == 1):
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
            t['candidate_quotes'].extend(s['candidate_quotes'])
            for _asp, _scores in s['aspect_scores'].items():
                t['aspect_scores'][_asp].extend(_scores)
            t['sources'].update(s['sources'])
            t['raw_sentiments'].extend(s['raw_sentiments'])
            t['sqm_values'].extend(s['sqm_values'])
            t['evidence_values'].extend(s['evidence_values'])
            for _tid, _uts in s['thread_units'].items():
                t['thread_units'][_tid].update(_uts)
            del entity_stats[_short]

    # Create ranking items
    ranking_items = []
    for normalized_name, stats in entity_stats.items():
        if stats['mentions'] >= min_mentions:
            raw_stars = _weighted_mean(stats['overall_scores']) if stats['overall_scores'] else 3.0

            # Entities that were NEVER a primary mention (only appeared in
            # comparison lists or passing context) get blended toward neutral.
            # This prevents "Course A, B, C are all great" from inflating B and C
            # to the same 5★ as the comment's primary focus (Course A).
            if stats['primary_mentions'] == 0:
                raw_stars = 0.6 * raw_stars + 0.4 * 3.0

            # Evidence quality for this entity: blend of typical and best evidence,
            # so one authoritative/detailed piece counts even amid casual mentions.
            _ev = stats['evidence_values'] or [0.5]
            evidence_quality = 0.5 * (sum(_ev) / len(_ev)) + 0.5 * max(_ev)
            # Baseline at the neutral default (0.5) so ONLY above-average evidence
            # lifts anything; neutral/missing evidence reproduces prior behavior.
            evidence_lift = max(0.0, (evidence_quality - 0.5) / 0.5)

            # Apply Bayesian credibility adjustment so a single glowing mention
            # doesn't appear equal to an entity praised across many reviews — but
            # let strong evidence (a detailed review or a credible ranking) act
            # like extra observations so authoritative single mentions resist
            # being shrunk back toward neutral.
            effective_n = stats['mentions'] + ConfidenceConfig.EVIDENCE_PRIOR * evidence_lift
            overall_stars = _bayesian_stars(raw_stars, effective_n)

            aspect_scores = {}
            for aspect_name, scores in stats['aspect_scores'].items():
                aspect_scores[aspect_name] = _weighted_mean(scores) if scores else 3.0

            confidence = stats['confidence_sum'] / stats['mentions'] if stats['mentions'] > 0 else 0.0

            original_names = list(stats['original_names'])
            display_name = max(original_names, key=len) if original_names else normalized_name

            # Pick top-upvoted unique quotes (deduplicate by first 60 chars)
            seen_q: set = set()
            top_quotes: list = []
            for _, q in sorted(stats['candidate_quotes'], key=lambda x: -x[0]):
                key = q[:60]
                if key not in seen_q:
                    seen_q.add(key)
                    top_quotes.append(q)
                if len(top_quotes) >= 3:
                    break

            # Volume-based label (legacy, kept for backward compat)
            n = stats['mentions']
            if n >= 10:
                data_confidence = "high"
            elif n >= 4:
                data_confidence = "medium"
            elif n >= 2:
                data_confidence = "low"
            else:
                data_confidence = "very_low"

            # Corroboration: ≥2 distinct threads, and/or a single thread where the
            # post/transcript (OP/creator) and a comment (crowd) both name the entity.
            _thread_units = stats['thread_units']
            distinct_threads = len(_thread_units)
            cross_unit = any(
                ({"post", "transcript"} & uts) and ("comment" in uts)
                for uts in _thread_units.values()
            )
            corroboration = _corroboration_strength(distinct_threads, cross_unit)

            # An authoritative/detailed single mention is as trustworthy as
            # multi-source corroboration, so let strong evidence lift confidence
            # too — this is what surfaces a Michelin/"50 Best" pick that only one
            # comment cited out of the "insufficient" tier.
            lift = max(corroboration, evidence_lift)

            # Four-factor composite confidence score (lifted by corroboration/evidence)
            conf_score = _four_factor_confidence(
                n_mentions=stats['mentions'],
                sentiment_scores=stats['raw_sentiments'],
                sources_contributing=stats['sources'],
                total_platforms_queried=_total_platforms,
                sqm_values=stats['sqm_values'],
                corroboration=lift,
            )
            tier = _confidence_tier(conf_score)

            # Suppress entities that don't clear the minimum confidence threshold.
            # Suppression can be disabled for the relaxation fallback so the
            # pipeline always returns something when data exists.
            if suppress_insufficient and tier == ConfidenceConfig.TIER_LABELS["insufficient"]:
                logger.debug(f"Suppressed '{display_name}' — confidence {conf_score:.3f} below threshold")
                continue

            ranking_items.append(RankingItem(
                name=display_name,
                overall_stars=round(overall_stars, 2),
                aspect_scores=aspect_scores,
                mentions=stats['mentions'],
                confidence=confidence,
                quotes=top_quotes,
                data_confidence=data_confidence,
                confidence_score=conf_score,
                confidence_tier=tier,
            ))

    # Sort by credibility-adjusted score descending, then by mentions descending
    ranking_items.sort(key=lambda x: (-x.overall_stars, -x.mentions))
    
    return ranking_items

def rank_entities_with_relaxation(annos: List[GPTCommentAnno], upvote_map: Dict[str, int], entity_type: str, min_mentions: int = SearchConstants.DEFAULT_MIN_MENTIONS, query: str = "", comments: List[Dict] = None, source_map: Dict[str, str] = None, query_category: str = None) -> List[RankingItem]:
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
    floor = SearchConstants.MIN_RANKED_RESULTS
    thresholds = sorted({min_mentions, max(1, min_mentions // 2), 1}, reverse=True)
    ranked = []
    for threshold in thresholds:
        logger.info(f"Entity ranking attempt: min_mentions={threshold}")
        ranked = rank_entities(annos, upvote_map, entity_type, threshold, query=query, comments=comments, source_map=source_map, query_category=query_category)
        if len(ranked) >= floor:
            logger.info(f"Entity ranking: {len(ranked)} confident entities at min_mentions={threshold}")
            return ranked

    # The confident ranking is sparser than the floor (or empty).  Do a final
    # pass with suppression disabled so the pipeline surfaces the long tail when
    # data exists.  Entities that only clear this relaxed pass carry
    # tier="insufficient" so the UI can show them under a separate low-confidence
    # "also mentioned" section rather than mixing them with the primary picks.
    full = rank_entities(annos, upvote_map, entity_type, 1, query=query,
                         comments=comments, source_map=source_map,
                         query_category=query_category,
                         suppress_insufficient=False)
    if len(full) > len(ranked):
        ranked = full
        logger.info(f"Entity ranking: {len(ranked)} entities (sparse rescue, suppression off)")

    logger.info(f"Entity ranking: {len(ranked)} entities after full relaxation")
    return ranked
