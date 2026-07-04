"""
Multi-Platform Fusion Engine.

Combines Reddit and YouTube annotations into a single trust-weighted result
with per-aspect confidence scores and source contribution breakdowns.

Example output per aspect:
    {
        "Battery": {
            "score": 4.4,
            "confidence": "high",
            "sample_size": 23,
            "sources": {"reddit": 0.35, "youtube": 0.65}
        }
    }
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.constants import SourceQualityMultipliers, ConfidenceConfig
from ..core.models import GPTCommentAnno

logger = logging.getLogger(__name__)


@dataclass
class AspectFusion:
    """Fused result for a single aspect."""
    aspect:        str
    score:         float          # weighted average 1–5
    confidence:    str            # "high" | "medium" | "low" | "insufficient"
    sample_size:   int            # number of comments that scored this aspect
    sources:       Dict[str, float] = field(default_factory=dict)
    # e.g. {"reddit": 0.35, "youtube": 0.65} — fraction of weighted signal


@dataclass
class FusionResult:
    """Complete fused result for one query."""
    query:          str
    overall_score:  float
    overall_confidence: str
    total_comments: int
    source_counts:  Dict[str, int]      # raw comment counts per platform
    aspect_fusion:  Dict[str, AspectFusion]


def _confidence_label(n: int) -> str:
    """Map sample size to a human-readable confidence label."""
    if n >= 15: return "high"
    if n >= 6:  return "medium"
    if n >= 2:  return "low"
    return "insufficient"


def fuse(
    annos:          List[GPTCommentAnno],
    upvote_map:     Dict[str, int],
    source_map:     Dict[str, str],
    query:          str,
    query_category: str = "product_ranking",
) -> FusionResult:
    """
    Fuse annotations from multiple platforms into a single FusionResult.

    Weighting:
        Each comment's contribution = platform_weight × SQM
        where platform_weight normalises engagement per-platform
        and SQM (Source Quality Multiplier) reflects how well the platform
        matches the query category.
    """
    from ..core.scoring import _platform_weight

    aspect_weighted_sums:   Dict[str, float] = defaultdict(float)
    aspect_weight_totals:   Dict[str, float] = defaultdict(float)
    aspect_source_weights:  Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    aspect_counts:          Dict[str, int]   = defaultdict(int)

    overall_weighted_sum   = 0.0
    overall_weight_total   = 0.0
    source_counts: Dict[str, int] = defaultdict(int)

    for anno in annos:
        src      = source_map.get(anno.comment_id, "reddit")
        upvotes  = upvote_map.get(anno.comment_id, 1)
        sqm      = SourceQualityMultipliers.get(query_category, src)
        weight   = _platform_weight(upvotes, src) * sqm

        source_counts[src] += 1

        # Overall score
        if anno.overall_score:
            overall_weighted_sum  += anno.overall_score * weight
            overall_weight_total  += weight

        # Per-aspect scores
        for asp, score in (anno.aspect_scores or {}).items():
            aspect_weighted_sums[asp]  += score * weight
            aspect_weight_totals[asp]  += weight
            aspect_source_weights[asp][src] += weight
            aspect_counts[asp]         += 1

    # Compute overall
    overall = (
        round(overall_weighted_sum / overall_weight_total, 2)
        if overall_weight_total > 0 else 3.0
    )

    # Compute per-aspect fusion
    aspect_fusion: Dict[str, AspectFusion] = {}
    for asp in aspect_weighted_sums:
        w_total = aspect_weight_totals[asp]
        asp_score = round(aspect_weighted_sums[asp] / w_total, 2) if w_total > 0 else 3.0

        # Source contribution as fraction of total weighted signal
        src_weights = aspect_source_weights[asp]
        total_src_w = sum(src_weights.values()) or 1.0
        src_fractions = {
            src: round(w / total_src_w, 3)
            for src, w in src_weights.items()
        }

        n = aspect_counts[asp]
        aspect_fusion[asp] = AspectFusion(
            aspect=asp,
            score=asp_score,
            confidence=_confidence_label(n),
            sample_size=n,
            sources=src_fractions,
        )

    total_comments = len(annos)
    overall_conf   = _confidence_label(total_comments)

    logger.info(
        f"fusion: {total_comments} comments  overall={overall}  "
        f"aspects={list(aspect_fusion.keys())}  "
        f"sources={dict(source_counts)}"
    )

    return FusionResult(
        query=query,
        overall_score=overall,
        overall_confidence=overall_conf,
        total_comments=total_comments,
        source_counts=dict(source_counts),
        aspect_fusion=aspect_fusion,
    )


def format_aspect_breakdown(fusion: FusionResult) -> List[Dict]:
    """
    Serialize FusionResult into a list of dicts suitable for the report
    and Streamlit UI.

    Each dict: { aspect, score, confidence, sample_size, sources }
    Sorted descending by sample_size so the best-evidenced aspects appear first.
    """
    rows = []
    for asp_name, af in fusion.aspect_fusion.items():
        rows.append({
            "aspect":       asp_name,
            "score":        af.score,
            "confidence":   af.confidence,
            "sample_size":  af.sample_size,
            "sources":      af.sources,
        })
    rows.sort(key=lambda r: -r["sample_size"])
    return rows
