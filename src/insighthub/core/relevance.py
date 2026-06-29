"""
Fast token-overlap relevance pre-filter.

Runs before the GPT relevance call to cheaply eliminate comments whose
text has almost no overlap with the user query's meaningful vocabulary.

Typical examples caught here:
  - "arm injury" thread mentioning iPhone in passing
  - Privacy policy discussion with one product keyword
  - Completely off-topic replies that share a keyword with the query
"""

import re
import logging
from typing import Dict, List, Tuple

from .constants import RelevanceFilterConfig

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Lower-case, split on non-alpha, strip stop words."""
    cfg = RelevanceFilterConfig
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in cfg.STOP_WORDS and len(t) >= 2]


def extract_query_terms(query: str) -> List[str]:
    """
    Extract the meaningful vocabulary from the user query.

    Deduplicates and preserves order so multi-word phrases that appear as
    a single token (e.g. "iphone16") still match.
    """
    tokens = _tokenize(query)
    seen, terms = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def score_comment(query_terms: List[str], comment_text: str) -> float:
    """
    Return a relevance score in [0, 1] for a single comment.

    Score = matched_terms / len(query_terms)

    Partial credit: a term is "matched" if it appears as a substring of any
    word in the comment — handles plurals, compounds, and slight variations
    (e.g. query term "battery" matches "batteries", "battery-life").
    """
    if not query_terms:
        return 1.0  # no query terms to match → don't filter

    comment_lower = comment_text.lower()
    matched = sum(1 for term in query_terms if term in comment_lower)
    return matched / len(query_terms)


def pre_filter_comments(
    comments: List[Dict],
    query: str,
) -> Tuple[List[Dict], Dict]:
    """
    Apply the token-overlap pre-filter to a list of comment dicts.

    Returns:
        kept      — comments that passed the filter
        stats     — dict with filtering statistics for logging and UI
    """
    cfg = RelevanceFilterConfig
    query_terms = extract_query_terms(query)

    if not query_terms:
        logger.debug("pre_filter: no meaningful query terms — skipping pre-filter")
        return comments, {"pre_filter_applied": False, "query_terms": []}

    kept, dropped_scores = [], []

    for comment in comments:
        text = comment.get("text", "")

        # Skip pre-filter for very short comments; let GPT decide
        if len(text) < cfg.MIN_LENGTH_FOR_PRE_FILTER:
            kept.append(comment)
            continue

        score = score_comment(query_terms, text)

        if score >= cfg.PRE_FILTER_THRESHOLD:
            kept.append(comment)
        else:
            dropped_scores.append(score)
            logger.debug(
                f"pre_filter DROP  score={score:.2f}  "
                f"text={text[:80].replace(chr(10), ' ')!r}"
            )

    n_total    = len(comments)
    n_kept     = len(kept)
    n_dropped  = n_total - n_kept
    avg_drop_score = (
        round(sum(dropped_scores) / len(dropped_scores), 3)
        if dropped_scores else None
    )

    stats = {
        "pre_filter_applied":   True,
        "query_terms":          query_terms,
        "threshold":            cfg.PRE_FILTER_THRESHOLD,
        "total_input":          n_total,
        "kept":                 n_kept,
        "dropped":              n_dropped,
        "drop_rate":            round(n_dropped / n_total, 3) if n_total else 0,
        "avg_dropped_score":    avg_drop_score,
    }

    logger.info(
        f"pre_filter: {n_kept}/{n_total} kept  "
        f"(dropped {n_dropped}, drop_rate={stats['drop_rate']:.1%})  "
        f"query_terms={query_terms}"
    )
    return kept, stats
