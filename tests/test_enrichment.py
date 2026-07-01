"""
Tests for post-as-content enrichment and cross-source corroboration.

Covers:
  1. _corroboration_strength signal math
  2. _four_factor_confidence corroboration lift stays in [0, 1] and is monotonic
  3. rank_entities: post↔comment (cross-unit) corroboration lifts an entity out
     of the "insufficient" tier, while a lonely single mention does not
  4. rank_entities: multi-thread corroboration lifts confidence
  5. Reddit scraper emits a tagged post-unit from the submission body

All tests are offline — no API keys required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insighthub.core.models import GPTCommentAnno, EntityRef
from insighthub.core.constants import ConfidenceConfig
from insighthub.core.scoring import (
    _corroboration_strength,
    _four_factor_confidence,
    rank_entities,
)


def _anno(comment_id, name, score=4.4, entity_type="restaurant", evidence=0.5):
    return GPTCommentAnno(
        comment_id=comment_id,
        overall_score=score,
        aspect_scores={},
        entities=[EntityRef(name=name, entity_type=entity_type, confidence=0.9,
                            sentiment_score=score, is_primary=True,
                            evidence_strength=evidence)],
    )


# ── 1. corroboration strength ────────────────────────────────────────────────

def test_corroboration_strength_signals():
    assert _corroboration_strength(1, False) == 0.0       # lonely
    assert _corroboration_strength(2, False) == 0.6       # multi-thread only
    assert _corroboration_strength(1, True) == 0.6        # cross-unit only
    assert _corroboration_strength(3, True) == 1.0        # both, capped at 1.0


# ── 2. confidence lift ───────────────────────────────────────────────────────

def test_corroboration_lift_bounds_and_monotonic():
    base = _four_factor_confidence(1, [4.0], {"reddit"}, 2, [1.0], corroboration=0.0)
    lifted = _four_factor_confidence(1, [4.0], {"reddit"}, 2, [1.0], corroboration=1.0)
    assert 0.0 <= base <= lifted <= 1.0
    assert lifted > base  # corroboration must raise, never lower


# ── 3. cross-unit corroboration in rank_entities ─────────────────────────────

def test_post_comment_corroboration_lifts_entity():
    insufficient = ConfidenceConfig.TIER_LABELS["insufficient"]
    annos = [_anno("post_t1", "Atomix"), _anno("c1", "Atomix")]
    comments = [
        {"id": "post_t1", "text": "Atomix is incredible", "thread_id": "t1", "unit_type": "post"},
        {"id": "c1", "text": "agree, Atomix was amazing", "thread_id": "t1", "unit_type": "comment"},
    ]
    res = rank_entities(annos, {"post_t1": 10, "c1": 8}, "restaurant", min_mentions=1,
                        comments=comments, source_map={"post_t1": "reddit", "c1": "reddit"},
                        query="best restaurant")
    assert res, "entity should survive suppression after corroboration lift"
    assert res[0].name == "Atomix"
    assert res[0].confidence_tier != insufficient


def test_lonely_single_mention_stays_insufficient():
    insufficient = ConfidenceConfig.TIER_LABELS["insufficient"]
    annos = [_anno("c9", "Lonely")]
    comments = [{"id": "c9", "text": "Lonely was ok", "thread_id": "t9", "unit_type": "comment"}]
    # suppression off so we can inspect the tier rather than getting an empty list
    res = rank_entities(annos, {"c9": 3}, "restaurant", min_mentions=1, comments=comments,
                        source_map={"c9": "reddit"}, query="best restaurant",
                        suppress_insufficient=False)
    assert res and res[0].confidence_tier == insufficient


def test_multi_thread_corroboration_beats_single_thread():
    """Same mention count, but spread across two threads should score higher."""
    # Two mentions, two distinct threads
    annos_multi = [_anno("a", "Mapo"), _anno("b", "Mapo")]
    cm_multi = [
        {"id": "a", "text": "Mapo great", "thread_id": "t1", "unit_type": "comment"},
        {"id": "b", "text": "Mapo great", "thread_id": "t2", "unit_type": "comment"},
    ]
    multi = rank_entities(annos_multi, {"a": 5, "b": 5}, "restaurant", min_mentions=1,
                          comments=cm_multi, source_map={"a": "reddit", "b": "reddit"},
                          query="best", suppress_insufficient=False)

    # Two mentions, same thread
    annos_one = [_anno("a", "Mapo"), _anno("b", "Mapo")]
    cm_one = [
        {"id": "a", "text": "Mapo great", "thread_id": "t1", "unit_type": "comment"},
        {"id": "b", "text": "Mapo great", "thread_id": "t1", "unit_type": "comment"},
    ]
    one = rank_entities(annos_one, {"a": 5, "b": 5}, "restaurant", min_mentions=1,
                        comments=cm_one, source_map={"a": "reddit", "b": "reddit"},
                        query="best", suppress_insufficient=False)

    assert multi[0].confidence_score > one[0].confidence_score


def test_missing_thread_info_no_false_corroboration():
    """Without thread_id, two comments must NOT look like two corroborating threads."""
    annos = [_anno("a", "Mapo"), _anno("b", "Mapo")]
    # no thread_id / unit_type keys at all
    comments = [{"id": "a", "text": "Mapo"}, {"id": "b", "text": "Mapo"}]
    res = rank_entities(annos, {"a": 5, "b": 5}, "restaurant", min_mentions=1,
                        comments=comments, source_map={"a": "reddit", "b": "reddit"},
                        query="best", suppress_insufficient=False)
    # Collapses to a single bucket → must NOT get the 0.6 multi-thread bump.
    # Upper bound: the same entity WITH a multi-thread lift would score higher.
    lifted = _four_factor_confidence(2, [4.4, 4.4], {"reddit"}, 1, [1.0, 1.0], corroboration=0.6)
    assert res[0].confidence_score < lifted


# ── 4b. evidence strength (authority / review quality) ───────────────────────

def test_authoritative_single_mention_surfaces():
    """A single high-evidence mention (e.g. cited in a credible ranking) should
    clear suppression and earn a non-neutral score."""
    insufficient = ConfidenceConfig.TIER_LABELS["insufficient"]
    annos = [_anno("c1", "Atomix", score=4.5, evidence=0.9)]
    comments = [{"id": "c1", "text": "#7 on 50 Best", "thread_id": "t1", "unit_type": "comment"}]
    res = rank_entities(annos, {"c1": 64}, "restaurant", min_mentions=1, comments=comments,
                        source_map={"c1": "reddit"}, query="best restaurant",
                        suppress_insufficient=False)
    assert res[0].name == "Atomix"
    assert res[0].confidence_tier != insufficient
    assert res[0].overall_stars > 3.5  # not shrunk back to neutral


def test_neutral_evidence_preserves_prior_behavior():
    """Default evidence_strength (0.5) must NOT lift — a lonely casual mention
    stays insufficient and near-neutral, identical to pre-evidence behavior."""
    insufficient = ConfidenceConfig.TIER_LABELS["insufficient"]
    annos = [_anno("c2", "Casual", score=4.5, evidence=0.5)]
    comments = [{"id": "c2", "text": "casual", "thread_id": "t2", "unit_type": "comment"}]
    res = rank_entities(annos, {"c2": 64}, "restaurant", min_mentions=1, comments=comments,
                        source_map={"c2": "reddit"}, query="best restaurant",
                        suppress_insufficient=False)
    assert res[0].confidence_tier == insufficient


def test_evidence_weights_sentiment_mean():
    """Between two entities with identical mentions, the one whose praise carries
    stronger evidence should score at least as high."""
    strong = rank_entities([_anno("a", "Strong", 4.6, evidence=0.9)], {"a": 5}, "restaurant",
                           min_mentions=1, comments=[{"id": "a", "text": "x", "thread_id": "t1", "unit_type": "comment"}],
                           source_map={"a": "reddit"}, query="best", suppress_insufficient=False)
    weak = rank_entities([_anno("b", "Weak", 4.6, evidence=0.3)], {"b": 5}, "restaurant",
                         min_mentions=1, comments=[{"id": "b", "text": "y", "thread_id": "t2", "unit_type": "comment"}],
                         source_map={"b": "reddit"}, query="best", suppress_insufficient=False)
    assert strong[0].overall_stars >= weak[0].overall_stars


# ── 4c. single-word prefix merge (the "Cote" bug) ────────────────────────────

def test_single_word_prefix_merges_when_unambiguous():
    annos = [_anno("c1", "Cote Korean Steakhouse"), _anno("c2", "Cote Korean Steakhouse"),
             _anno("c3", "Cote")]
    comments = [{"id": "c1", "text": "x"}, {"id": "c2", "text": "y"}, {"id": "c3", "text": "z"}]
    res = rank_entities(annos, {"c1": 10, "c2": 8, "c3": 5}, "restaurant", min_mentions=1,
                        comments=comments, source_map={"c1": "reddit", "c2": "reddit", "c3": "reddit"},
                        query="best", suppress_insufficient=False)
    assert len(res) == 1
    assert res[0].name == "Cote Korean Steakhouse"
    assert res[0].mentions == 3


def test_single_word_prefix_not_merged_when_ambiguous():
    annos = [_anno("p1", "Park Slope Ramen"), _anno("p2", "Park Ave Diner"), _anno("p3", "Park")]
    comments = [{"id": "p1", "text": "x"}, {"id": "p2", "text": "y"}, {"id": "p3", "text": "z"}]
    res = rank_entities(annos, {"p1": 5, "p2": 5, "p3": 5}, "restaurant", min_mentions=1,
                        comments=comments, source_map={"p1": "reddit", "p2": "reddit", "p3": "reddit"},
                        query="best", suppress_insufficient=False)
    names = sorted(r.name for r in res)
    assert names == ["Park", "Park Ave Diner", "Park Slope Ramen"]


# ── 5. Reddit scraper emits post-units ───────────────────────────────────────

def test_reddit_scraper_emits_post_unit():
    from insighthub.services.reddit_client import _SyntheticUnit, _is_support_thread
    # _SyntheticUnit must behave like a comment for the pipeline (dict access)
    u = _SyntheticUnit(id="post_x", body="Title\n\nbody text", score=12,
                       _post_id="x", _unit_type="post")
    assert u.__dict__.get("_unit_type") == "post"
    assert u.__dict__.get("body").startswith("Title")
    assert not _is_support_thread("Best Korean restaurant in NYC?")
    assert _is_support_thread("iPhone won't turn on, need help")
