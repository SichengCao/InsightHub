"""
Comprehensive quality tests for InsightHub.

Covers:
  1. Pure math helpers (winsorize, weighted_mean, wilson_lb, etc.)
  2. compute_global correctness and guardrail
  3. aggregate_generic weighting
  4. rank_entities — all filtering rules
  5. rank_entities — weighting & scoring semantics
  6. rank_entities — word-subset deduplication
  7. rank_entities_with_relaxation auto-relaxation
  8. Result integrity invariants (stars range, sorted, no dupes)

All tests are offline — no API keys required.
"""

import math
import sys
import time
from collections import defaultdict
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insighthub.core.models import GPTCommentAnno, EntityRef, RankingItem
from insighthub.core.scoring import (
    _winsorize,
    _winsor_weight,
    _weighted_mean,
    _wilson_lower_bound,
    _posrate_to_stars,
    _label_from_stars,
    _normalize_entity_name,
    _extract_query_location,
    compute_global,
    aggregate_generic,
    rank_entities,
    rank_entities_with_relaxation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build test fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_entity(
    name,
    entity_type="golf_course",
    confidence=0.9,
    sentiment_score=4.0,
    is_primary=True,
    aspect_scores=None,
):
    return EntityRef(
        name=name,
        entity_type=entity_type,
        confidence=confidence,
        sentiment_score=sentiment_score,
        is_primary=is_primary,
        aspect_scores=aspect_scores or {},
    )


def make_anno(
    comment_id,
    overall_score=3.5,
    aspect_scores=None,
    entities=None,
):
    return GPTCommentAnno(
        comment_id=comment_id,
        overall_score=overall_score,
        aspect_scores=aspect_scores or {},
        entities=entities or [],
    )


def _upvotes(*pairs):
    """Build upvote_map from (comment_id, upvotes) pairs."""
    return dict(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure math helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestWinsorize:
    def test_no_clamp_needed(self):
        assert _winsorize(3.0) == 3.0

    def test_clamp_below_min(self):
        assert _winsorize(0.0) == 1.0

    def test_clamp_above_max(self):
        assert _winsorize(6.0) == 5.0

    def test_exact_boundaries(self):
        assert _winsorize(1.0) == 1.0
        assert _winsorize(5.0) == 5.0

    def test_exception_returns_neutral(self):
        assert _winsorize("bad_value") == 3.0  # type: ignore

    def test_custom_range(self):
        assert _winsorize(0.0, lo=0.0, hi=1.0) == 0.0
        assert _winsorize(2.0, lo=0.0, hi=1.0) == 1.0


class TestWinsorWeight:
    def test_positive_normal(self):
        assert _winsor_weight(10) == 10

    def test_floor_at_one(self):
        assert _winsor_weight(0) == 1
        assert _winsor_weight(-5) == 1

    def test_cap_at_50(self):
        assert _winsor_weight(999) == 50

    def test_custom_cap(self):
        assert _winsor_weight(999, cap=20) == 20

    def test_exception_returns_one(self):
        assert _winsor_weight("oops") == 1  # type: ignore


class TestWeightedMean:
    def test_equal_weights(self):
        result = _weighted_mean([(2.0, 1), (4.0, 1)])
        assert abs(result - 3.0) < 0.001

    def test_unequal_weights(self):
        # 4.0 with weight 3, 2.0 with weight 1 → (12+2)/4 = 3.5
        result = _weighted_mean([(4.0, 3), (2.0, 1)])
        assert abs(result - 3.5) < 0.001

    def test_single_item(self):
        result = _weighted_mean([(4.5, 7)])
        assert abs(result - 4.5) < 0.001

    def test_result_is_clamped_to_1_5(self):
        result = _weighted_mean([(0.0, 1)])  # raw 0 → clamped to 1
        assert result == 1.0


class TestWilsonLowerBound:
    def test_zero_n_returns_zero(self):
        assert _wilson_lower_bound(0, 0) == 0.0

    def test_all_positive(self):
        lb = _wilson_lower_bound(100, 100)
        assert lb > 0.9, "100% positive with 100 samples should have LB > 0.9"

    def test_all_negative(self):
        lb = _wilson_lower_bound(0, 100)
        assert lb == 0.0

    def test_half_positive(self):
        lb = _wilson_lower_bound(50, 100)
        assert 0.3 < lb < 0.6, f"50% positive should give LB near 0.4, got {lb}"

    def test_output_in_0_1(self):
        for pos, n in [(10, 20), (1, 2), (99, 100), (0, 1)]:
            lb = _wilson_lower_bound(pos, n)
            assert 0.0 <= lb <= 1.0


class TestPosrateToStars:
    def test_zero_rate_is_one_star(self):
        assert _posrate_to_stars(0.0) == 1.0

    def test_full_rate_is_five_stars(self):
        assert _posrate_to_stars(1.0) == 5.0

    def test_half_rate_is_three_stars(self):
        assert abs(_posrate_to_stars(0.5) - 3.0) < 0.001

    def test_clamps_above_one(self):
        assert _posrate_to_stars(2.0) == 5.0

    def test_clamps_below_zero(self):
        assert _posrate_to_stars(-1.0) == 1.0


class TestLabelFromStars:
    def test_positive_threshold(self):
        assert _label_from_stars(3.6) == "POS"
        assert _label_from_stars(5.0) == "POS"

    def test_negative_threshold(self):
        assert _label_from_stars(2.4) == "NEG"
        assert _label_from_stars(1.0) == "NEG"

    def test_neutral_range(self):
        assert _label_from_stars(3.0) == "NEU"
        assert _label_from_stars(2.5) == "NEU"
        assert _label_from_stars(3.5) == "NEU"

    def test_none_is_neutral(self):
        assert _label_from_stars(None) == "NEU"  # type: ignore


class TestNormalizeEntityName:
    def test_lowercases(self):
        assert _normalize_entity_name("Pebble Beach") == "pebble beach"

    def test_strips_punctuation(self):
        assert _normalize_entity_name("Half Moon Bay!") == "half moon bay"

    def test_strips_apostrophes(self):
        assert _normalize_entity_name("O'Hare") == "ohare"

    def test_collapses_spaces(self):
        assert _normalize_entity_name("Golden   Gate") == "golden gate"

    def test_strips_leading_trailing(self):
        assert _normalize_entity_name("  Bandon Dunes  ") == "bandon dunes"


class TestExtractQueryLocation:
    def test_extracts_in_phrase(self):
        loc = _extract_query_location("best golf courses in bay area")
        assert loc == "bay area"

    def test_extracts_near_phrase(self):
        loc = _extract_query_location("restaurants near downtown Seattle")
        assert loc == "downtown seattle"

    def test_no_location_returns_empty(self):
        loc = _extract_query_location("Tesla Model Y review")
        assert loc == ""

    def test_case_insensitive(self):
        loc = _extract_query_location("Best Ramen In Tokyo")
        assert loc == "tokyo"


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_global
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGlobal:
    def test_empty_returns_neutral(self):
        assert compute_global([]) == 3.0

    def test_all_positive_high_score(self):
        reviews = [{"stars": 4.8, "label": "POSITIVE"}] * 20
        result = compute_global(reviews)
        assert result >= 4.0, f"Expected ≥ 4.0 for all-positive, got {result}"

    def test_all_negative_low_score(self):
        reviews = [{"stars": 1.2, "label": "NEGATIVE"}] * 20
        result = compute_global(reviews)
        assert result <= 2.5, f"Expected ≤ 2.5 for all-negative, got {result}"

    def test_guardrail_blends_extreme_mismatch(self):
        # 5 stars but all labeled NEGATIVE → average 5 vs baseline near 1 → triggers blend
        reviews = [{"stars": 5.0, "label": "NEGATIVE"}] * 30
        result = compute_global(reviews)
        # Without guardrail: 5.0. With guardrail: blended down toward ~1.0 baseline.
        assert result < 5.0, "Guardrail should reduce extreme mismatch"

    def test_result_always_clamped_to_1_5(self):
        # Bad star values (0.5 and 6.0) must still produce valid output
        reviews = [
            {"stars": 0.5, "label": "NEGATIVE"},
            {"stars": 6.0, "label": "POSITIVE"},
        ]
        result = compute_global(reviews)
        assert 1.0 <= result <= 5.0

    def test_none_stars_treated_as_neutral(self):
        reviews = [
            {"stars": None, "label": "NEUTRAL"},
            {"stars": 3.0, "label": "NEUTRAL"},
        ]
        result = compute_global(reviews)
        assert result == 3.0

    def test_mixed_returns_middle(self):
        reviews = (
            [{"stars": 5.0, "label": "POSITIVE"}] * 10
            + [{"stars": 1.0, "label": "NEGATIVE"}] * 10
        )
        result = compute_global(reviews)
        assert 2.0 <= result <= 4.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. aggregate_generic
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateGeneric:
    def test_empty_returns_neutral(self):
        overall, aspects = aggregate_generic(["quality"], [], {})
        assert overall == 3.0
        assert aspects == {}

    def test_basic_overall_average(self):
        annos = [
            make_anno("c1", overall_score=5.0),
            make_anno("c2", overall_score=3.0),
        ]
        upvotes = {"c1": 1, "c2": 1}
        overall, _ = aggregate_generic(["quality"], annos, upvotes)
        assert abs(overall - 4.0) < 0.1

    def test_higher_upvotes_dominate(self):
        annos = [
            make_anno("c1", overall_score=5.0),  # 1 upvote
            make_anno("c2", overall_score=1.0),  # 50 upvotes (capped) → heavy weight
        ]
        upvotes = {"c1": 1, "c2": 100}
        overall, _ = aggregate_generic(["quality"], annos, upvotes)
        assert overall < 3.0, "High-upvote 1-star review should pull score below 3"

    def test_aspect_missing_not_fabricated(self):
        """Aspects with no signal should be absent, not filled with a 3.0 placeholder."""
        annos = [make_anno("c1", overall_score=4.0, aspect_scores={})]
        upvotes = {"c1": 1}
        _, aspects = aggregate_generic(["quality", "price"], annos, upvotes)
        assert aspects.get("quality") is None, "Missing aspect must not be fabricated as 3.0"
        assert aspects.get("price") is None,   "Missing aspect must not be fabricated as 3.0"

    def test_aspect_weighted_correctly(self):
        annos = [
            make_anno("c1", overall_score=4.0, aspect_scores={"quality": 5.0}),
            make_anno("c2", overall_score=2.0, aspect_scores={"quality": 1.0}),
        ]
        upvotes = {"c1": 1, "c2": 1}
        _, aspects = aggregate_generic(["quality"], annos, upvotes)
        assert abs(aspects["quality"] - 3.0) < 0.1

    def test_output_clamped_to_1_5(self):
        annos = [make_anno("c1", overall_score=0.0)]
        upvotes = {"c1": 1}
        overall, _ = aggregate_generic([], annos, upvotes)
        assert 1.0 <= overall <= 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. rank_entities — filtering rules
# ─────────────────────────────────────────────────────────────────────────────

def _rank(annos, upvote_map=None, entity_type="golf_course", min_mentions=1, query=""):
    return rank_entities(
        annos,
        upvote_map or {},
        entity_type,
        min_mentions=min_mentions,
        query=query,
    )


class TestRankEntitiesFiltering:
    def test_low_confidence_excluded(self):
        """Entities with confidence < 0.5 must be dropped."""
        annos = [
            make_anno("c1", entities=[
                make_entity("Pebble Beach", confidence=0.4),  # below threshold
            ])
        ] * 5
        result = _rank(annos)
        assert all(e.name != "Pebble Beach" for e in result)

    def test_exactly_05_confidence_included(self):
        """confidence == 0.5 is the threshold — entity should be included."""
        annos = [
            make_anno("c1", entities=[make_entity("Pebble Beach", confidence=0.5)])
        ] * 5
        result = _rank(annos)
        names = [e.name for e in result]
        assert "Pebble Beach" in names

    def test_geographic_entity_type_excluded(self):
        """Entities GPT labels as 'city', 'region', etc. must be dropped."""
        for geo_type in ["city", "region", "area", "location", "country", "state"]:
            annos = [
                make_anno("c1", entities=[
                    make_entity("San Francisco", entity_type=geo_type, confidence=0.9)
                ])
            ] * 5
            result = _rank(annos, entity_type="restaurant")
            assert not any(e.name == "San Francisco" for e in result), \
                f"Geographic type '{geo_type}' should be filtered"

    def test_query_location_name_excluded(self):
        """If entity name == query location phrase, drop it."""
        annos = [
            make_anno("c1", entities=[
                make_entity("Bay Area", entity_type="golf_course", confidence=0.9)
            ])
        ] * 5
        result = _rank(annos, query="best golf courses in bay area")
        assert not any("bay area" in e.name.lower() for e in result)

    def test_single_char_name_excluded(self):
        """Names shorter than 2 chars must be dropped."""
        annos = [
            make_anno("c1", entities=[make_entity("A", confidence=0.9)])
        ] * 5
        result = _rank(annos)
        assert not any(e.name == "A" for e in result)

    def test_geo_placeholder_excluded(self):
        """Placeholder names like 'the area' must be dropped."""
        for placeholder in ["the area", "the city", "the region", "the bay"]:
            annos = [
                make_anno("c1", entities=[
                    make_entity(placeholder, entity_type="golf_course", confidence=0.9)
                ])
            ] * 5
            result = _rank(annos)
            assert not any(placeholder in e.name.lower() for e in result), \
                f"Placeholder '{placeholder}' should be filtered"

    def test_min_mentions_threshold(self):
        """Entities below min_mentions must not appear."""
        entity_names = ["Pebble Beach", "Half Moon Bay Golf", "Bandon Dunes"]
        annos = [
            make_anno("c1", entities=[make_entity(entity_names[0])]),
            make_anno("c2", entities=[make_entity(entity_names[0])]),
            make_anno("c3", entities=[make_entity(entity_names[0])]),
            make_anno("c4", entities=[make_entity(entity_names[1])]),  # 1 mention only
        ]
        result = _rank(annos, min_mentions=2)
        names = [e.name for e in result]
        assert entity_names[0] in names
        assert entity_names[1] not in names


# ─────────────────────────────────────────────────────────────────────────────
# 5. rank_entities — weighting & scoring semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestRankEntitiesScoring:
    def test_higher_sentiment_ranks_first(self):
        """Entity with higher sentiment_score should rank above lower one."""
        annos_a = [
            make_anno(f"a{i}", entities=[make_entity("Pebble Beach", sentiment_score=4.8)])
            for i in range(5)
        ]
        annos_b = [
            make_anno(f"b{i}", entities=[make_entity("Spyglass Hill", sentiment_score=2.0)])
            for i in range(5)
        ]
        result = _rank(annos_a + annos_b)
        assert len(result) >= 2
        assert result[0].name == "Pebble Beach"

    def test_mentions_tiebreak_when_same_score(self):
        """With equal scores, more mentions should rank higher."""
        annos_a = [
            make_anno(f"a{i}", entities=[make_entity("A Course", sentiment_score=4.0)])
            for i in range(6)
        ]
        annos_b = [
            make_anno(f"b{i}", entities=[make_entity("B Course", sentiment_score=4.0)])
            for i in range(3)
        ]
        result = _rank(annos_a + annos_b)
        top_names = [e.name for e in result[:2]]
        assert top_names[0] == "A Course", "More mentions should rank first on equal score"

    def test_nonprimary_entity_gets_reduced_weight(self):
        """Non-primary entity with the same sentiment should rank lower than a primary one."""
        # Primary entity mentioned 3 times at 5.0
        annos_primary = [
            make_anno(f"p{i}", entities=[
                make_entity("Star Course", sentiment_score=5.0, is_primary=True)
            ])
            for i in range(3)
        ]
        # Non-primary entity mentioned 3 times with same comment-level sentiment
        annos_nonprimary = [
            make_anno(f"n{i}", overall_score=5.0, entities=[
                make_entity("Side Course", sentiment_score=5.0, is_primary=False)
            ])
            for i in range(3)
        ]
        result = _rank(annos_primary + annos_nonprimary)
        # Both should appear; primary should rank first or equal
        names = [e.name for e in result]
        assert "Star Course" in names
        assert "Side Course" in names
        star_idx = names.index("Star Course")
        side_idx = names.index("Side Course")
        assert star_idx <= side_idx, "Primary entity should rank at least as high as non-primary"

    def test_never_primary_blended_toward_neutral(self):
        """Entity that is NEVER a primary mention gets stars blended toward 3.0."""
        annos = [
            make_anno(f"c{i}", overall_score=5.0, entities=[
                make_entity("Background Course", sentiment_score=5.0, is_primary=False)
            ])
            for i in range(4)
        ]
        result = _rank(annos)
        assert result, "Should return at least one entity"
        bg = next((e for e in result if "Background" in e.name), None)
        assert bg is not None
        assert bg.overall_stars < 5.0, f"Never-primary entity should be blended down from 5.0, got {bg.overall_stars}"

    def test_aspect_scores_aggregated(self):
        """Aspect scores should be averaged across mentions."""
        annos = [
            make_anno("c1", entities=[
                make_entity("Best Club", sentiment_score=4.0,
                            aspect_scores={"course_quality": 5.0, "value": 3.0})
            ]),
            make_anno("c2", entities=[
                make_entity("Best Club", sentiment_score=4.0,
                            aspect_scores={"course_quality": 3.0, "value": 5.0})
            ]),
        ]
        result = _rank(annos)
        assert result
        e = result[0]
        assert "course_quality" in e.aspect_scores
        assert abs(e.aspect_scores["course_quality"] - 4.0) < 0.2
        assert abs(e.aspect_scores["value"] - 4.0) < 0.2

    def test_confidence_is_average_of_mentions(self):
        """Entity confidence should be average of all mention confidences."""
        annos = [
            make_anno("c1", entities=[make_entity("Pebble Beach", confidence=0.8)]),
            make_anno("c2", entities=[make_entity("Pebble Beach", confidence=0.6)]),
        ]
        result = _rank(annos)
        assert result
        assert abs(result[0].confidence - 0.7) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 6. rank_entities — word-subset deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityDeduplication:
    def test_word_subset_absorbed(self):
        """'half moon' (2 words) should be absorbed into 'half moon bay' (superset)."""
        annos = (
            [make_anno(f"s{i}", entities=[make_entity("Half Moon")])
             for i in range(3)]
            + [make_anno(f"l{i}", entities=[make_entity("Half Moon Bay")])
               for i in range(4)]
        )
        result = _rank(annos)
        names_lower = [e.name.lower() for e in result]
        # "half moon" should be absorbed — only one entry for half moon bay
        half_moon_entries = [n for n in names_lower if "half moon" in n]
        assert len(half_moon_entries) == 1, \
            f"'Half Moon' should be merged into 'Half Moon Bay', got: {names_lower}"
        assert "half moon bay" in half_moon_entries[0]

    def test_single_word_not_absorbed(self):
        """A single-word entity should NOT trigger dedup (requires ≥2 matching words)."""
        annos = (
            [make_anno(f"s{i}", entities=[make_entity("Park")])
             for i in range(4)]
            + [make_anno(f"l{i}", entities=[make_entity("Golden Gate Park")])
               for i in range(4)]
        )
        result = _rank(annos)
        names_lower = [e.name.lower() for e in result]
        # "park" alone should not be absorbed into "golden gate park"
        assert any("golden gate park" in n for n in names_lower)
        assert any("park" == n for n in names_lower), \
            "Single-word 'park' should remain separate (not absorbed)"

    def test_display_name_is_longest_variant(self):
        """When deduplicating, the display name should be the longest original form."""
        annos = (
            [make_anno(f"a{i}", entities=[make_entity("Pebble Beach Golf")])
             for i in range(3)]
            + [make_anno(f"b{i}", entities=[make_entity("Pebble Beach Golf Links")])
               for i in range(3)]
            + [make_anno(f"c{i}", entities=[make_entity("Pebble Beach")])
               for i in range(3)]
        )
        result = _rank(annos)
        # All three are the same entity; display name should be the longest
        assert len(result) == 1
        assert result[0].name == "Pebble Beach Golf Links"

    def test_merged_mentions_count_summed(self):
        """After merge, the combined entity should have the sum of all mention counts."""
        annos = (
            [make_anno(f"s{i}", entities=[make_entity("Half Moon")]) for i in range(3)]
            + [make_anno(f"l{i}", entities=[make_entity("Half Moon Bay")]) for i in range(4)]
        )
        result = _rank(annos)
        assert result
        merged = result[0]
        assert merged.mentions == 7, \
            f"Merged entity should have 3+4=7 mentions, got {merged.mentions}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. rank_entities_with_relaxation
# ─────────────────────────────────────────────────────────────────────────────

class TestRelaxation:
    def test_returns_results_when_sufficient_data(self):
        """With ≥3 qualifying entities, should return ≥3 results."""
        annos = []
        for name in ["A Course", "B Course", "C Course", "D Course"]:
            for i in range(4):
                annos.append(make_anno(
                    f"{name}_{i}",
                    entities=[make_entity(name, sentiment_score=4.0)]
                ))
        result = rank_entities_with_relaxation(annos, {}, "golf_course", query="")
        assert len(result) >= 3

    def test_relaxes_min_mentions_when_sparse(self):
        """With only 1 mention per entity, relaxation should still return results."""
        annos = [
            make_anno("c1", entities=[make_entity("Alpha Golf", sentiment_score=4.5)]),
            make_anno("c2", entities=[make_entity("Beta Golf", sentiment_score=4.0)]),
            make_anno("c3", entities=[make_entity("Gamma Golf", sentiment_score=3.5)]),
        ]
        result = rank_entities_with_relaxation(annos, {}, "golf_course",
                                               min_mentions=3, query="")
        # With 3 relaxation steps (3→2→1), single-mention entities are eventually included
        assert len(result) >= 1, "Should return entities even with sparse data after relaxation"

    def test_never_returns_empty_when_any_entity_exists(self):
        """If any entities pass filters, relaxation should surface them."""
        annos = [make_anno("c1", entities=[make_entity("Solo Club", confidence=0.9)])]
        result = rank_entities_with_relaxation(annos, {}, "golf_course",
                                               min_mentions=5, query="")
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Result integrity invariants
# ─────────────────────────────────────────────────────────────────────────────

def _make_realistic_annos(n_entities=5, mentions_each=4):
    """Build a realistic set of annotations for invariant testing."""
    names = [f"Course {chr(65+i)}" for i in range(n_entities)]
    sentiments = [4.8, 4.2, 3.8, 3.1, 2.5][:n_entities]
    annos = []
    counter = 0
    for name, sentiment in zip(names, sentiments):
        for _ in range(mentions_each):
            annos.append(make_anno(
                f"c{counter}",
                overall_score=sentiment,
                entities=[make_entity(name, sentiment_score=sentiment)],
            ))
            counter += 1
    return annos


class TestResultInvariants:
    def setup_method(self):
        self.annos = _make_realistic_annos()
        self.result = _rank(self.annos)

    def test_all_stars_in_valid_range(self):
        for item in self.result:
            assert 1.0 <= item.overall_stars <= 5.0, \
                f"Stars {item.overall_stars} out of [1,5] for {item.name}"

    def test_all_confidence_in_valid_range(self):
        for item in self.result:
            assert 0.0 <= item.confidence <= 1.0, \
                f"Confidence {item.confidence} out of [0,1] for {item.name}"

    def test_result_sorted_by_stars_descending(self):
        stars = [e.overall_stars for e in self.result]
        assert stars == sorted(stars, reverse=True), \
            f"Result not sorted desc by stars: {stars}"

    def test_no_duplicate_entity_names(self):
        seen = set()
        for item in self.result:
            normalized = item.name.strip().lower()
            assert normalized not in seen, f"Duplicate entity: {item.name}"
            seen.add(normalized)

    def test_all_mentions_positive(self):
        for item in self.result:
            assert item.mentions > 0, f"Zero mentions for {item.name}"

    def test_result_is_list_of_ranking_items(self):
        assert isinstance(self.result, list)
        for item in self.result:
            assert isinstance(item, RankingItem)

    def test_empty_annotations_returns_empty(self):
        result = _rank([])
        assert result == []

    def test_all_low_confidence_returns_empty(self):
        annos = [
            make_anno(f"c{i}", entities=[make_entity("Bad Entity", confidence=0.1)])
            for i in range(10)
        ]
        result = _rank(annos)
        assert result == []

    def test_consistency_with_multiple_calls(self):
        """rank_entities is deterministic — same input must yield same output."""
        result1 = _rank(self.annos)
        result2 = _rank(self.annos)
        assert [e.name for e in result1] == [e.name for e in result2]
        assert [round(e.overall_stars, 2) for e in result1] == \
               [round(e.overall_stars, 2) for e in result2]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Boundary and regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryAndRegression:
    def test_entity_with_no_sentiment_score_uses_overall_for_primary(self):
        """If entity.sentiment_score is None, fall back to anno.overall_score for primary."""
        anno = make_anno("c1", overall_score=4.5, entities=[
            EntityRef(
                name="Fallback Club",
                entity_type="golf_course",
                confidence=0.9,
                sentiment_score=None,  # type: ignore
                is_primary=True,
            )
        ])
        result = _rank([anno] * 4)
        assert result
        assert result[0].overall_stars > 3.0, "Should use anno.overall_score=4.5 as fallback"

    def test_entity_with_no_sentiment_score_and_nonprimary_uses_neutral(self):
        """Non-primary entity with no sentiment gets 3.0, not anno.overall_score."""
        anno = make_anno("c1", overall_score=5.0, entities=[
            EntityRef(
                name="Context Club",
                entity_type="golf_course",
                confidence=0.9,
                sentiment_score=None,  # type: ignore
                is_primary=False,
            )
        ])
        result = _rank([anno] * 4)
        assert result
        # Score should be neutral (3.0) blended, not 5.0
        assert result[0].overall_stars < 5.0

    def test_very_high_upvotes_capped_at_50(self):
        """Upvote weight should cap at 50 regardless of actual upvote count."""
        anno_high = make_anno("c1", overall_score=5.0,
                              entities=[make_entity("Elite Club", sentiment_score=5.0)])
        anno_low = make_anno("c2", overall_score=1.0,
                             entities=[make_entity("Budget Club", sentiment_score=1.0)])
        upvotes = {"c1": 1, "c2": 999999}  # c2 has absurd upvotes but capped at 50
        annos = [anno_high] * 3 + [anno_low] * 3
        result = rank_entities(annos, upvotes, "golf_course", min_mentions=1)
        budget = next((e for e in result if "Budget" in e.name), None)
        assert budget is not None
        assert budget.overall_stars < 2.0, "Even with huge upvotes, score shouldn't exceed raw sentiment"

    def test_single_entity_many_aspects(self):
        """Entity with many aspects should correctly track all of them."""
        # Use values in 1-5 range (scores are clamped by _winsorize)
        aspects = {f"aspect_{i}": float(i + 1) for i in range(5)}
        anno = make_anno("c1", entities=[
            make_entity("Rich Club", sentiment_score=4.0, aspect_scores=aspects)
        ])
        result = _rank([anno] * 3)
        assert result
        for key, val in aspects.items():
            assert key in result[0].aspect_scores
            assert abs(result[0].aspect_scores[key] - val) < 0.1
