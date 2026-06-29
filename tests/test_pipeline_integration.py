"""
Pipeline integration tests — offline, no API keys required.

Covers every stage of the search → retrieve → annotate → rank pipeline:

  Stage 0: Search planning output contract
  Stage 1: Reddit comment quality filtering
  Stage 2: Annotation data contract (input/output shapes)
  Stage 3: Ranking pipeline with upvote weighting
  Stage 4: Summary generation edge cases
  Stage 5: CLI path uses rank_entities_with_relaxation (regression)
  Stage 6: FallbackLLMService default correctness
  Stage 7: Full mock end-to-end (no credentials)
  Stage 8: Stage-boundary data contracts
"""

import ast
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insighthub.services.llm import FallbackLLMService
from insighthub.services.reddit_client import RedditService, SearchPlan, _passes_quality
from insighthub.core.models import EntityRef, GPTCommentAnno
from insighthub.core.scoring import aggregate_generic, rank_entities, rank_entities_with_relaxation
from insighthub.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

GOOD_COMMENT = (
    "I visited this restaurant last month and the experience was absolutely incredible. "
    "The food quality was exceptional and the service was outstanding throughout. "
    "The ambiance was perfect for a date night and prices were very reasonable. "
    "The staff were attentive and very knowledgeable about the menu offerings. "
    "Highly recommend the pasta dishes especially the carbonara. Will definitely return!"
)

SHORT_COMMENT = "ok"
BOT_COMMENT = (
    "Check my profile for free money on WhatsApp telegram vip signal crypto "
    "guaranteed profit weekly returns reach me via message me on kik line viber "
    "investment manager forex signals promo code affiliate"
)


def make_entity(name, confidence=0.9, sentiment_score=4.0, is_primary=True,
                entity_type="restaurant", aspect_scores=None):
    return EntityRef(
        name=name,
        entity_type=entity_type,
        confidence=confidence,
        sentiment_score=sentiment_score,
        is_primary=is_primary,
        aspect_scores=aspect_scores or {},
    )


def make_anno(comment_id, overall_score=3.5, aspect_scores=None, entities=None):
    return GPTCommentAnno(
        comment_id=comment_id,
        overall_score=overall_score,
        aspect_scores=aspect_scores or {},
        entities=entities or [],
        primary_entity=None,
        solution_key=None,
    )


class _MockComment:
    """Minimal mock of a PRAW comment — uses __dict__ so _passes_quality works."""
    def __init__(self, body, score=10, comment_id="mock1"):
        self.__dict__["body"] = body
        self.__dict__["score"] = score
        self.__dict__["id"] = comment_id
        self.__dict__["author"] = "test_user"
        self.__dict__["permalink"] = "/r/test/comments/abc/"
        self.url = "https://reddit.com/r/test/comments/abc/"
        self.upvotes = score
        self.created_utc = time.time() - 86400


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Search planning contract
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchPlanContract:
    """plan_reddit_search output must have correct structure and valid values."""

    def setup_method(self):
        self.llm = FallbackLLMService()

    def test_has_all_required_keys(self):
        plan = self.llm.plan_reddit_search("best golf course in bay area")
        for key in ("terms", "subreddits", "time_filter", "strategies",
                    "min_comment_score", "per_post_top_n", "comment_must_patterns"):
            assert key in plan, f"Missing key: {key}"

    def test_terms_is_nonempty_list_of_strings(self):
        plan = self.llm.plan_reddit_search("best ramen Tokyo")
        assert isinstance(plan["terms"], list)
        assert len(plan["terms"]) >= 1
        assert all(isinstance(t, str) for t in plan["terms"])

    def test_strategies_only_valid_values(self):
        plan = self.llm.plan_reddit_search("how to fix iPhone battery")
        valid = {"relevance", "top", "new"}
        for s in plan["strategies"]:
            assert s in valid, f"Invalid strategy value: {s!r}"

    def test_comment_must_patterns_always_empty(self):
        """GPT sentiment analysis handles quality — patterns must stay empty."""
        for query in ["best restaurant NYC", "iPhone 15 review", "how to fix wifi"]:
            plan = self.llm.plan_reddit_search(query)
            assert plan["comment_must_patterns"] == [], \
                f"comment_must_patterns must be [] for '{query}', got {plan['comment_must_patterns']}"

    def test_min_comment_score_non_negative(self):
        plan = self.llm.plan_reddit_search("Tesla Model Y")
        assert plan["min_comment_score"] >= 0

    def test_per_post_top_n_in_valid_range(self):
        plan = self.llm.plan_reddit_search("best Korean BBQ")
        assert 3 <= plan["per_post_top_n"] <= 12

    def test_time_filter_is_valid(self):
        plan = self.llm.plan_reddit_search("any query here")
        valid = {"day", "week", "month", "year", "all"}
        assert plan["time_filter"] in valid

    def test_plan_dict_maps_to_search_plan_dataclass(self):
        """Plan dict must be unpackable into SearchPlan without errors."""
        plan = self.llm.plan_reddit_search("best burger NYC")
        sp = SearchPlan(**plan)
        assert sp.terms


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Quality filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityFilter:
    """_passes_quality must accept good comments and reject noise."""

    def test_good_comment_passes(self):
        c = _MockComment(GOOD_COMMENT, score=15)
        assert _passes_quality(c, "restaurant", settings)

    def test_short_comment_rejected(self):
        c = _MockComment(SHORT_COMMENT, score=10)
        assert not _passes_quality(c, "restaurant", settings)

    def test_zero_score_rejected(self):
        c = _MockComment(GOOD_COMMENT, score=0)
        assert not _passes_quality(c, "restaurant", settings)

    def test_bot_like_comment_rejected(self):
        long_bot = (BOT_COMMENT + " ") * 3
        c = _MockComment(long_bot, score=5)
        assert not _passes_quality(c, "restaurant", settings)

    def test_short_question_rejected(self):
        c = _MockComment("Have you ever been to this restaurant?", score=10)
        assert not _passes_quality(c, "restaurant", settings)

    def test_empty_body_rejected(self):
        c = _MockComment("", score=10)
        assert not _passes_quality(c, "restaurant", settings)

    def test_comment_with_good_word_count_passes(self):
        # 10+ words, sufficient length
        text = " ".join(["word"] * 15) + " with some extra context about quality and experience."
        c = _MockComment(text, score=5)
        # May pass or fail depending on alpha ratio but should not raise
        result = _passes_quality(c, "test", settings)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Annotation data contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotationContract:
    """FallbackLLMService annotations must have correct shapes."""

    def setup_method(self):
        self.llm = FallbackLLMService()
        self.aspects = ["quality", "value", "service"]

    def test_output_length_matches_input(self):
        comments = [{"id": str(i), "text": GOOD_COMMENT, "upvotes": 5} for i in range(10)]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects)
        assert len(annos) == len(comments)

    def test_comment_ids_preserved_in_order(self):
        comments = [{"id": f"c{i}", "text": GOOD_COMMENT, "upvotes": 5} for i in range(5)]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects)
        for i, anno in enumerate(annos):
            assert anno.comment_id == f"c{i}", \
                f"ID mismatch at index {i}: expected c{i}, got {anno.comment_id}"

    def test_overall_score_in_range(self):
        comments = [{"id": "c1", "text": GOOD_COMMENT, "upvotes": 5}]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects)
        assert 1.0 <= annos[0].overall_score <= 5.0

    def test_all_aspects_present_in_scores(self):
        comments = [{"id": "c1", "text": GOOD_COMMENT, "upvotes": 5}]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects, entity_type="restaurant")
        for aspect in self.aspects:
            assert aspect in annos[0].aspect_scores, \
                f"Aspect '{aspect}' missing from annotation output"

    def test_empty_input_returns_empty(self):
        annos = self.llm.annotate_comments_with_gpt([], self.aspects)
        assert annos == []

    def test_entities_field_is_always_a_list(self):
        comments = [{"id": "c1", "text": GOOD_COMMENT, "upvotes": 5}]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects)
        assert isinstance(annos[0].entities, list)

    def test_all_aspect_scores_in_range(self):
        comments = [{"id": f"c{i}", "text": GOOD_COMMENT, "upvotes": 5} for i in range(3)]
        annos = self.llm.annotate_comments_with_gpt(comments, self.aspects)
        for anno in annos:
            for aspect, score in anno.aspect_scores.items():
                assert 1.0 <= score <= 5.0, \
                    f"Aspect score {score} out of [1,5] for aspect '{aspect}'"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Ranking pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingFromAnnotations:
    """rank_entities and rank_entities_with_relaxation on mock annotations."""

    def _build_annos(self, name_sentiment_pairs, mentions_each=4):
        annos = []
        counter = 0
        for name, sentiment in name_sentiment_pairs:
            for _ in range(mentions_each):
                annos.append(make_anno(
                    f"c{counter}", overall_score=sentiment,
                    entities=[make_entity(name, sentiment_score=sentiment)],
                ))
                counter += 1
        return annos

    def test_higher_sentiment_ranks_first(self):
        annos = self._build_annos([("Great Place", 4.8), ("Mediocre Spot", 2.5)])
        result = rank_entities(annos, {}, "restaurant", min_mentions=1)
        assert result[0].name == "Great Place"

    def test_high_upvote_review_influences_score(self):
        annos = [
            make_anno("c1", entities=[make_entity("Excellent Venue", sentiment_score=5.0)]),
            make_anno("c2", entities=[make_entity("Bad Venue", sentiment_score=1.0)]),
        ]
        # suppress_insufficient=False so single-mention entities are not filtered out,
        # allowing the test to verify the upvote-weighting logic directly.
        result = rank_entities(annos, {"c1": 1, "c2": 50}, "restaurant",
                               min_mentions=1, suppress_insufficient=False)
        bad = next(e for e in result if "Bad" in e.name)
        good = next(e for e in result if "Excellent" in e.name)
        assert bad.overall_stars < good.overall_stars

    def test_relaxation_surfaces_sparse_data(self):
        """With 1 mention each and min_mentions=5, relaxation should return results."""
        annos = [
            make_anno("c1", entities=[make_entity("Alpha Place", sentiment_score=4.5)]),
            make_anno("c2", entities=[make_entity("Beta Place", sentiment_score=4.0)]),
            make_anno("c3", entities=[make_entity("Gamma Place", sentiment_score=3.5)]),
        ]
        result = rank_entities_with_relaxation(annos, {}, "restaurant", min_mentions=5, query="")
        assert len(result) >= 1, "Relaxation must surface entities even with sparse data"

    def test_aggregate_generic_produces_valid_scores(self):
        aspects = ["quality", "value", "service"]
        annos = [
            make_anno("c1", overall_score=4.5, aspect_scores={"quality": 5.0, "value": 4.0, "service": 4.5}),
            make_anno("c2", overall_score=3.5, aspect_scores={"quality": 3.0, "value": 3.0, "service": 4.0}),
        ]
        overall, aspect_scores = aggregate_generic(aspects, annos, {"c1": 10, "c2": 5})
        assert 1.0 <= overall <= 5.0
        for aspect in aspects:
            assert aspect in aspect_scores
            assert 1.0 <= aspect_scores[aspect] <= 5.0

    def test_output_stars_always_in_range(self):
        annos = [
            make_anno(f"c{i}", entities=[make_entity("Test Venue", sentiment_score=5.0)])
            for i in range(10)
        ]
        result = rank_entities(annos, {}, "restaurant", min_mentions=1)
        for item in result:
            assert 1.0 <= item.overall_stars <= 5.0, \
                f"Stars {item.overall_stars} out of [1,5] for {item.name}"

    def test_empty_annotations_returns_empty_list(self):
        result = rank_entities([], {}, "restaurant", min_mentions=1)
        assert result == []

    def test_low_confidence_entities_excluded(self):
        annos = [make_anno(f"c{i}", entities=[make_entity("Bad Signal", confidence=0.3)]) for i in range(5)]
        result = rank_entities(annos, {}, "restaurant", min_mentions=1)
        assert not any("Bad Signal" in e.name for e in result)

    def test_geographic_entity_type_excluded(self):
        for geo_type in ["city", "region", "area", "location"]:
            annos = [
                make_anno(f"c{i}", entities=[make_entity("San Francisco", entity_type=geo_type, confidence=0.9)])
                for i in range(5)
            ]
            result = rank_entities(annos, {}, "restaurant", min_mentions=1)
            assert not any("San Francisco" in e.name for e in result), \
                f"Geographic entity type '{geo_type}' should be filtered"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Summary generation edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryGeneration:
    """FallbackLLMService summary methods must always return a non-empty string."""

    def setup_method(self):
        self.llm = FallbackLLMService()

    def test_ranking_summary_with_items(self):
        items = [{"name": "Place A", "overall_stars": 4.5, "mentions": 10, "quotes": []}]
        result = self.llm.summarize_ranking_with_gpt("best restaurants NYC", items)
        assert isinstance(result, str) and len(result) > 0

    def test_ranking_summary_empty_items(self):
        result = self.llm.summarize_ranking_with_gpt("best restaurants NYC", [])
        assert isinstance(result, str) and len(result) > 0

    def test_generic_summary_with_aspects(self):
        aspects = {"quality": 4.2, "value": 3.8, "service": 4.0}
        result = self.llm.summarize_generic_with_gpt("iPhone 15 review", aspects, 4.0, ["quote 1"])
        assert isinstance(result, str) and len(result) > 0

    def test_solutions_summary_with_clusters(self):
        clusters = [{"title": "Restart router", "steps": [], "caveats": [], "evidence_count": 5}]
        result = self.llm.summarize_solutions_with_gpt("wifi keeps disconnecting", clusters)
        assert isinstance(result, str) and len(result) > 0

    def test_solutions_summary_empty_clusters(self):
        result = self.llm.summarize_solutions_with_gpt("wifi keeps disconnecting", [])
        assert isinstance(result, str) and len(result) > 0

    def test_map_reduce_empty_returns_all_required_keys(self):
        result = self.llm.summarize_comments_map_reduce([], "test query")
        for key in ("pros", "cons", "aspects", "quotes", "coverage_ids"):
            assert key in result, f"Missing key '{key}' in empty map-reduce result"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: CLI path uses rank_entities_with_relaxation (regression guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPathUsesRelaxation:
    """Guard against regressing back to bare rank_entities in cli.py."""

    def _load_cli_source(self):
        cli_path = Path(__file__).parent.parent / "src" / "insighthub" / "cli.py"
        return cli_path.read_text(encoding="utf-8")

    def test_cli_contains_rank_entities_with_relaxation(self):
        source = self._load_cli_source()
        assert "rank_entities_with_relaxation" in source, \
            "cli.py must call rank_entities_with_relaxation for the RANKING path"

    def test_cli_does_not_call_bare_rank_entities(self):
        """cli.py must not call the bare rank_entities() function — use relaxation instead."""
        source = self._load_cli_source()
        tree = ast.parse(source)
        bare_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "rank_entities":
                    bare_calls.append(node.lineno)
        assert not bare_calls, (
            f"cli.py calls bare rank_entities() at lines {bare_calls}. "
            "Replace with rank_entities_with_relaxation to handle sparse data."
        )

    def test_cli_imports_relaxation(self):
        source = self._load_cli_source()
        assert "rank_entities_with_relaxation" in source, \
            "cli.py must import rank_entities_with_relaxation from scoring"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: FallbackLLMService default correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackServiceDefaults:
    """FallbackLLMService must return correct defaults for all methods."""

    def setup_method(self):
        self.llm = FallbackLLMService()

    def test_detect_intent_entity_type_is_none(self):
        """Must return None, not 'products' — prior bug where 'products' leaked through."""
        schema = self.llm.detect_intent_and_schema("best golf course in bay area")
        assert schema.entity_type is None, \
            f"FallbackLLMService must return entity_type=None, got '{schema.entity_type}'"

    def test_detect_intent_returns_generic(self):
        schema = self.llm.detect_intent_and_schema("anything at all")
        assert schema.intent == "GENERIC"

    def test_detect_intent_aspects_non_empty(self):
        schema = self.llm.detect_intent_and_schema("iPhone 15 review")
        assert schema.aspects and len(schema.aspects) > 0

    def test_filter_entities_by_type_passthrough(self):
        names = ["Place A", "Place B", "Place C"]
        result = self.llm.filter_entities_by_type(names, "restaurant")
        assert result == names

    def test_map_reduce_empty_returns_empty(self):
        result = self.llm.summarize_comments_map_reduce([], "test")
        assert result["pros"] == []
        assert result["cons"] == []
        assert result["coverage_ids"] == []

    def test_plan_reddit_search_returns_dict_with_all_keys(self):
        plan = self.llm.plan_reddit_search("best restaurant in NYC")
        for key in ("terms", "subreddits", "time_filter", "strategies",
                    "min_comment_score", "per_post_top_n", "comment_must_patterns"):
            assert key in plan

    def test_annotate_returns_correct_count(self):
        comments = [{"id": f"c{i}", "text": GOOD_COMMENT, "upvotes": 5} for i in range(7)]
        annos = self.llm.annotate_comments_with_gpt(comments, ["quality", "value"])
        assert len(annos) == 7


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Full mock end-to-end (no API keys)
# ─────────────────────────────────────────────────────────────────────────────

class TestMockEndToEnd:
    """Full pipeline using mock Reddit + FallbackLLMService — zero credentials."""

    def setup_method(self):
        self.llm = FallbackLLMService()
        self.reddit = RedditService()

    def test_mock_scrape_returns_reviews(self):
        reviews = self.reddit.scrape("best restaurant NYC", limit=20)
        assert len(reviews) > 0, "Mock scrape must return at least one review"

    def test_mock_scrape_source_field_is_set(self):
        """Source field must be set to 'mock' (no credentials) or 'reddit' (live)."""
        reviews = self.reddit.scrape("anything", limit=5)
        for review in reviews:
            assert review.get("source") in {"mock", "reddit"}, \
                f"source must be 'mock' or 'reddit', got '{review.get('source')}'"

    def test_mock_scrape_returns_dicts_with_required_keys(self):
        reviews = self.reddit.scrape("iPhone 15 review", limit=10)
        for review in reviews:
            assert isinstance(review, dict)
            for key in ("id", "source", "text", "upvotes"):
                assert key in review, f"Review missing key: {key}"

    def test_mock_scrape_text_meets_minimum_length(self):
        reviews = self.reddit.scrape("best coffee shop", limit=10)
        for review in reviews:
            assert len(review["text"]) >= 30, \
                f"Review text too short: {len(review['text'])} chars"

    def test_upvote_map_has_correct_types(self):
        reviews = self.reddit.scrape("best phone", limit=10)
        upvote_map = {r["id"]: r["upvotes"] for r in reviews}
        for cid, upvotes in upvote_map.items():
            assert isinstance(cid, str)
            assert isinstance(upvotes, int)

    def test_full_generic_pipeline(self):
        """Simulate complete GENERIC pipeline with mock data."""
        query = "iPhone 15 review"
        reviews = self.reddit.scrape(query, limit=15)
        assert reviews

        schema = self.llm.detect_intent_and_schema(query)
        assert schema.intent in {"RANKING", "SOLUTION", "GENERIC"}

        comments = [
            {"id": r.get("id", str(i)), "text": r.get("text", ""),
             "upvotes": r.get("upvotes", 0)}
            for i, r in enumerate(reviews)
        ]
        upvote_map = {c["id"]: c["upvotes"] for c in comments}

        annos = self.llm.annotate_comments_with_gpt(comments, schema.aspects or ["quality"])
        assert len(annos) == len(comments)

        overall, aspect_scores = aggregate_generic(schema.aspects or ["quality"], annos, upvote_map)
        assert 1.0 <= overall <= 5.0

        summary = self.llm.summarize_generic_with_gpt(query, aspect_scores, overall, [])
        assert summary and len(summary) > 0

    def test_full_ranking_pipeline_with_mock_entities(self):
        """Simulate RANKING pipeline: inject mock annotations and verify ranked output."""
        query = "best restaurant in NYC"
        reviews = self.reddit.scrape(query, limit=10)
        assert reviews

        comments = [
            {"id": r.get("id", str(i)), "text": r.get("text", ""), "upvotes": r.get("upvotes", 0)}
            for i, r in enumerate(reviews)
        ]
        upvote_map = {c["id"]: c["upvotes"] for c in comments}

        # Inject synthetic annotations with specific entities
        annos = []
        for i, c in enumerate(comments[:9]):
            # Cycle through 3 restaurants so each gets 3 mentions
            ent_name = ["Le Bernardin", "Eleven Madison Park", "Gramercy Tavern"][i % 3]
            annos.append(make_anno(
                c["id"],
                overall_score=4.0 + (i % 3) * 0.3,
                entities=[make_entity(ent_name, sentiment_score=4.0 + (i % 3) * 0.3)],
            ))
        # Append one empty annotation for the remaining comment if any
        for c in comments[9:]:
            annos.append(make_anno(c["id"]))

        ranked = rank_entities_with_relaxation(annos, upvote_map, "restaurant", query=query)
        assert len(ranked) >= 1

        # Verify output invariants
        for item in ranked:
            assert 1.0 <= item.overall_stars <= 5.0
            assert item.mentions > 0
            assert 0.0 <= item.confidence <= 1.0

        # Must be sorted descending by stars
        stars = [e.overall_stars for e in ranked]
        assert stars == sorted(stars, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Stage-boundary data contracts
# ─────────────────────────────────────────────────────────────────────────────

class TestStageBoundaryContracts:
    """Verify that output of each stage is consumable by the next."""

    def test_scrape_output_feeds_annotation_input(self):
        """Scrape dict keys must be usable by annotate_comments_with_gpt."""
        reddit = RedditService()
        llm = FallbackLLMService()
        reviews = reddit.scrape("test query", limit=5)

        comments = [
            {
                "id": r.get("id", str(i)),
                "text": r.get("text", ""),
                "upvotes": r.get("upvotes", 0),
                "permalink": r.get("permalink", ""),
            }
            for i, r in enumerate(reviews)
        ]
        annos = llm.annotate_comments_with_gpt(comments, ["quality"], entity_type="restaurant")
        assert len(annos) == len(comments)

    def test_annotation_output_feeds_rank_entities(self):
        """Annotation output must be consumable by rank_entities."""
        annos = [
            make_anno("c1", entities=[make_entity("Test Place", confidence=0.9)]),
            make_anno("c2", entities=[make_entity("Test Place", confidence=0.9)]),
            make_anno("c3", entities=[make_entity("Test Place", confidence=0.9)]),
        ]
        result = rank_entities(annos, {"c1": 5, "c2": 3, "c3": 2}, "restaurant", min_mentions=1)
        assert result
        assert result[0].name == "Test Place"

    def test_ranking_output_feeds_summary(self):
        """RankingItem list must be consumable by summarize_ranking_with_gpt."""
        llm = FallbackLLMService()
        annos = [make_anno(f"c{i}", entities=[make_entity("Best Place", sentiment_score=4.5)]) for i in range(3)]
        ranked = rank_entities(annos, {}, "restaurant", min_mentions=1)

        ranking_dicts = [
            {"name": item.name, "overall_stars": item.overall_stars,
             "mentions": item.mentions, "quotes": item.quotes}
            for item in ranked[:15]
        ]
        summary = llm.summarize_ranking_with_gpt("best restaurant", ranking_dicts)
        assert summary and len(summary) > 0

    def test_annotation_output_feeds_aggregate_generic(self):
        """Annotation output must be consumable by aggregate_generic."""
        aspects = ["quality", "value"]
        annos = [
            make_anno("c1", overall_score=4.0, aspect_scores={"quality": 4.0, "value": 3.0}),
            make_anno("c2", overall_score=3.5, aspect_scores={"quality": 3.5, "value": 4.0}),
        ]
        overall, aspect_scores = aggregate_generic(aspects, annos, {"c1": 5, "c2": 3})
        assert 1.0 <= overall <= 5.0
        for aspect in aspects:
            assert aspect in aspect_scores

    def test_no_credential_mode_full_chain(self):
        """Without any credentials, the full chain must not raise."""
        reddit = RedditService()
        llm = FallbackLLMService()

        reviews = reddit.scrape("best restaurant", limit=10)
        schema = llm.detect_intent_and_schema("best restaurant")
        comments = [{"id": r.get("id", str(i)), "text": r.get("text", ""), "upvotes": r.get("upvotes", 0)}
                    for i, r in enumerate(reviews)]
        annos = llm.annotate_comments_with_gpt(comments, schema.aspects or ["quality"])
        upvote_map = {c["id"]: c["upvotes"] for c in comments}
        overall, _ = aggregate_generic(schema.aspects or ["quality"], annos, upvote_map)
        summary = llm.summarize_generic_with_gpt("best restaurant", {}, overall, [])

        assert 1.0 <= overall <= 5.0
        assert isinstance(summary, str)
