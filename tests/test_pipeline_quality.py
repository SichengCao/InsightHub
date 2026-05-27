"""
Self-test for the two pipeline quality fixes:
  1. filter_relevant_comments — drops off-topic noise before annotation
  2. validate_entity_locations — strips out-of-area venues from rankings

Run: python -m pytest tests/test_pipeline_quality.py -v -s
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from insighthub.services.llm import LLMServiceFactory, OpenAIService


QUERY_GOLF = "best golf course in bay area"

# ── Relevance filter test data ─────────────────────────────────────────────────

RELEVANT_COMMENTS = [
    {
        "id": "r1",
        "text": "Harding Park is the best muni course in the Bay Area, well maintained greens and great views of the lake.",
        "upvotes": 45,
        "permalink": "",
    },
    {
        "id": "r2",
        "text": "I played Presidio Golf Club last weekend, greens were a bit slow but the scenery is unbeatable.",
        "upvotes": 22,
        "permalink": "",
    },
    {
        "id": "r3",
        "text": "Crystal Springs is underrated — rolling hills, reasonable green fee, and rarely crowded on weekdays.",
        "upvotes": 18,
        "permalink": "",
    },
    {
        "id": "r4",
        "text": "Chuck Corica Oakland is great value at around $60 for residents, the South Course renovation is nice.",
        "upvotes": 31,
        "permalink": "",
    },
]

IRRELEVANT_COMMENTS = [
    {
        "id": "i1",
        "text": "Has anyone tried the new pho place on Mission St? The broth is incredible.",
        "upvotes": 1312,
        "permalink": "",
    },
    {
        "id": "i2",
        "text": "https://i.imgur.com/abc123.jpg",
        "upvotes": 477,
        "permalink": "",
    },
    {
        "id": "i3",
        "text": "This new YouTube drama between streamers is wild, I can't stop watching the drama unfold.",
        "upvotes": 100,
        "permalink": "",
    },
    {
        "id": "i4",
        "text": "Fort Mason has the best wedding venue in the city, we're planning our reception there next spring.",
        "upvotes": 4,
        "permalink": "",
    },
]

ALL_COMMENTS = RELEVANT_COMMENTS + IRRELEVANT_COMMENTS


# ── Location validator test data ───────────────────────────────────────────────

class FakeEntity:
    def __init__(self, name):
        self.name = name
        self.overall_stars = 4.0
        self.mentions = 5
        self.confidence = 0.8


OUT_OF_AREA_ENTITIES = [
    FakeEntity("Gold Mountain Golf Course"),      # Bremerton, WA
    FakeEntity("Chambers Bay Golf Course"),        # University Place, WA
    FakeEntity("Home Course"),                     # DuPont, WA
    FakeEntity("Bear Mountain Golf Resort"),       # Victoria, BC, Canada
]

IN_AREA_ENTITIES = [
    FakeEntity("TPC Harding Park"),               # San Francisco, CA
    FakeEntity("Presidio Golf Club"),             # San Francisco, CA
    FakeEntity("Crystal Springs Golf Course"),    # Burlingame, CA (Peninsula / Bay Area)
    FakeEntity("Pebble Beach Golf Links"),        # Monterey, CA (borderline — but often considered "Bay Area adjacent")
]

MIXED_ENTITIES = OUT_OF_AREA_ENTITIES + IN_AREA_ENTITIES


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def llm():
    svc = LLMServiceFactory.create()
    assert isinstance(svc, OpenAIService), "Need a real OpenAI key for this test"
    return svc


class TestRelevanceFilter:
    def test_keeps_all_relevant(self, llm):
        result = llm.filter_relevant_comments(RELEVANT_COMMENTS, QUERY_GOLF)
        kept_ids = {c["id"] for c in result}
        print(f"\n[relevance] kept {len(result)}/{len(RELEVANT_COMMENTS)} relevant comments: {kept_ids}")
        # At minimum 3 of 4 clearly on-topic comments should pass
        assert len(result) >= 3, f"Too many relevant comments dropped: {kept_ids}"

    def test_drops_irrelevant(self, llm):
        result = llm.filter_relevant_comments(IRRELEVANT_COMMENTS, QUERY_GOLF)
        kept_ids = {c["id"] for c in result}
        print(f"\n[relevance] kept {len(result)}/{len(IRRELEVANT_COMMENTS)} irrelevant comments: {kept_ids}")
        # All 4 off-topic comments should be dropped (or at most 1 passes)
        assert len(result) <= 1, f"Too many irrelevant comments kept: {kept_ids}"

    def test_mixed_batch(self, llm):
        result = llm.filter_relevant_comments(ALL_COMMENTS, QUERY_GOLF)
        kept_ids = {c["id"] for c in result}
        relevant_ids = {c["id"] for c in RELEVANT_COMMENTS}
        irrelevant_ids = {c["id"] for c in IRRELEVANT_COMMENTS}

        true_positives = kept_ids & relevant_ids
        false_positives = kept_ids & irrelevant_ids
        false_negatives = relevant_ids - kept_ids

        print(f"\n[relevance] mixed test:")
        print(f"  Kept {len(result)}/{len(ALL_COMMENTS)} total")
        print(f"  True positives  (relevant kept):    {true_positives}")
        print(f"  False positives (irrelevant kept):  {false_positives}")
        print(f"  False negatives (relevant dropped): {false_negatives}")

        assert len(true_positives) >= 3, f"Too many relevant comments dropped: {false_negatives}"
        assert len(false_positives) <= 1, f"Too many irrelevant comments passed through: {false_positives}"

    def test_empty_input(self, llm):
        result = llm.filter_relevant_comments([], QUERY_GOLF)
        assert result == []

    def test_fail_open_on_bad_response(self, llm, monkeypatch):
        """If GPT returns garbage, all comments should be kept (fail-open)."""
        monkeypatch.setattr(llm, "chat", lambda **kw: "this is not json at all!!!")
        result = llm.filter_relevant_comments(RELEVANT_COMMENTS[:2], QUERY_GOLF)
        assert len(result) == 2, "Should fail-open and keep all comments when GPT fails"


class TestLocationValidator:
    def test_drops_out_of_area(self, llm):
        result = llm.validate_entity_locations(OUT_OF_AREA_ENTITIES, QUERY_GOLF)
        kept_names = [e.name for e in result]
        print(f"\n[location] out-of-area test — kept: {kept_names}")
        # All PNW/Canada venues should be rejected for "bay area"
        assert len(result) == 0, f"Out-of-area venues should all be dropped, kept: {kept_names}"

    def test_keeps_in_area(self, llm):
        result = llm.validate_entity_locations(IN_AREA_ENTITIES, QUERY_GOLF)
        kept_names = [e.name for e in result]
        print(f"\n[location] in-area test — kept: {kept_names}")
        # TPC Harding Park, Presidio, Crystal Springs are unambiguously Bay Area
        unambiguous = {"TPC Harding Park", "Presidio Golf Club", "Crystal Springs Golf Course"}
        kept_set = set(kept_names)
        missing = unambiguous - kept_set
        assert len(missing) == 0, f"Clear Bay Area courses were wrongly dropped: {missing}"

    def test_mixed_entities(self, llm):
        result = llm.validate_entity_locations(MIXED_ENTITIES, QUERY_GOLF)
        kept_names = {e.name for e in result}
        print(f"\n[location] mixed test — kept: {kept_names}")

        out_of_area_names = {e.name for e in OUT_OF_AREA_ENTITIES}
        leaked = kept_names & out_of_area_names
        assert len(leaked) == 0, f"Out-of-area venues leaked into results: {leaked}"

    def test_no_location_query(self, llm):
        """Query with no location should keep most or all entities (no geographic restriction applies)."""
        result = llm.validate_entity_locations(OUT_OF_AREA_ENTITIES, "best golf course")
        print(f"\n[location] no-location query — kept {len(result)}/{len(OUT_OF_AREA_ENTITIES)}")
        # GPT may still occasionally drop 1 entity with ambiguous name; require at least 3/4 kept
        assert len(result) >= len(OUT_OF_AREA_ENTITIES) - 1, (
            f"Too many entities dropped for a no-location query: "
            f"kept {len(result)}/{len(OUT_OF_AREA_ENTITIES)}"
        )

    def test_empty_input(self, llm):
        result = llm.validate_entity_locations([], QUERY_GOLF)
        assert result == []
