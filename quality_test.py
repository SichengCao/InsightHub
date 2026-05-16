"""
End-to-end quality test for InsightHub pipeline.
Runs: scrape → intent → annotate → rank
Prints a graded report on result quality.
"""
import sys
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("insighthub.services.llm").setLevel(logging.DEBUG)

sys.path.insert(0, "src")

from insighthub.services.llm import LLMServiceFactory
from insighthub.services.reddit_client import RedditService
from insighthub.core.scoring import rank_entities_with_relaxation

QUERIES = [
    "best golf courses in bay area",
    "best ramen in Tokyo",
]

COMMENT_LIMIT = 40


def grade_entities(entities, query):
    issues = []
    if not entities:
        issues.append("FAIL: no entities returned")
        return issues
    if len(entities) < 3:
        issues.append(f"WARN: only {len(entities)} entities found (sparse data)")
    single_mention = [e for e in entities if e.mentions == 1]
    if single_mention:
        issues.append(f"WARN: {len(single_mention)} single-mention entit{'y' if len(single_mention)==1 else 'ies'} in results: {[e.name for e in single_mention]}")
    low_conf = [e for e in entities if e.confidence < 0.6]
    if low_conf:
        issues.append(f"WARN: {len(low_conf)} low-confidence entities: {[e.name for e in low_conf]}")
    geo_keywords = {"area", "city", "region", "bay", "valley", "downtown", "neighborhood"}
    geo_slip = [e for e in entities if any(w in e.name.lower().split() for w in geo_keywords)]
    if geo_slip:
        issues.append(f"WARN: possible geographic entity leaked through: {[e.name for e in geo_slip]}")
    return issues


def run(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    llm = LLMServiceFactory.create()
    reddit = RedditService()

    # Step 1: intent + aspects
    print("\n[1] Intent detection...")
    schema = llm.detect_intent_and_schema(query)
    print(f"    Intent      : {schema.intent}")
    print(f"    Entity type : {schema.entity_type}")
    print(f"    Aspects     : {schema.aspects}")

    # Step 2: scrape
    print(f"\n[2] Scraping Reddit (limit={COMMENT_LIMIT})...")
    reviews = reddit.scrape(query, limit=COMMENT_LIMIT)
    print(f"    Got {len(reviews)} comments")
    if not reviews:
        print("    FAIL: no comments scraped")
        return

    # Step 3: annotate
    print(f"\n[3] Annotating with GPT...")
    def _get(r, *keys):
        for k in keys:
            v = r.get(k) if isinstance(r, dict) else getattr(r, k, None)
            if v is not None:
                return v
        return None

    comments = [{"id": _get(r, "id"), "text": _get(r, "text", "body"), "upvotes": _get(r, "upvotes", "score") or 0} for r in reviews]
    upvote_map = {c["id"]: c["upvotes"] for c in comments}
    annos = llm.annotate_comments_with_gpt(comments, schema.aspects, entity_type=schema.entity_type, query=query)
    with_entities = [a for a in annos if a.entities]
    print(f"    Annotated {len(annos)} comments — {len(with_entities)} with entities, {len(annos)-len(with_entities)} empty")

    # Show a sample annotation
    sample = next((a for a in annos if a.entities), None)
    if sample:
        print(f"\n    Sample annotation:")
        print(f"      overall_score : {sample.overall_score}")
        print(f"      aspect_scores : {dict(list(sample.aspect_scores.items())[:3])}")
        print(f"      entities extracted:")
        for e in sample.entities[:4]:
            name = e.name.encode("ascii", "replace").decode("ascii")
            print(f"        - {name!r:30s} type={e.entity_type}  conf={e.confidence:.2f}  sentiment={e.sentiment_score}  primary={e.is_primary}")

    # Step 4: rank
    print(f"\n[4] Ranking entities...")
    if schema.intent == "RANKING":
        ranked = rank_entities_with_relaxation(annos, upvote_map, schema.entity_type, query=query)
        print(f"    {len(ranked)} entities ranked\n")
        print(f"    {'Rank':<5} {'Name':<35} {'Stars':<7} {'Mentions':<10} {'Confidence'}")
        print(f"    {'-'*5} {'-'*35} {'-'*7} {'-'*10} {'-'*10}")
        for i, e in enumerate(ranked[:10], 1):
            name = e.name.encode("ascii", "replace").decode("ascii")
            print(f"    {i:<5} {name:<35} {e.overall_stars:<7.2f} {e.mentions:<10} {e.confidence:.2f}")

        # Quality grading
        issues = grade_entities(ranked, query)
        print(f"\n[5] Quality report:")
        if not issues:
            print("    OK: no issues detected")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"    (intent={schema.intent}, skipping entity ranking)")
        print(f"    OK: pipeline completed for non-ranking query")


if __name__ == "__main__":
    queries = sys.argv[1:] if len(sys.argv) > 1 else QUERIES
    for q in queries:
        run(q)
    print("\nDone.")
