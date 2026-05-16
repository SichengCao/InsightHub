"""
Result quality validator for InsightHub.
Checks: entity_type correctness, cuisine match, annotation rate, entity signal strength.
Usage: python validate.py "best Korean restaurant in New York"
       python validate.py  (runs default test suite)
"""
import sys, logging, json, re
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from insighthub.services.llm import LLMServiceFactory
from insighthub.services.reddit_client import RedditService
from insighthub.core.scoring import rank_entities

LIMIT = 40

def validate(query, expected_type_keyword=None):
    llm    = LLMServiceFactory.create()
    reddit = RedditService()
    issues = []
    score  = 0
    total  = 0

    print(f"\n{'='*60}")
    print(f"  {query}")
    print(f"{'='*60}")

    # 1. Intent / entity_type
    schema = llm.detect_intent_and_schema(query)
    et = (schema.entity_type or "").lower()
    print(f"\n[1] intent={schema.intent}  entity_type={et}")
    print(f"    aspects={schema.aspects}")
    total += 1
    if expected_type_keyword and expected_type_keyword.lower() in et:
        print(f"    PASS entity_type contains '{expected_type_keyword}'")
        score += 1
    elif expected_type_keyword:
        issues.append(f"entity_type '{et}' missing '{expected_type_keyword}' -- wrong venue type")
    else:
        score += 1

    # 2. Scrape
    reviews = reddit.scrape(query, limit=LIMIT)
    print(f"\n[2] scrape: {len(reviews)} comments")
    total += 1
    if len(reviews) >= 10:
        print(f"    PASS sufficient data")
        score += 1
    else:
        issues.append(f"only {len(reviews)} comments -- too sparse")

    # 3. Annotation completeness
    def _val(r, *keys):
        for k in keys:
            v = r.get(k) if isinstance(r, dict) else getattr(r, k, None)
            if v is not None: return v
        return None

    comments   = [{"id": _val(r,"id"), "text": _val(r,"text","body"), "upvotes": _val(r,"upvotes","score") or 0} for r in reviews]
    upvote_map = {c["id"]: c["upvotes"] for c in comments}
    annos      = llm.annotate_comments_with_gpt(comments, schema.aspects, entity_type=schema.entity_type, query=query)
    with_entities = [a for a in annos if a.entities]
    rate = len(annos) / len(comments) * 100 if comments else 0

    print(f"\n[3] annotate: {len(annos)}/{len(comments)} ({rate:.0f}%)  -- {len(with_entities)} have entities")
    total += 1
    if rate >= 90:
        print(f"    PASS annotation completeness OK")
        score += 1
    else:
        issues.append(f"only {rate:.0f}% annotated -- batch/token issue")

    # 4. Ranking
    ranked = rank_entities(annos, upvote_map, schema.entity_type, min_mentions=1, query=query)
    if schema.entity_type and ranked:
        valid_names = set(llm.filter_entities_by_type([e.name for e in ranked], schema.entity_type))
        ranked = [e for e in ranked if e.name in valid_names]
    print(f"\n[4] ranking: {len(ranked)} entities")
    print(f"    {'#':<4} {'Name':<32} {'Stars':<6} {'Mentions':<10} Conf")
    print(f"    {'-'*4} {'-'*32} {'-'*6} {'-'*10} {'-'*6}")
    for i, e in enumerate(ranked[:10], 1):
        name = e.name.encode("ascii", "replace").decode()
        flag = "  <-- single mention" if e.mentions == 1 else ""
        print(f"    {i:<4} {name:<32} {e.overall_stars:<6.1f} {e.mentions:<10} {e.confidence:.0%}{flag}")

    total += 1
    if len(ranked) >= 3:
        print(f"    PASS {len(ranked)} entities ranked")
        score += 1
    else:
        issues.append(f"only {len(ranked)} entities -- very sparse")

    single = sum(1 for e in ranked if e.mentions == 1)
    total += 1
    if len(ranked) == 0 or single / len(ranked) <= 0.6:
        print(f"    PASS single-mention ratio OK ({single}/{len(ranked)})")
        score += 1
    else:
        issues.append(f"{single}/{len(ranked)} entities have 1 mention -- low signal")

    # 5. GPT cuisine check -- ask GPT to verify top-5 match expected type
    if ranked and expected_type_keyword:
        names = [e.name for e in ranked[:5]]
        # Use full entity_type (e.g. "korean restaurant") not just keyword ("Korean")
        entity_label = et.replace("_", " ") if et else expected_type_keyword
        verdict = ""
        try:
            verdict = llm.chat(
                system="You are a fact-checker. Answer ONLY with a JSON array of booleans.",
                user=(
                    f"For each name below, return true if it is a {entity_label} "
                    f"and false if it is not.\nNames: {names}\nReturn ONLY a JSON array like [true, false, true, ...]"
                ),
                temperature=0.0,
                max_tokens=60,
            )
            flags = json.loads(re.search(r'\[.*\]', verdict, re.S).group(0))
            wrong = [names[i] for i, ok in enumerate(flags) if not ok]
            total += 1
            if not wrong:
                print("    PASS GPT verified top-5 are all %s" % entity_label)
                score += 1
            else:
                print("    FAIL wrong-type entities in top-5: %s" % wrong)
                issues.append("wrong-type entities in top-5: %s" % wrong)
        except Exception as e:
            raw = repr(verdict)[:120] if verdict else "(no verdict)"
            print("    WARN GPT cuisine check failed: %s | raw: %s" % (repr(e), raw))

    # Summary
    print(f"\n{'-'*60}")
    pct = score / total * 100
    status = "PASS" if pct >= 80 else "WARN" if pct >= 60 else "FAIL"
    print(f"  [{status}] Quality score: {score}/{total} ({pct:.0f}%)")
    for iss in issues:
        print(f"  ISSUE: {iss}")
    if not issues:
        print(f"  No issues detected.")
    return pct


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(" ".join(sys.argv[1:]))
    else:
        results = [
            validate("best Korean restaurant in New York", expected_type_keyword="Korean"),
            validate("best golf courses in bay area",      expected_type_keyword="golf"),
            validate("best ramen in Tokyo",                expected_type_keyword="ramen"),
        ]
        print(f"\n{'='*60}")
        print(f"  Overall avg: {sum(results)/len(results):.0f}%")
