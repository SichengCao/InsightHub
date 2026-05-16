"""
Verify detect_intent_and_schema returns venue/shop types for food queries,
not the item being served.

Success criteria:
- Food/drink queries → entity_type contains "restaurant", "shop", "bar", or "cafe"
- Product queries    → entity_type is a product type (phone, laptop, etc.)
- Venue queries      → entity_type is the venue category (golf_course, hotel, etc.)
- No bare food words → entity_type is never "ramen", "pizza", "coffee", "taco"
"""
import sys, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from insighthub.services.llm import LLMServiceFactory

llm = LLMServiceFactory.create()

FOOD_QUERIES = [
    "best ramen in Tokyo",
    "best pizza in NYC",
    "best coffee in Seattle",
    "best tacos in LA",
]

PRODUCT_QUERIES = [
    ("best iPhone 15", ["phone", "smartphone"]),
    ("best laptop for students", ["laptop", "computer"]),
]

VENUE_QUERIES = [
    ("best golf courses in bay area", ["golf_course", "golf"]),
    ("best hotels in Paris", ["hotel"]),
]

BARE_FOOD_WORDS = {"ramen", "pizza", "coffee", "taco", "tacos", "sushi", "burger", "food", "drink"}

passed = 0
failed = 0

print("=== entity_type tests ===\n")

for q in FOOD_QUERIES:
    schema = llm.detect_intent_and_schema(q)
    et = (schema.entity_type or "").lower()
    is_venue = any(w in et for w in ["restaurant", "shop", "bar", "cafe"])
    is_bare_food = et in BARE_FOOD_WORDS
    ok = is_venue and not is_bare_food
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {q!r}")
    print(f"       entity_type={et!r}  (expected venue type, not bare food word)")
    if ok:
        passed += 1
    else:
        failed += 1

for q, expected_words in PRODUCT_QUERIES:
    schema = llm.detect_intent_and_schema(q)
    et = (schema.entity_type or "").lower()
    ok = any(w in et for w in expected_words)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {q!r}")
    print(f"       entity_type={et!r}  (expected one of {expected_words})")
    if ok:
        passed += 1
    else:
        failed += 1

for q, expected_words in VENUE_QUERIES:
    schema = llm.detect_intent_and_schema(q)
    et = (schema.entity_type or "").lower()
    ok = any(w in et for w in expected_words)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {q!r}")
    print(f"       entity_type={et!r}  (expected one of {expected_words})")
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{'='*30}")
print(f"Result: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
