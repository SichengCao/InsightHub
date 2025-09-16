"""Basic usage examples for InsightHub."""

import os
from insighthub import RedditService, LLMServiceFactory
from insighthub.core.scoring import aggregate_generic, rank_entities

def example_ranking_query():
    """Example: Ranking query analysis."""
    print("🔍 Analyzing ranking query: iPhone vs Samsung Galaxy S24")
    
    # Initialize services
    reddit_service = RedditService()
    llm_service = LLMServiceFactory.create()
    
    # Scrape reviews
    reviews = reddit_service.scrape("iPhone vs Samsung Galaxy S24", limit=10)
    print(f"📊 Scraped {len(reviews)} reviews")
    
    # Detect intent and generate schema
    intent_schema = llm_service.detect_intent_and_schema("iPhone vs Samsung Galaxy S24")
    print(f"🎯 Intent: {intent_schema.intent}")
    print(f"📋 Aspects: {intent_schema.aspects}")
    
    # Annotate comments
    annotations = llm_service.annotate_comments_with_gpt(reviews, intent_schema.aspects)
    print(f"🤖 Annotated {len(annotations)} comments")
    
    # Create upvote map
    upvote_map = {review.get("id"): review.get("upvotes", 0) for review in reviews}
    
    # Rank entities
    if intent_schema.intent == "RANKING":
        ranking = rank_entities(annotations, upvote_map, intent_schema.entity_type, min_mentions=1)
        print(f"🏆 Found {len(ranking)} ranked entities:")
        for i, item in enumerate(ranking[:5], 1):
            print(f"  {i}. {item.name}: {item.overall_stars:.1f}/5 ({item.mentions} mentions)")

def example_generic_query():
    """Example: Generic query analysis."""
    print("\n🔍 Analyzing generic query: iPhone 15")
    
    # Initialize services
    reddit_service = RedditService()
    llm_service = LLMServiceFactory.create()
    
    # Scrape reviews
    reviews = reddit_service.scrape("iPhone 15", limit=10)
    print(f"📊 Scraped {len(reviews)} reviews")
    
    # Detect intent and generate schema
    intent_schema = llm_service.detect_intent_and_schema("iPhone 15")
    print(f"🎯 Intent: {intent_schema.intent}")
    
    # Annotate comments
    annotations = llm_service.annotate_comments_with_gpt(reviews, intent_schema.aspects)
    
    # Create upvote map
    upvote_map = {review.get("id"): review.get("upvotes", 0) for review in reviews}
    
    # Aggregate results
    if intent_schema.intent == "GENERIC":
        overall, aspect_averages = aggregate_generic(intent_schema.aspects, annotations, upvote_map)
        print(f"⭐ Overall rating: {overall:.1f}/5")
        print("📊 Aspect scores:")
        for aspect, score in aspect_averages.items():
            print(f"  {aspect}: {score:.1f}/5")

def example_solution_query():
    """Example: Solution query analysis."""
    print("\n🔍 Analyzing solution query: iPhone battery drain fix")
    
    # Initialize services
    reddit_service = RedditService()
    llm_service = LLMServiceFactory.create()
    
    # Scrape reviews
    reviews = reddit_service.scrape("iPhone battery drain fix", limit=10)
    print(f"📊 Scraped {len(reviews)} reviews")
    
    # Detect intent and generate schema
    intent_schema = llm_service.detect_intent_and_schema("iPhone battery drain fix")
    print(f"🎯 Intent: {intent_schema.intent}")
    
    # Annotate comments
    annotations = llm_service.annotate_comments_with_gpt(reviews, intent_schema.aspects)
    
    # Group by solution clusters
    if intent_schema.intent == "SOLUTION":
        clusters = {}
        for anno in annotations:
            if anno.solution_key:
                if anno.solution_key not in clusters:
                    clusters[anno.solution_key] = []
                clusters[anno.solution_key].append(anno)
        
        print(f"🔧 Found {len(clusters)} solution clusters:")
        for cluster_key, cluster_annos in clusters.items():
            print(f"  {cluster_key}: {len(cluster_annos)} solutions")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("🚀 InsightHub Examples")
    print("=" * 50)
    
    try:
        example_ranking_query()
        example_generic_query()
        example_solution_query()
        print("\n✅ All examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
