"""Simple Streamlit UI for InsightHub."""

import streamlit as st
import logging
import hashlib
import re
import time
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from organized modules
from insighthub.core.config import settings
from insighthub.core.constants import SearchConstants
from insighthub.services.reddit_client import RedditService
from insighthub.services.llm import LLMServiceFactory
from insighthub.services.cross_platform_manager import CrossPlatformManager
from insighthub.core.cross_platform_models import Platform, QueryIntent
from insighthub.core.scoring import aggregate_generic, rank_entities
from insighthub.utils.data_prep import export_to_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions for real comment display
def _excerpt(s, n=240):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def _sig(s: str) -> str:
    s = re.sub(r"https?://\S+","", s or "").lower()
    s = re.sub(r"\s+"," ", s).strip()
    return hashlib.md5(s.encode()).hexdigest()

def _dedupe_keep_order(items, key=lambda x: x):
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k in seen: 
            continue
        seen.add(k)
        out.append(it)
    return out

# Page configuration
st.set_page_config(
    page_title="InsightHub — Review Analysis",
    page_icon="📈",
    layout="wide"
)

# Ensure config files exist
# Config files are no longer needed with GPT-only pipeline

# Initialize services
reddit_service = RedditService()
llm_service = LLMServiceFactory.create()
cross_platform_manager = CrossPlatformManager()

# Main UI
st.title("📈 InsightHub — Review Analysis")
st.write("Analyze reviews across multiple platforms (Reddit, YouTube) with AI-powered sentiment and aspect scoring.")

# Sidebar for search
with st.sidebar:
    st.header("🔎 Search")
    
    # Search input
    query = st.text_input("Search Reviews", value=st.session_state.get("query", "Tesla Model Y"))
    
    # Limit slider
    limit = st.slider("Number of Comments", 
                     SearchConstants.MIN_COMMENTS_UI, 
                     SearchConstants.MAX_COMMENTS_UI, 
                     SearchConstants.DEFAULT_COMMENTS_UI, 
                     step=10)
    
    # Platform selection
    st.subheader("🌐 Platforms")
    enable_cross_platform = st.checkbox("Enable Cross-Platform Analysis", value=True, 
                                       help="Analyze reviews from multiple platforms")
    
    if enable_cross_platform:
        default_platforms = [Platform.REDDIT, Platform.YOUTUBE]
        platform_options = [Platform.REDDIT.value, Platform.YOUTUBE.value]
        # Future platforms: Platform.GOOGLE.value, Platform.XIAOHONGSHU.value
        selected_platform_names = st.multiselect(
            "Select Platforms", 
            platform_options, 
            default=[p.value for p in default_platforms],
            help="Choose which platforms to search"
        )
        selected_platforms = [Platform(p) for p in selected_platform_names]
    else:
        selected_platforms = [Platform.REDDIT]
    
    # Subreddit count slider (only show for Reddit-only mode)
    if not enable_cross_platform or Platform.REDDIT in selected_platforms:
        st.subheader("🔍 Reddit Settings")
        subreddit_count = st.slider("Number of Subreddits", 
                                   SearchConstants.MIN_SUBREDDITS_UI, 
                                   SearchConstants.MAX_SUBREDDITS_UI, 
                                   SearchConstants.DEFAULT_SUBREDDITS_UI, 
                                   step=1, 
                                   help="More subreddits = better coverage but longer search time")
    else:
        subreddit_count = SearchConstants.DEFAULT_SUBREDDITS_UI
    
    # Platform bias calibration settings
    st.subheader("⚙️ Platform Calibration")
    
    # Import calibration functions
    from insighthub.core.scoring import set_bias_enabled, set_shrinkage_k, PLATFORM_BIAS
    
    # Bias calibration toggle
    bias_enabled = st.checkbox("Enable Platform Bias Calibration", value=True,
                              help="Adjust scores based on platform sentiment patterns")
    
    # Shrinkage parameter
    shrinkage_k = st.slider("Shrinkage Parameter (k)", 1, 100, 40,
                           help="Higher k = less aggressive calibration. Formula: α = n/(n+k)")
    
    # Apply settings
    set_bias_enabled(bias_enabled)
    set_shrinkage_k(shrinkage_k)
    
    # Show platform bias factors
    with st.expander("📊 Platform Bias Factors"):
        for platform, bias in PLATFORM_BIAS.items():
            bias_emoji = "📉" if bias < 0 else "📈" if bias > 0 else "➡️"
            st.write(f"{bias_emoji} **{platform.title()}**: {bias:+.2f}")
        st.caption("Negative = more critical, Positive = more positive, Zero = neutral")
    
    # Analyze button
    run_analysis = st.button("📊 Analyze Reviews", width='stretch')

# Main content area
if run_analysis:
    try:
        with st.spinner("Analyzing reviews..."):
            # Detect intent first
            intent_schema = llm_service.detect_intent_and_schema(query)
            intent = QueryIntent(intent_schema.intent) if intent_schema.intent in ["RANKING", "SOLUTION", "GENERIC"] else QueryIntent.GENERIC
            
            # Determine search strategy based on platform selection
            if len(selected_platforms) > 1 or (len(selected_platforms) == 1 and selected_platforms[0] != Platform.REDDIT):
                # Use cross-platform search
                logger.info(f"Using cross-platform search across {len(selected_platforms)} platforms...")
                
                # Show search plan
                with st.expander("🔎 Cross-platform search plan"):
                    st.info(f"**Intent**: {intent.value}")
                    st.info(f"**Platforms**: {', '.join([p.value for p in selected_platforms])}")
                    st.info(f"**Limit per platform**: {limit}")
                
                # Execute cross-platform search
                start_time = time.time()
                cross_platform_results = cross_platform_manager.search_cross_platform(
                    query, intent, limit_per_platform=limit, enabled_platforms=selected_platforms
                )
                search_time = time.time() - start_time
                
                # Extract reviews from all platforms
                all_reviews = []
                for platform_name, platform_reviews in cross_platform_results["platform_results"].items():
                    all_reviews.extend(platform_reviews)
                
                reviews = all_reviews
                
                # Show platform data availability info
                aggregated = cross_platform_results["aggregated"]
                total_reviews = sum(len(reviews) for reviews in cross_platform_results["platform_results"].values())
                st.info(f"📊 **Total reviews collected**: {total_reviews} across {len(cross_platform_results['platform_results'])} platforms")
                
            else:
                # Use Reddit-only search
                logger.info(f"Using Reddit-only search...")
                
                # Show Reddit search plan (debug)
                with st.expander("🔎 Reddit search plan"):
                    try:
                        plan = reddit_service._plan_search(query)
                        st.json({
                            "terms": plan.terms,
                            "subreddits": plan.subreddits,
                            "time_filter": plan.time_filter,
                            "strategies": plan.strategies,
                            "min_comment_score": plan.min_comment_score,
                            "per_post_top_n": plan.per_post_top_n,
                            "comment_must_patterns": plan.comment_must_patterns,
                        })
                    except Exception as e:
                        st.caption(f"(plan unavailable) {e}")
                
                # Scrape Reddit reviews with timing
                start_time = time.time()
                reviews = reddit_service.scrape(query, limit, subreddit_count)
                search_time = time.time() - start_time
            
            logger.info(f"Analyzing {len(reviews)} reviews...")
            
            # Show filtering transparency
            st.caption(f"🔍 Total reviews collected: {len(reviews)} in {search_time:.1f}s")
            
            if not reviews:
                st.warning("No reviews found! Try a different search term.")
            else:
                # Display intent information (already detected above)
                st.info(f"🎯 **Detected intent**: {intent_schema.intent}")
                if intent_schema.entity_type:
                    st.info(f"🏷️ **Entity type**: {intent_schema.entity_type}")
                st.info(f"🔍 **Aspects**: {', '.join(intent_schema.aspects[:5])}{'...' if len(intent_schema.aspects) > 5 else ''}")
                
                # Convert reviews to comment format for annotation
                comments = []
                for review in reviews:
                    comment = {
                        "id": review.get("id") if isinstance(review, dict) else review.id,
                        "text": review.get("text") if isinstance(review, dict) else review.text,
                        "upvotes": review.get("upvotes", 0) if isinstance(review, dict) else getattr(review, "upvotes", 0),
                        "permalink": review.get("permalink", "") if isinstance(review, dict) else getattr(review, "permalink", "")
                    }
                    comments.append(comment)
                
                # Annotate comments with GPT
                with st.spinner("Annotating comments with GPT..."):
                    annos = llm_service.annotate_comments_with_gpt(comments, intent_schema.aspects, intent_schema.entity_type, query)
                
                # Show platform breakdown with calibration info (for cross-platform searches)
                if len(selected_platforms) > 1:
                    with st.expander("📊 Platform Results"):
                        for platform_name, platform_reviews in cross_platform_results["platform_results"].items():
                            # Calculate raw platform score (before calibration)
                            platform_comments = []
                            for review in platform_reviews:
                                comment = {
                                    "id": review.get("id") if isinstance(review, dict) else review.id,
                                    "text": review.get("text") if isinstance(review, dict) else review.text,
                                    "upvotes": review.get("upvotes", 0) if isinstance(review, dict) else getattr(review, "upvotes", 0),
                                }
                                platform_comments.append(comment)
                            
                            # Get platform-specific annotations
                            platform_annos = [anno for anno in annos if any(comment["id"] == anno.comment_id for comment in platform_comments)]
                            platform_upvote_map = {comment["id"]: comment["upvotes"] for comment in platform_comments}
                            
                            if platform_annos:
                                # Calculate raw score
                                raw_scores = [(anno.overall_score, platform_upvote_map.get(anno.comment_id, 1)) for anno in platform_annos]
                                raw_score = sum(s*w for s, w in raw_scores) / sum(w for _, w in raw_scores) if raw_scores else 3.0
                                
                                # Calculate calibrated score
                                from insighthub.core.scoring import apply_platform_bias
                                calibrated_score = apply_platform_bias(raw_score, platform_name, len(platform_reviews))
                                
                                # Calculate shrinkage factor
                                from insighthub.core.scoring import _shrink_factor
                                alpha = _shrink_factor(len(platform_reviews))
                                
                                st.write(f"**{platform_name.title()}**: {len(platform_reviews)} reviews")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Raw Score", f"{raw_score:.2f}/5")
                                with col2:
                                    st.metric("Calibrated", f"{calibrated_score:.2f}/5")
                                with col3:
                                    st.metric("Shrink α", f"{alpha:.3f}")
                            else:
                                st.write(f"**{platform_name.title()}**: {len(platform_reviews)} reviews (no valid annotations)")
                
                # Create upvote map for weighting
                upvote_map = {comment["id"]: comment["upvotes"] for comment in comments}
                
                # Process based on intent
                if intent_schema.intent == "RANKING":
                    ranking = rank_entities(annos, upvote_map, intent_schema.entity_type, min_mentions=1)
                    
                    # Add quotes to ranking items
                    for item in ranking:
                        entity_comments = []
                        for comment in comments:
                            if item.name.lower() in comment["text"].lower():
                                entity_comments.append(comment["text"][:200] + "...")
                        item.quotes = entity_comments[:3]
                    
                    # Prepare ranking data for summary
                    ranking_data = []
                    for item in ranking:
                        ranking_data.append({
                            "name": item.name,
                            "overall_stars": item.overall_stars,
                            "mentions": item.mentions,
                            "quotes": item.quotes
                        })
                    
                    summary = llm_service.summarize_ranking_with_gpt(query, ranking_data)
                    
                    payload = {
                        "query": query,
                        "intent": intent_schema.intent,
                        "summary": summary,
                        "metadata": {"timestamp": time.time()},
                        "ranking": [
                            {
                                "name": item.name,
                                "overall_stars": item.overall_stars,
                                "aspect_scores": item.aspect_scores,
                                "mentions": item.mentions,
                                "confidence": item.confidence,
                                "quotes": item.quotes
                            }
                            for item in ranking
                        ]
                    }
                    
                elif intent_schema.intent == "SOLUTION":
                    clusters = defaultdict(list)
                    for anno in annos:
                        if anno.solution_key:
                            clusters[anno.solution_key].append(anno)
                    
                    solution_clusters = []
                    for cluster_key, cluster_annos in clusters.items():
                        if len(cluster_annos) >= 2:
                            cluster = {
                                "title": cluster_key,
                                "steps": [],
                                "caveats": [],
                                "evidence_count": len(cluster_annos)
                            }
                            solution_clusters.append(cluster)
                    
                    summary = llm_service.summarize_solutions_with_gpt(query, solution_clusters)
                    
                    payload = {
                        "query": query,
                        "intent": intent_schema.intent,
                        "summary": summary,
                        "metadata": {"timestamp": time.time()},
                        "solutions": solution_clusters
                    }
                    
                else:  # GENERIC
                    # Get platform counts for bias calibration
                    platform_counts = {}
                    if len(selected_platforms) > 1 or (len(selected_platforms) == 1 and selected_platforms[0] != Platform.REDDIT):
                        # Cross-platform search - get counts from results
                        platform_counts = {platform_name: len(platform_reviews) 
                                         for platform_name, platform_reviews in cross_platform_results["platform_results"].items()}
                    else:
                        # Reddit-only search
                        platform_counts = {"reddit": len(comments)}
                    
                    overall, aspect_averages = aggregate_generic(intent_schema.aspects, annos, upvote_map, platform_counts)
                    
                    # Select high-quality quotes using GPT sentiment analysis
                    quotes = []
                    positive_annos = [anno for anno in annos if anno.overall_score >= 3.5]  # High sentiment threshold
                    sorted_annos = sorted(positive_annos, key=lambda a: a.overall_score, reverse=True)
                    
                    for anno in sorted_annos[:8]:  # Top 8 positive comments
                        comment = next((c for c in comments if c["id"] == anno.comment_id), None)
                        if comment:
                            quotes.append(comment["text"][:200] + "...")
                    
                    summary = llm_service.summarize_generic_with_gpt(query, aspect_averages, overall, quotes)
                    
                    payload = {
                        "query": query,
                        "intent": intent_schema.intent,
                        "summary": summary,
                        "metadata": {"timestamp": time.time()},
                        "overall": overall,
                        "aspects": aspect_averages,
                        "quotes": quotes
                    }
                
                # Display results
                st.header(f"Analysis Results for '{query}'")
                
                # Show raw comment data for inspection
                if st.checkbox("🔧 Debug: show first 3 raw comments"):
                    import pandas as pd
                    df = pd.DataFrame(reviews[:3])
                    st.dataframe(df[["id","author","upvotes","permalink","text"]])
                
                # Key metrics - simplified for all intents
                st.subheader("Key Metrics")
                
                meaningful_reviews = [r for r in reviews if len((r.get("text") or "")) >= 100]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", len(reviews))
                with col2:
                    st.metric("Meaningful Reviews", len(meaningful_reviews))
                with col3:
                    if intent_schema.intent == "RANKING":
                        st.metric("Ranked Entities", len(payload.get("ranking", [])))
                    elif intent_schema.intent == "SOLUTION":
                        st.metric("Solution Clusters", len(payload.get("solutions", [])))
                    else:
                        st.metric("Analysis Complete", "✅")
                with col4:
                    if intent_schema.intent == "GENERIC":
                        overall_rating = payload.get("overall", 3.0)
                        st.metric("Overall Rating", f"{overall_rating:.1f}/5")
                    else:
                        st.metric("Analysis Complete", "")
                
                # Add prominent rating display for generic search
                if intent_schema.intent == "GENERIC":
                    st.subheader(" Overall Assessment")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        overall_rating = payload.get("overall", 3.0)
                        st.metric("Overall Rating", f"{overall_rating:.1f}/5")
                        if overall_rating >= 4.0:
                            st.success("Highly Recommended")
                        elif overall_rating >= 3.0:
                            st.info(" Generally Positive")
                        else:
                            st.warning(" Mixed Reviews")
                    
                    with col2:
                        st.subheader("Aspect Breakdown")
                        aspect_scores = payload.get("aspects", {})
                        if aspect_scores:
                            for aspect, score in list(aspect_scores.items())[:4]:  # Show top 4 aspects
                                st.progress(score / 5.0)
                                st.caption(f"{aspect}: {score:.1f}/5")
                        else:
                            st.info("Aspect scores not available")
                
                # Detailed Summary Section
                st.subheader(" Detailed Summary")
                st.write(payload["summary"])
                
                # Display results based on intent
                if intent_schema.intent == "RANKING":
                    st.subheader("🏆 Ranking Results")
                    if payload["ranking"]:
                        # Show top entities for ranking queries (configurable)
                        for i, item in enumerate(payload["ranking"][:SearchConstants.MAX_ENTITIES_TO_DISPLAY], 1):
                            st.write(f"**{i}. {item['name']}** ({item['mentions']} mentions)")
                            
                            # Show quotes
                            if item['quotes']:
                                with st.expander(f"Key insights about {item['name']}"):
                                    for quote in item['quotes'][:3]:
                                        st.write(f"• {quote}")
                            st.write("---")
                    else:
                        st.info("No ranked entities found. Try a more specific query or increase the comment limit.")
                        
                elif intent_schema.intent == "SOLUTION":
                    st.subheader("🔧 Solution Clusters")
                    if payload["solutions"]:
                        for i, cluster in enumerate(payload["solutions"], 1):
                            st.write(f"**{i}. {cluster['title']}** ({cluster['evidence_count']} comments)")
                            if cluster.get('steps'):
                                st.write("Steps:")
                                for step in cluster['steps']:
                                    st.write(f"  - {step}")
                            if cluster.get('caveats'):
                                st.write("Caveats:")
                                for caveat in cluster['caveats']:
                                    st.write(f"  - {caveat}")
                            st.write("---")
                    else:
                        st.info("No solution clusters found. Try a more specific query or increase the comment limit.")
                        
                else:  # GENERIC
                    # Show representative quotes
                    if payload.get("quotes"):
                        st.subheader("Key Insights")
                        for quote in payload["quotes"][:5]:
                            st.write(f"• {quote}")
                
                # Show raw comments for all intents
                st.subheader("Related Reviews")
                meaningful_reviews = [r for r in reviews if len((r.get("text") or "")) >= 100]
                
                # Sort by upvotes
                sorted_reviews = sorted(meaningful_reviews, key=lambda r: -(r.get("upvotes", 0) or 0))
                
                # Show first 5 reviews
                initial_reviews = sorted_reviews[:5]
                
                for i, review in enumerate(initial_reviews, 1):
                    link = review.get("url") or (f"https://reddit.com{review.get('permalink','')}" if review.get("permalink") else "")
                    st.write(f"**{i}. ↑{review.get('upvotes', 0)} · [{review.get('author','u/unknown')}]({link})**")
                    st.caption(_excerpt(review.get("text","")))
                    st.write("---")
                
                # Show more reviews if available
                if len(sorted_reviews) > 5:
                    with st.expander(f"View More Reviews ({len(sorted_reviews) - 5} additional)"):
                        for i, review in enumerate(sorted_reviews[5:], 6):
                            link = review.get("url") or (f"https://reddit.com{review.get('permalink','')}" if review.get("permalink") else "")
                            st.write(f"**{i}. ↑{review.get('upvotes', 0)} · [{review.get('author','u/unknown')}]({link})**")
                            st.caption(_excerpt(review.get("text","")))
                            st.write("---")
                
                # Display search time
                st.success(f" **Search completed in {search_time:.1f} seconds**")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {e}")
        st.info("Please try again with a different search term.")

# Popular searches
st.subheader("Popular Searches")
cols = st.columns(4, gap="large")

POPULAR = [
    {"title": "Top 10 NYC Restaurant", "cat": "Restaurants", "q": "top 10 nyc restaurant"},
    {"title": "Tesla Model Y", "cat": "Cars", "q": "Tesla Model Y"},
    {"title": "Nintendo Switch", "cat": "Tech", "q": "Nintendo Switch"},
    {"title": "Best Golf Course in Bay Area", "cat": "Golf", "q": "best golf course in bay area"},
]

for i, p in enumerate(POPULAR):
    with cols[i]:
        st.write(f"**{p['title']}** · {p['cat']}")
        if st.button(f'Use "{p["title"]}"', key=f"use_{i}", width='stretch'):
            st.session_state["query"] = p["q"]
            st.rerun()

def main():
    """Main function for Streamlit app."""
    pass  # The app runs automatically when imported