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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tech-style custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Orbitron:wght@500;700&display=swap');
    
    .stApp { background: linear-gradient(180deg, #0a0e17 0%, #0d1321 50%, #0a0e17 100%); }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }
    
    .insighthub-hero {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    }
    .insighthub-hero h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .insighthub-hero p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #8b9dc3;
        margin: 0;
    }
    
    .popular-card {
        background: linear-gradient(145deg, rgba(18, 24, 42, 0.95) 0%, rgba(10, 14, 23, 0.98) 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 1px rgba(0, 212, 255, 0.2);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .popular-card:hover { border-color: rgba(0, 212, 255, 0.5); box-shadow: 0 0 24px rgba(0, 212, 255, 0.15); }
    .popular-card img { border-radius: 8px; width: 100%; height: 120px; object-fit: cover; }
    .popular-card .cat { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #00d4ff; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }
    .popular-card .title { font-family: 'Orbitron', sans-serif; font-weight: 600; color: #e4e8f0; font-size: 0.95rem; margin: 0.25rem 0 0.5rem 0; }
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1321 0%, #0a0e17 100%); }
    [data-testid="stSidebar"] .stMarkdown { color: #8b9dc3; }
    [data-testid="metric-value"] { font-family: 'Orbitron', sans-serif; color: #00d4ff !important; }
    h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #e4e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize services
reddit_service = RedditService()
llm_service = LLMServiceFactory.create()
cross_platform_manager = CrossPlatformManager()

# Main UI — hero
st.markdown("""
<div class="insighthub-hero">
    <h1>◈ INSIGHTHUB</h1>
    <p>AI-powered review analysis across Reddit & YouTube · Sentiment & aspect scoring</p>
</div>
""", unsafe_allow_html=True)

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
                
                # Show platform breakdown
                with st.expander("📊 Platform Results"):
                    for platform_name, platform_reviews in cross_platform_results["platform_results"].items():
                        st.write(f"**{platform_name.title()}**: {len(platform_reviews)} reviews")
                
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
                    overall, aspect_averages = aggregate_generic(intent_schema.aspects, annos, upvote_map)
                    
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
                    st.subheader("Overall Assessment")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        overall_rating = payload.get("overall", 3.0)
                        st.metric("Overall Rating", f"{overall_rating:.1f}/5")
                        if overall_rating >= 4.0:
                            st.success("Highly Recommended")
                        elif overall_rating >= 3.0:
                            st.info("Generally Positive")
                        else:
                            st.warning("Mixed Reviews")
                    
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
                st.subheader("Detailed Summary")
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

# Popular searches — cards with images
st.markdown("---")
st.subheader("🔮 Recommended Searches")
cols = st.columns(4, gap="large")

POPULAR = [
    {
        "title": "Top 10 NYC Restaurant",
        "cat": "Restaurants",
        "q": "top 10 nyc restaurant",
        "image": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&q=80",
    },
    {
        "title": "Tesla Model Y",
        "cat": "Cars",
        "q": "Tesla Model Y",
        "image": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=400&q=80",
    },
    {
        "title": "Nintendo Switch",
        "cat": "Tech",
        "q": "Nintendo Switch",
        "image": "https://images.unsplash.com/photo-1578303512597-81e6cc155b3e?w=400&q=80",
    },
    {
        "title": "Best Golf Course in Bay Area",
        "cat": "Golf",
        "q": "best golf course in bay area",
        "image": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=400&q=80",
    },
]

for i, p in enumerate(POPULAR):
    with cols[i]:
        st.markdown(
            f"""
            <div class="popular-card">
                <img src="{p['image']}" alt="{p['title']}" />
                <p class="cat">{p['cat']}</p>
                <p class="title">{p['title']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Use \"{p['title']}\"", key=f"use_{i}", use_container_width=True):
            st.session_state["query"] = p["q"]
            st.rerun()

def main():
    """Main function for Streamlit app."""
    pass  # The app runs automatically when imported