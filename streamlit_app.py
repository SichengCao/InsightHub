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
sys.path.append(str(Path(__file__).parent))

# Import from flat modules
from config import ensure_config_files
from reddit_client import RedditService
from llm import LLMServiceFactory
from scoring import aggregate_generic, rank_entities
from data_prep import export_to_json

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
ensure_config_files()

# Initialize services
reddit_service = RedditService()
llm_service = LLMServiceFactory.create()

# Main UI
st.title("📈 InsightHub — Review Analysis")
st.write("Analyze Reddit reviews with AI-powered sentiment and aspect scoring.")

# Sidebar for search
with st.sidebar:
    st.header("🔎 Search")
    
    # Search input
    query = st.text_input("Search Reviews", value=st.session_state.get("query", "Tesla Model Y"))
    
    # Limit slider
    limit = st.slider("Number of Comments", 10, 200, 50, step=10)
    
    # Analyze button
    run_analysis = st.button("📊 Analyze Reviews", width='stretch')

# Main content area
if run_analysis:
    try:
        with st.spinner("Analyzing reviews..."):
            # Show Reddit search plan (debug)
            with st.expander("🔎 Reddit search plan (debug)"):
                try:
                    # call the internal planner without hitting the network
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
            
            # Scrape reviews
            logger.info(f"Scraping Reddit for '{query}'...")
            reviews = reddit_service.scrape(query, limit)
            logger.info(f"Analyzing {len(reviews)} reviews...")
            
            # Show filtering transparency
            st.caption(f"🔍 Quality filtering applied: Using {len(reviews)} high-quality comments")
            
            if not reviews:
                st.warning("No reviews found! Try a different search term.")
            else:
                # Detect intent and generate schema
                intent_schema = llm_service.detect_intent_and_schema(query)
                st.info(f"🎯 Detected intent: **{intent_schema.intent}**")
                if intent_schema.entity_type:
                    st.info(f"📋 Entity type: **{intent_schema.entity_type}**")
                st.info(f"🔍 Aspects: {', '.join(intent_schema.aspects[:5])}{'...' if len(intent_schema.aspects) > 5 else ''}")
                
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
                with st.spinner("🤖 Annotating comments with GPT..."):
                    annos = llm_service.annotate_comments_with_gpt(comments, intent_schema.aspects, intent_schema.entity_type)
                
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
                    
                    summary = llm_service.summarize_generic_with_gpt(
                        query, 
                        {item.name: item.overall_stars for item in ranking[:5]}, 
                        sum(item.overall_stars for item in ranking[:5]) / len(ranking[:5]) if ranking else 3.0,
                        [quote for item in ranking[:3] for quote in item.quotes]
                    )
                    
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
                    
                    quotes = []
                    for comment in comments[:10]:
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
                st.header(f"📊 Analysis Results for '{query}'")
                
                # Debug mode to see raw comment data
                if st.checkbox("🔧 Debug: show first 3 raw comments"):
                    import pandas as pd
                    df = pd.DataFrame(reviews[:3])
                    st.dataframe(df[["id","author","upvotes","permalink","text"]])
                
                # Key metrics based on intent
                st.subheader("📈 Key Metrics")
                
                if intent_schema.intent == "RANKING":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        st.metric("Ranked Entities", len(payload["ranking"]))
                    with col3:
                        avg_rating = sum(item["overall_stars"] for item in payload["ranking"]) / len(payload["ranking"]) if payload["ranking"] else 0
                        st.metric("Average Rating", f"{avg_rating:.1f}/5")
                        
                elif intent_schema.intent == "SOLUTION":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        st.metric("Solution Clusters", len(payload["solutions"]))
                    with col3:
                        total_evidence = sum(cluster["evidence_count"] for cluster in payload["solutions"])
                        st.metric("Evidence Comments", total_evidence)
                        
                else:  # GENERIC
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        st.metric("Overall Rating", f"{payload['overall']:.1f}/5")
                    with col3:
                        st.metric("Aspects Analyzed", len(payload["aspects"]))
                    with col4:
                        meaningful_reviews = [r for r in reviews if len((r.get("text") or "")) >= 100]
                        st.metric("Meaningful Reviews", len(meaningful_reviews))
                
                # Detailed Summary Section
                st.subheader("📝 Detailed Summary")
                st.write(payload["summary"])
                
                # Display results based on intent
                if intent_schema.intent == "RANKING":
                    st.subheader("🏆 Ranking Results")
                    if payload["ranking"]:
                        for i, item in enumerate(payload["ranking"][:5], 1):
                            st.write(f"**{i}. {item['name']}**: {item['overall_stars']:.1f}/5 ({item['mentions']} mentions)")
                            
                            # Show aspect scores
                            if item['aspect_scores']:
                                st.write("Aspect scores:")
                                for aspect, score in item['aspect_scores'].items():
                                    color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
                                    st.write(f"  - {aspect}: {score:.1f}/5 {color}")
                            
                            # Show quotes
                            if item['quotes']:
                                with st.expander(f"Quotes about {item['name']}"):
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
                    st.subheader("🎯 Aspect Analysis")
                    if payload["aspects"]:
                        st.write("**📊 Aspects (score):**")
                        for aspect, score in payload["aspects"].items():
                            color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
                            st.write(f"- **{aspect}**: {score:.1f}/5 {color}")
                    
                    # Show representative quotes
                    if payload.get("quotes"):
                        st.subheader("💬 Representative Quotes")
                        for quote in payload["quotes"][:5]:
                            st.write(f"• {quote}")
                
                # Show raw comments for all intents
                st.subheader("💬 Raw Reddit Comments")
                meaningful_reviews = [r for r in reviews if len((r.get("text") or "")) >= 100]
                
                # Sort by upvotes
                sorted_reviews = sorted(meaningful_reviews, key=lambda r: -(r.get("upvotes", 0) or 0))[:10]
                
                for i, review in enumerate(sorted_reviews, 1):
                    link = review.get("url") or (f"https://reddit.com{review.get('permalink','')}" if review.get("permalink") else "")
                    st.write(f"**{i}. ↑{review.get('upvotes', 0)} · [{review.get('author','u/unknown')}]({link})**")
                    st.caption(_excerpt(review.get("text","")))
                    st.write("---")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {e}")
        st.info("Please try again with a different search term.")

# Popular searches
st.subheader("🔥 Popular Searches")
cols = st.columns(4, gap="large")

POPULAR = [
    {"title": "iPhone 15", "cat": "Phones", "q": "iPhone 15"},
    {"title": "Tesla Model Y", "cat": "Cars", "q": "Tesla Model Y"},
    {"title": "Nintendo Switch", "cat": "Tech", "q": "Nintendo Switch"},
    {"title": "MacBook Pro", "cat": "Tech", "q": "MacBook Pro"},
]

for i, p in enumerate(POPULAR):
    with cols[i]:
        st.write(f"**{p['title']}** · {p['cat']}")
        if st.button(f'Use "{p["title"]}"', key=f"use_{i}", width='stretch'):
            st.session_state["query"] = p["q"]
            st.rerun()