"""Simple Streamlit UI for InsightHub."""

import streamlit as st
import logging
import hashlib
import re
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from insighthub.config import ensure_config_files
from insighthub.services.reddit_client import RedditService
from insighthub.services.llm import LLMServiceFactory
from insighthub.analysis.sentiment import VADERSentimentAnalyzer
from insighthub.analysis.aspect import YAMLAspectDetector
from insighthub.analysis.scoring import create_analysis_summary, compute_aspect_scores, aspect_aggregate, overall_rating_wilson, aspect_rating_wilson, crowd_trust_stars
from insighthub.reporting.data_prep import build_summary_payload

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
sentiment_analyzer = VADERSentimentAnalyzer()
aspect_detector = YAMLAspectDetector()

# Evidence rendering functions
def render_section_with_evidence(title, items, reviews_by_id):
    """Render pros/cons with evidence expanders."""
    st.subheader(title)
    for item in items:
        text = item.get("text") or ""
        ids = item.get("ids") or []
        with st.expander(text):
            if not ids:
                st.caption("No explicit evidence ids provided.")
            else:
                st.caption(f"Evidence comments: {len(ids)}")
                for rid in ids[:12]:
                    c = reviews_by_id.get(rid)
                    if c:
                        link = getattr(c, "post_url", getattr(c, "permalink", "")) or ""
                        up = getattr(c, "upvotes", getattr(c, "score", 0)) or 0
                        st.markdown(f"- [{rid}]({link})  ↑{up}")
                    else:
                        st.markdown(f"- {rid}")

def render_coverage_meter(coverage_ids, total_comments):
    """Render coverage meter."""
    backed = len(set(coverage_ids or []))
    total = max(total_comments, 1)
    pct = int(100 * backed / total)
    st.write(f"**Coverage:** {backed} / {total} comments referenced ({pct}%)")

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
            # Scrape reviews
            logger.info(f"Scraping Reddit for '{query}'...")
            reviews = reddit_service.scrape(query, limit)
            logger.info(f"Analyzing {len(reviews)} reviews...")
            
            # Show filtering transparency
            st.caption(f"🔍 Quality filtering applied: Using {len(reviews)} high-quality comments")
            
            # Analyze sentiment
            sentiment_results = []
            for review in reviews:
                text = review.get("text") if isinstance(review, dict) else review.text
                result = sentiment_analyzer.analyze(text)
                # Store results in the review dict
                review["sentiment_compound"] = result['compound']
                review["sentiment_label"] = result['label']
                review["stars"] = result['stars']
                review["overall_rating"] = result['stars']  # Add overall_rating for UI display
                sentiment_results.append(result)
            
            # Detect aspects
            reviews_by_aspect = {}
            for review in reviews:
                text = review.get("text") if isinstance(review, dict) else review.text
                stars = review.get("stars") if isinstance(review, dict) else review.stars
                aspects = aspect_detector.detect_aspects(text, "tech_products")
                for aspect in aspects:
                    if aspect not in reviews_by_aspect:
                        reviews_by_aspect[aspect] = []
                    reviews_by_aspect[aspect].append({
                        'stars': stars,
                        'text': text
                    })
            
            aspect_scores = compute_aspect_scores(reviews_by_aspect)
            
            # Create summary
            summary = create_analysis_summary(query, reviews, sentiment_results, aspect_scores)
            
            # Generate pros/cons using map-reduce pipeline
            reduce_json = llm_service.summarize_comments_map_reduce(reviews, query)
            
            # Create reviews lookup by ID
            reviews_by_id = {r.get("id") if isinstance(r, dict) else r.id: r for r in reviews}
            
            # Build weighted aspects directly from Review.aspect_scores
            weighted_aspects = aspect_aggregate(reviews)
            
            # Build summary payload for UI
            summary_payload = build_summary_payload(query, reduce_json, reviews_by_id)
            
            # Show evidence coverage warning if needed
            cov = len(reduce_json.get("coverage_ids", []))
            total = len(reviews or [])
            if cov == 0 and total > 0:
                st.warning("Evidence coverage is 0% — showing raw top comments below. Try broadening the query or increasing comments.")
            
            # Fallback to old method if map-reduce fails
            if not reduce_json.get("pros") and not reduce_json.get("cons"):
                pros_cons = llm_service.generate_pros_cons(reviews, query)
            else:
                pros_cons = {"summary": "Analysis completed using evidence-first map-reduce pipeline."}
            
            # Display results
            st.header(f"📊 Analysis Results for '{query}'")
            
            # Debug mode to see raw comment data
            if st.checkbox("🔧 Debug: show first 3 raw comments"):
                import pandas as pd
                df = pd.DataFrame(reviews[:3])
                st.dataframe(df[["id","author","upvotes","permalink","text"]])
            
            # Key metrics
            st.subheader("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Rating", f"{summary.avg_stars:.1f}/5")
            
            with col2:
                st.metric("Total Reviews", summary.total)
            
            with col3:
                pos_pct = (summary.pos / summary.total * 100) if summary.total > 0 else 0
                st.metric("Positive", f"{pos_pct:.1f}%")
            
            with col4:
                neg_pct = (summary.neg / summary.total * 100) if summary.total > 0 else 0
                st.metric("Negative", f"{neg_pct:.1f}%")
            
            # Compute meaningful_reviews once and use that everywhere
            meaningful_reviews = [r for r in reviews if len((r.get("text") or "")) >= 100]
            st.write(f"**Total Reviews:** {len(reviews)}")
            st.write(f"**Meaningful Reviews (>100 chars):** {len(meaningful_reviews)}")
            
            # Only render one trust score
            try:
                crowd_stars = crowd_trust_stars(meaningful_reviews)
                st.markdown(f"🎯 Crowd-trust score: **{crowd_stars} / 5.0**")
            except Exception as e:
                logger.warning(f"Crowd trust scoring failed: {e}")
            
            # Tiny debug switch (helps confirm fields are real)
            if st.checkbox("🔧 Debug: first 3 raw comments"):
                import pandas as pd
                df = pd.DataFrame(meaningful_reviews[:3])
                st.dataframe(df[["id","author","upvotes","url","permalink","text"]])
            
            # Detailed Summary Section
            st.subheader("📝 Detailed Summary")
            
            # Get detailed summary from GPT
            detailed_summary = pros_cons.get('summary', '')
            if detailed_summary:
                st.write(f"**Comprehensive Analysis:**")
                st.write(detailed_summary)
            else:
                st.write(f"**Analysis Overview:**")
                st.write(f"Analyzed {len(reviews)} reviews from Reddit. Sentiment distribution: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative, {100-pos_pct-neg_pct:.1f}% neutral.")
                st.write(f"**Overall Rating:** {summary.avg_stars:.1f}/5 stars based on comprehensive sentiment analysis.")
            
            # Real Reddit Comments Section
            st.subheader("💬 Real Reddit Comments")
            
            # Pick by score and stars; use meaningful_reviews for better quality
            pos = [r for r in meaningful_reviews if float(r.get("stars", r.get("overall_rating", 0)) or 0) >= 4.0]
            neg = [r for r in meaningful_reviews if float(r.get("stars", r.get("overall_rating", 0)) or 0) <= 2.0]
            
            # Sort by upvotes then rating
            pos = sorted(pos, key=lambda r: (-(r.get("upvotes", r.get("score", 0)) or 0), -(r.get("stars", r.get("overall_rating", 0)) or 0)))[:12]
            neg = sorted(neg, key=lambda r: (-(r.get("upvotes", r.get("score", 0)) or 0),  (r.get("stars", r.get("overall_rating", 0)) or 0)))[:12]
            
            # Deduplicate to avoid near-duplicates
            pos = _dedupe_keep_order(pos, key=lambda r: _sig(r.get("text","")))[:5]
            neg = _dedupe_keep_order(neg, key=lambda r: _sig(r.get("text","")))[:5]
            
            st.markdown("#### 🟢 Top 5 Positive Comments")
            for r in pos:
                link = r.get("url") or (f"https://reddit.com{r.get('permalink','')}" if r.get("permalink") else "")
                st.write(f"**⭐ {float(r.get('stars', r.get('overall_rating', 0))):.1f}/5** · ↑{r.get('upvotes', r.get('score', 0))} · [{r.get('author','u/unknown')}]({link})")
                st.caption(_excerpt(r.get("text","")))

            st.markdown("#### 🔴 Top 5 Negative Comments")
            for r in neg:
                link = r.get("url") or (f"https://reddit.com{r.get('permalink','')}" if r.get("permalink") else "")
                st.write(f"**⭐ {float(r.get('stars', r.get('overall_rating', 0))):.1f}/5** · ↑{r.get('upvotes', r.get('score', 0))} · [{r.get('author','u/unknown')}]({link})")
                st.caption(_excerpt(r.get("text","")))
            
            # Show expandable full comments
            with st.expander("📖 View Full Comments", expanded=False):
                st.write("**Top 5 Positive Comments:**")
                for i, review in enumerate(pos[:5]):
                    st.write(f"**Comment {i + 1}:**")
                    st.write(f"**Text:** {getattr(review, 'text', getattr(review, 'body', ''))}")
                    st.write(f"**Rating:** {float(getattr(review, 'overall_rating', 0)):.1f}/5 stars")
                    st.write("---")
                
                st.write("**Top 5 Negative Comments:**")
                for i, review in enumerate(neg[:5]):
                    st.write(f"**Comment {i + 1}:**")
                    st.write(f"**Text:** {getattr(review, 'text', getattr(review, 'body', ''))}")
                    st.write(f"**Rating:** {float(getattr(review, 'overall_rating', 0)):.1f}/5 stars")
                    st.write("---")
            
            # Coverage meter
            st.write(f"Coverage: {cov} / {total} comments referenced ({cov/total*100:.0f}%)")
            
            # Evidence-based Pros & Cons (only show if coverage > 0%)
            if cov > 0 and (reduce_json.get("pros") or reduce_json.get("cons")):
                st.subheader("✅ Pros")
                for item in reduce_json.get("pros", [])[:5]:
                    st.write(f"• {item.get('text', '')}")
                    if item.get('ids'):
                        st.caption(f"Sources: {', '.join(item['ids'][:3])}")
                
                st.subheader("❌ Cons")
                for item in reduce_json.get("cons", [])[:5]:
                    st.write(f"• {item.get('text', '')}")
                    if item.get('ids'):
                        st.caption(f"Sources: {', '.join(item['ids'][:3])}")
            elif cov == 0:
                st.info("⚠️ Evidence coverage is 0% — LLM-generated pros/cons hidden. Showing raw top comments above.")
            else:
                # Fallback to old pros/cons display
                st.subheader("✅ Pros & ❌ Cons")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**✅ Positive Aspects:**")
                    # Try structured bullets first
                    if summary_payload.get("pros"):
                        for item in summary_payload["pros"][:5]:
                            st.write(f"• {item['text']}")
                    else:
                        pros_data = pros_cons.get('pros', 'No pros identified.')
                        if isinstance(pros_data, list):
                            pros_list = pros_data
                        else:
                            pros_list = pros_data.split('. ')
                        for pro in pros_list[:5]:
                            if pro and str(pro).strip():
                                st.write(f"• {str(pro).strip()}")
                
                with col2:
                    st.write("**❌ Negative Aspects:**")
                    # Try structured bullets first
                    if summary_payload.get("cons"):
                        for item in summary_payload["cons"][:5]:
                            st.write(f"• {item['text']}")
                    else:
                        cons_data = pros_cons.get('cons', 'No cons identified.')
                        if isinstance(cons_data, list):
                            cons_list = cons_data
                        else:
                            cons_list = cons_data.split('. ')
                        for con in cons_list[:5]:
                            if con and str(con).strip():
                                st.write(f"• {str(con).strip()}")
            
            # Aspect Analysis - Prefer structured aspects from map-reduce, fallback to weighted
            st.subheader("🎯 Aspect Analysis")
            
            # Prefer structured aspects from the map-reduce output; if missing, use weighted ones
            ui_aspects = summary_payload.get("aspects") or [
                {"name": a["name"], "score": a["score"], "count": a.get("count", 0)}
                for a in weighted_aspects
            ]
            
            if ui_aspects:
                st.write("**📊 Aspects (score, count):**")
                for a in ui_aspects[:8]:
                    score = a.get("score", 3.0)
                    count = a.get("count", 0)
                    color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
                    st.write(f"- **{a['name']}**: {score} ⭐ · n={count} {color}")
            elif aspect_scores:
                # Fallback to old aspect analysis
                sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                st.write("Top 5 most relevant aspects based on review analysis:")
                
                for i, (aspect, rating) in enumerate(sorted_aspects, 1):
                    color = "🟢" if rating >= 4 else "🟡" if rating >= 3 else "🔴"
                    st.write(f"{i}. **{aspect.title()}**: {rating:.1f}/5 {color}")
            else:
                st.write("No specific aspects detected in the reviews.")
            
            # Trust-weighted aspect scores (Wilson LB)
            try:
                trust_aspects = aspect_rating_wilson(reviews)
                if trust_aspects:
                    with st.expander("🎯 Trust-weighted aspect scores (Wilson LB)"):
                        st.write("Statistically robust aspect scores weighted by upvotes and recency:")
                        for a in trust_aspects[:8]:
                            st.write(f"- **{a['name']}**: {a['score']} ⭐ · eff n={a['count']}")
            except Exception as e:
                logger.warning(f"Trust-weighted aspects failed: {e}")
            
            # Representative Quotes
            quotes = summary_payload.get("quotes", [])
            if quotes:
                st.subheader("💬 Representative Quotes")
                st.write("Key quotes from the analysis:")
                for quote in quotes[:8]:  # Show up to 8 quotes
                    author = quote.get("author", "Unknown")
                    upvotes = quote.get("upvotes", 0)
                    permalink = quote.get("permalink", "")
                    quote_text = quote.get("quote", "")
                    
                    if permalink:
                        st.markdown(f"**{author}** ↑{upvotes}: [{quote_text[:100]}...]({permalink})")
                    else:
                        st.markdown(f"**{author}** ↑{upvotes}: {quote_text[:200]}{'...' if len(quote_text) > 200 else ''}")
            
            # Statistics Summary
            st.subheader("📊 Statistics Summary")
            st.write(f"**Total Reviews Analyzed:** {len(reviews)}")
            st.write(f"**Meaningful Reviews (>100 chars):** {meaningful_count}")
            st.write(f"**Sentiment Distribution:**")
            st.write(f"- Positive: {summary.pos} ({pos_pct:.1f}%)")
            st.write(f"- Negative: {summary.neg} ({neg_pct:.1f}%)")
            st.write(f"- Neutral: {summary.neu} ({100-pos_pct-neg_pct:.1f}%)")
            st.write(f"**Average Rating:** {summary.avg_stars:.1f}/5")
            if aspect_scores:
                top_aspect = max(aspect_scores.items(), key=lambda x: x[1])
                st.write(f"**Top Aspect:** {top_aspect[0].title()} ({top_aspect[1]:.1f}/5)")
                st.write(f"**Aspects Analyzed:** {len(aspect_scores)}")
    
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