"""Simple Streamlit UI for InsightHub."""

import streamlit as st
import logging
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from insighthub.config import ensure_config_files
from insighthub.services.reddit_client import RedditService
from insighthub.services.llm import LLMServiceFactory
from insighthub.analysis.sentiment import VADERSentimentAnalyzer
from insighthub.analysis.aspect import YAMLAspectDetector
from insighthub.analysis.scoring import create_analysis_summary, compute_aspect_scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
            # Analyze sentiment
            sentiment_results = []
            for review in reviews:
                result = sentiment_analyzer.analyze(review.text)
                review.sentiment_compound = result['compound']
                review.sentiment_label = result['label']
                review.stars = result['stars']
                sentiment_results.append(result)
            
            # Detect aspects
            reviews_by_aspect = {}
            for review in reviews:
                aspects = aspect_detector.detect_aspects(review.text, "tech_products")
                for aspect in aspects:
                    if aspect not in reviews_by_aspect:
                        reviews_by_aspect[aspect] = []
                    reviews_by_aspect[aspect].append({
                        'stars': review.stars,
                        'text': review.text
                    })
            
            aspect_scores = compute_aspect_scores(reviews_by_aspect)
            
            # Create summary
            summary = create_analysis_summary(query, reviews, sentiment_results, aspect_scores)
            
            # Generate pros/cons
            pros_cons = llm_service.generate_pros_cons(reviews, query)
            
            # Display results
            st.header(f"📊 Analysis Results for '{query}'")
            
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
            
            # Individual Comments Section - Top 5 positive and 5 negative
            st.subheader("💬 Individual Comments Analysis")
            
            # Filter for meaningful comments (longer than 100 characters)
            meaningful_reviews = [r for r in reviews if len(r.text) > 100]
            
            # Separate positive and negative comments
            positive_reviews = [r for r in meaningful_reviews if r.sentiment_label == 'POSITIVE']
            negative_reviews = [r for r in meaningful_reviews if r.sentiment_label == 'NEGATIVE']
            
            # Sort by rating for best examples
            positive_reviews.sort(key=lambda x: x.stars, reverse=True)
            negative_reviews.sort(key=lambda x: x.stars)  # Lowest ratings first
            
            st.write(f"Showing top 5 positive and 5 negative comments from {len(meaningful_reviews)} meaningful reviews")
            
            # Display top 5 positive comments
            st.write("**🟢 Top 5 Positive Comments:**")
            pos_col1, pos_col2 = st.columns(2)
            
            for i, review in enumerate(positive_reviews[:5]):
                col = pos_col1 if i % 2 == 0 else pos_col2
                with col:
                    # Create small block design
                    st.markdown(f"""
                    <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:8px;padding:12px;margin:8px 0">
                        <div style="font-weight:600;color:#22c55e;margin-bottom:8px">
                            ⭐ {review.stars:.1f}/5 - {review.sentiment_label}
                        </div>
                        <div style="font-size:14px;color:#374151;line-height:1.4;margin-bottom:8px">
                            {review.text[:150]}{'...' if len(review.text) > 150 else ''}
                        </div>
                        <div style="font-size:12px;color:#6b7280">
                            {len(review.text)} chars
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display top 5 negative comments
            st.write("**🔴 Top 5 Negative Comments:**")
            neg_col1, neg_col2 = st.columns(2)
            
            for i, review in enumerate(negative_reviews[:5]):
                col = neg_col1 if i % 2 == 0 else neg_col2
                with col:
                    # Create small block design
                    st.markdown(f"""
                    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:12px;margin:8px 0">
                        <div style="font-weight:600;color:#ef4444;margin-bottom:8px">
                            ⭐ {review.stars:.1f}/5 - {review.sentiment_label}
                        </div>
                        <div style="font-size:14px;color:#374151;line-height:1.4;margin-bottom:8px">
                            {review.text[:150]}{'...' if len(review.text) > 150 else ''}
                        </div>
                        <div style="font-size:12px;color:#6b7280">
                            {len(review.text)} chars
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show expandable full comments
            with st.expander("📖 View Full Comments", expanded=False):
                st.write("**Top 5 Positive Comments:**")
                for i, review in enumerate(positive_reviews[:5]):
                    st.write(f"**Comment {i + 1}:**")
                    st.write(f"**Text:** {review.text}")
                    st.write(f"**Rating:** {review.stars:.1f}/5 stars")
                    st.write("---")
                
                st.write("**Top 5 Negative Comments:**")
                for i, review in enumerate(negative_reviews[:5]):
                    st.write(f"**Comment {i + 1}:**")
                    st.write(f"**Text:** {review.text}")
                    st.write(f"**Rating:** {review.stars:.1f}/5 stars")
                    st.write("---")
            
            # Pros & Cons
            st.subheader("✅ Pros & ❌ Cons")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Positive Aspects:**")
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
                cons_data = pros_cons.get('cons', 'No cons identified.')
                if isinstance(cons_data, list):
                    cons_list = cons_data
                else:
                    cons_list = cons_data.split('. ')
                for con in cons_list[:5]:
                    if con and str(con).strip():
                        st.write(f"• {str(con).strip()}")
            
            # Aspect Analysis - Top 5 aspects
            st.subheader("🎯 Top 5 Aspect Analysis")
            if aspect_scores:
                # Sort aspects by rating and take top 5
                sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                st.write("Top 5 most relevant aspects based on review analysis:")
                
                for i, (aspect, rating) in enumerate(sorted_aspects, 1):
                    # Create a progress bar for visual appeal
                    progress = rating / 5.0
                    color = "🟢" if rating >= 4 else "🟡" if rating >= 3 else "🔴"
                    st.write(f"{i}. **{aspect.title()}**: {rating:.1f}/5 {color}")
            else:
                st.write("No specific aspects detected in the reviews.")
            
            # Statistics Summary
            st.subheader("📊 Statistics Summary")
            st.write(f"**Total Reviews Analyzed:** {len(reviews)}")
            st.write(f"**Meaningful Reviews (>100 chars):** {len(meaningful_reviews)}")
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