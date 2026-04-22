"""Streamlit UI for InsightHub — production-ready review analysis platform."""

import streamlit as st
import logging
import hashlib
import re
import time
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from insighthub.core.config import settings
from insighthub.core.constants import SearchConstants
from insighthub.services.reddit_client import RedditService
from insighthub.services.llm import LLMServiceFactory
from insighthub.services.cross_platform_manager import CrossPlatformManager
from insighthub.core.cross_platform_models import Platform, QueryIntent
from insighthub.core.scoring import aggregate_generic, rank_entities
from insighthub.utils.data_prep import export_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _excerpt(s, n=240):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _sig(s: str) -> str:
    s = re.sub(r"https?://\S+", "", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
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


def _score_color(score: float, max_score: float = 5.0) -> str:
    r = score / max_score
    if r >= 0.72:
        return "#22c55e"
    if r >= 0.50:
        return "#f59e0b"
    return "#ef4444"


def _stars(score: float, max_score: float = 5.0) -> str:
    filled = round(score / max_score * 5)
    return "★" * filled + "☆" * (5 - filled)


def _aspect_bar(label: str, score: float, max_score: float = 5.0) -> str:
    pct = score / max_score * 100
    color = _score_color(score, max_score)
    return f"""
<div class="ih-aspect-row">
  <span class="ih-aspect-label">{label.replace("_"," ").title()}</span>
  <div class="ih-aspect-bar-bg">
    <div class="ih-aspect-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
  </div>
  <span class="ih-aspect-score">{score:.1f}</span>
</div>"""


def _metric_chip(value: str, label: str) -> str:
    return f"""
<div class="ih-metric-chip">
  <div class="ih-metric-value">{value}</div>
  <div class="ih-metric-label">{label}</div>
</div>"""


def _quote_block(text: str) -> str:
    return f'<div class="ih-quote">{text}</div>'


def _review_card(rank: int, author: str, upvotes: int, text: str, link: str) -> str:
    author_html = f'<a href="{link}" target="_blank" style="color:#818cf8;text-decoration:none">{author}</a>' if link else author
    return f"""
<div class="ih-review-card">
  <div class="ih-review-author">{author_html}</div>
  <div class="ih-review-text">{text}</div>
  <div class="ih-review-upvotes">▲ {upvotes}</div>
</div>"""


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InsightHub — AI Review Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, .stApp { background:#0c0e14 !important; font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.block-container { padding:2rem 2.5rem !important; max-width:1280px !important; }
h1,h2,h3 { font-family:'Inter',sans-serif !important; font-weight:700 !important; color:#f1f5f9 !important; }

/* ── Hero ── */
.ih-hero { text-align:center; padding:2.5rem 1rem 1.5rem; }
.ih-logo { display:flex; align-items:center; justify-content:center; gap:.6rem; margin-bottom:.6rem; }
.ih-logo-mark {
  width:38px; height:38px;
  background:linear-gradient(135deg,#6366f1,#a78bfa);
  border-radius:9px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.1rem; font-weight:700; color:#fff;
}
.ih-logo-name { font-size:1.65rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em; }
.ih-logo-name span { color:#6366f1; }
.ih-tagline { font-size:.9rem; color:#475569; max-width:460px; margin:0 auto; line-height:1.6; }

/* ── Search strip ── */
.ih-search-wrap {
  background:rgba(255,255,255,.03);
  border:1px solid rgba(99,102,241,.2);
  border-radius:14px;
  padding:1.1rem 1.4rem;
  margin-bottom:1.5rem;
}
.ih-search-label { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.09em; color:#475569; font-family:'JetBrains Mono',monospace; margin-bottom:.5rem; }

/* ── Metric chips ── */
.ih-chips { display:flex; gap:.75rem; flex-wrap:wrap; margin:1rem 0; }
.ih-metric-chip {
  flex:1; min-width:110px;
  background:rgba(99,102,241,.08);
  border:1px solid rgba(99,102,241,.18);
  border-radius:10px;
  padding:.8rem 1rem;
  text-align:center;
}
.ih-metric-value { font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:600; color:#818cf8; line-height:1; margin-bottom:.2rem; }
.ih-metric-label { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; }

/* ── Intent badge ── */
.ih-badge {
  display:inline-block; font-family:'JetBrains Mono',monospace;
  font-size:.7rem; font-weight:500; padding:.2rem .65rem;
  border-radius:20px; letter-spacing:.05em; text-transform:uppercase;
}
.ih-badge-ranking { background:rgba(99,102,241,.14); color:#818cf8; border:1px solid rgba(99,102,241,.3); }
.ih-badge-solution { background:rgba(34,197,94,.1); color:#4ade80; border:1px solid rgba(34,197,94,.3); }
.ih-badge-generic  { background:rgba(245,158,11,.1); color:#fbbf24; border:1px solid rgba(245,158,11,.3); }

/* ── Section title ── */
.ih-section-title {
  font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em;
  color:#475569; font-family:'JetBrains Mono',monospace;
  margin:1.6rem 0 .8rem; padding-bottom:.4rem;
  border-bottom:1px solid rgba(255,255,255,.05);
}

/* ── Summary card ── */
.ih-summary-card {
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.07);
  border-radius:12px;
  padding:1.1rem 1.3rem;
  font-size:.92rem; color:#cbd5e1; line-height:1.7;
  margin-bottom:1rem;
}

/* ── Big rating ── */
.ih-big-rating {
  text-align:center;
  background:rgba(99,102,241,.07);
  border:1px solid rgba(99,102,241,.2);
  border-radius:12px;
  padding:1.5rem 1rem;
}
.ih-big-rating .score { font-family:'JetBrains Mono',monospace; font-size:3rem; font-weight:700; color:#6366f1; line-height:1; }
.ih-big-rating .denom { font-size:.95rem; color:#475569; }
.ih-big-rating .stars { font-size:1.05rem; color:#f59e0b; letter-spacing:.05em; margin-top:.3rem; }
.ih-big-rating .verdict { font-size:.82rem; font-weight:500; margin-top:.5rem; }
.verdict-good { color:#4ade80; }
.verdict-mid  { color:#fbbf24; }
.verdict-low  { color:#f87171; }

/* ── Aspect bars ── */
.ih-aspect-row { display:flex; align-items:center; gap:.7rem; margin-bottom:.5rem; }
.ih-aspect-label { font-size:.8rem; color:#94a3b8; min-width:130px; }
.ih-aspect-bar-bg { flex:1; height:5px; background:rgba(255,255,255,.06); border-radius:3px; overflow:hidden; }
.ih-aspect-bar-fill { height:100%; border-radius:3px; }
.ih-aspect-score { font-size:.78rem; font-family:'JetBrains Mono',monospace; color:#64748b; min-width:2.5rem; text-align:right; }

/* ── Rank cards ── */
.ih-rank-card {
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.06);
  border-radius:11px;
  padding:1rem 1.2rem;
  margin-bottom:.55rem;
  display:flex; align-items:flex-start; gap:1rem;
  transition:border-color .15s;
}
.ih-rank-card:hover { border-color:rgba(99,102,241,.4); }
.ih-rank-num { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:600; color:#334155; min-width:2rem; padding-top:.1rem; }
.ih-rank-num.gold { color:#f59e0b; }
.ih-rank-num.silver { color:#94a3b8; }
.ih-rank-num.bronze { color:#b45309; }
.ih-rank-name { font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:.15rem; }
.ih-rank-stars { color:#f59e0b; font-size:.9rem; letter-spacing:.05em; }
.ih-rank-meta { font-size:.75rem; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:.15rem; }

/* ── Solution cards ── */
.ih-solution-card {
  background:rgba(34,197,94,.04);
  border:1px solid rgba(34,197,94,.15);
  border-radius:10px;
  padding:1rem 1.2rem;
  margin-bottom:.6rem;
}
.ih-solution-title { font-size:.95rem; font-weight:600; color:#f1f5f9; margin-bottom:.3rem; }
.ih-solution-badge { font-size:.72rem; color:#4ade80; font-family:'JetBrains Mono',monospace; }

/* ── Quote block ── */
.ih-quote {
  border-left:2px solid rgba(99,102,241,.5);
  padding:.45rem .75rem;
  margin-bottom:.45rem;
  background:rgba(255,255,255,.018);
  border-radius:0 6px 6px 0;
  font-size:.86rem; color:#94a3b8; font-style:italic; line-height:1.55;
}

/* ── Review cards ── */
.ih-review-card {
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.055);
  border-left:3px solid rgba(99,102,241,.35);
  border-radius:0 8px 8px 0;
  padding:.75rem 1rem;
  margin-bottom:.5rem;
}
.ih-review-author { font-size:.78rem; font-weight:600; color:#818cf8; font-family:'JetBrains Mono',monospace; margin-bottom:.3rem; }
.ih-review-text { font-size:.88rem; color:#94a3b8; line-height:1.55; }
.ih-review-meta { font-size:.72rem; color:#334155; font-family:'JetBrains Mono',monospace; margin-top:.3rem; }

/* ── Popular cards ── */
.ih-pop-card {
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.065);
  border-radius:12px;
  overflow:hidden;
  transition:border-color .2s, transform .15s;
}
.ih-pop-card:hover { border-color:rgba(99,102,241,.4); transform:translateY(-2px); }
.ih-pop-card img { width:100%; height:108px; object-fit:cover; display:block; }
.ih-pop-body { padding:.6rem .75rem .45rem; }
.ih-pop-cat { font-size:.66rem; color:#6366f1; text-transform:uppercase; letter-spacing:.1em; font-family:'JetBrains Mono',monospace; margin-bottom:.15rem; }
.ih-pop-title { font-size:.86rem; font-weight:600; color:#e2e8f0; line-height:1.3; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#0a0c12 !important; border-right:1px solid rgba(255,255,255,.05) !important; }
[data-testid="stSidebar"] p { color:#64748b !important; font-size:.84rem !important; }
[data-testid="stSidebar"] h2 { font-size:.95rem !important; color:#94a3b8 !important; }
[data-testid="stSidebar"] h3 { font-size:.78rem !important; color:#475569 !important; text-transform:uppercase; letter-spacing:.08em; }

/* ── Streamlit widget overrides ── */
.stTextInput>div>div>input {
  background:rgba(255,255,255,.04) !important;
  border:1px solid rgba(255,255,255,.1) !important;
  border-radius:9px !important;
  color:#f1f5f9 !important;
  font-size:.98rem !important;
  padding:.65rem 1rem !important;
}
.stTextInput>div>div>input:focus {
  border-color:rgba(99,102,241,.6) !important;
  box-shadow:0 0 0 3px rgba(99,102,241,.1) !important;
  outline:none !important;
}
.stButton>button[kind="primary"] {
  background:linear-gradient(135deg,#6366f1,#818cf8) !important;
  border:none !important; border-radius:9px !important;
  font-weight:600 !important; font-size:.88rem !important;
  padding:.6rem 1.4rem !important; letter-spacing:.02em !important;
  transition:opacity .2s, transform .1s !important;
}
.stButton>button[kind="primary"]:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
.stButton>button[kind="secondary"] {
  background:rgba(255,255,255,.04) !important;
  border:1px solid rgba(255,255,255,.1) !important;
  border-radius:8px !important; color:#94a3b8 !important; font-size:.84rem !important;
}
.stButton>button[kind="secondary"]:hover { border-color:rgba(99,102,241,.4) !important; color:#818cf8 !important; }
.stMetric { background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06); border-radius:10px; padding:.75rem; }
[data-testid="metric-value"] { color:#818cf8 !important; font-family:'JetBrains Mono',monospace !important; }
[data-testid="metric-label"] { color:#64748b !important; font-size:.76rem !important; }
div[data-testid="stInfo"]    { background:rgba(99,102,241,.08) !important; border:1px solid rgba(99,102,241,.2) !important; border-radius:8px !important; }
div[data-testid="stSuccess"] { background:rgba(34,197,94,.07) !important; border:1px solid rgba(34,197,94,.2) !important; border-radius:8px !important; }
div[data-testid="stWarning"] { background:rgba(245,158,11,.08) !important; border:1px solid rgba(245,158,11,.2) !important; border-radius:8px !important; }
div[data-testid="stError"]   { background:rgba(239,68,68,.08) !important; border:1px solid rgba(239,68,68,.2) !important; border-radius:8px !important; }
.stProgress>div>div>div { background:#6366f1 !important; border-radius:3px !important; }
.stExpander { border:1px solid rgba(255,255,255,.07) !important; border-radius:10px !important; background:rgba(255,255,255,.015) !important; }
.stCheckbox span { color:#94a3b8 !important; }
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,.02); border-radius:10px; padding:.25rem; gap:.2rem; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:7px; color:#64748b; font-size:.86rem; font-weight:500; padding:.38rem .95rem; }
.stTabs [aria-selected="true"] { background:rgba(99,102,241,.18) !important; color:#818cf8 !important; }
hr { border-color:rgba(255,255,255,.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Services ──────────────────────────────────────────────────────────────────

reddit_service = RedditService()
llm_service = LLMServiceFactory.create()
cross_platform_manager = CrossPlatformManager()

# ── Session state bootstrap ───────────────────────────────────────────────────

if "pending_query" in st.session_state:
    q = st.session_state.pop("pending_query")
    st.session_state["search_input"] = q
    st.session_state["query"] = q

if "search_input" not in st.session_state:
    st.session_state["search_input"] = st.session_state.get("query", "Tesla Model Y")

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ih-hero">
  <div class="ih-logo">
    <div class="ih-logo-mark">◈</div>
    <span class="ih-logo-name">Insight<span>Hub</span></span>
  </div>
  <p class="ih-tagline">AI-powered review analysis across Reddit &amp; YouTube — sentiment scoring, entity ranking, and actionable insights.</p>
</div>
""", unsafe_allow_html=True)

# ── Search bar ────────────────────────────────────────────────────────────────

st.markdown('<div class="ih-search-wrap"><div class="ih-search-label">Search & Analyze</div>', unsafe_allow_html=True)
scol1, scol2 = st.columns([5, 1])
with scol1:
    st.text_input(
        "query",
        key="search_input",
        label_visibility="collapsed",
        placeholder="e.g. Tesla Model Y, best laptop 2024, fix PS5 controller drift…",
    )
with scol2:
    analyze_clicked = st.button("Analyze  ›", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

st.session_state["query"] = st.session_state.get("search_input", "Tesla Model Y")
run_analysis = st.session_state.pop("run_analysis", False) or analyze_clicked

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### Volume")
    limit = st.slider(
        "Comments to collect",
        SearchConstants.MIN_COMMENTS_UI,
        SearchConstants.MAX_COMMENTS_UI,
        SearchConstants.DEFAULT_COMMENTS_UI,
        step=10,
    )

    st.markdown("---")
    st.markdown("### Platforms")
    enable_cross_platform = st.checkbox(
        "Cross-platform analysis",
        value=True,
        help="Pull reviews from Reddit and YouTube simultaneously",
    )

    if enable_cross_platform:
        platform_options = [Platform.REDDIT.value, Platform.YOUTUBE.value]
        selected_platform_names = st.multiselect(
            "Active platforms",
            platform_options,
            default=platform_options,
            help="Choose which platforms to search",
        )
        selected_platforms = [Platform(p) for p in selected_platform_names]
    else:
        selected_platforms = [Platform.REDDIT]

    if not enable_cross_platform or Platform.REDDIT in selected_platforms:
        st.markdown("---")
        st.markdown("### Reddit")
        subreddit_count = st.slider(
            "Subreddits to search",
            SearchConstants.MIN_SUBREDDITS_UI,
            SearchConstants.MAX_SUBREDDITS_UI,
            SearchConstants.DEFAULT_SUBREDDITS_UI,
            step=1,
            help="More subreddits → better coverage, slower search",
        )
    else:
        subreddit_count = SearchConstants.DEFAULT_SUBREDDITS_UI

    st.markdown("---")
    st.markdown("### Debug")
    show_debug = st.checkbox("Show raw data", value=False)

# ── Analysis ──────────────────────────────────────────────────────────────────

query = st.session_state.get("query", "Tesla Model Y")

if run_analysis:
    try:
        status = st.status("Running analysis…", expanded=True)
        with status:
            # Step 1 — intent detection
            st.write("🔍 Detecting query intent…")
            intent_schema = llm_service.detect_intent_and_schema(query)
            intent = (
                QueryIntent(intent_schema.intent)
                if intent_schema.intent in ["RANKING", "SOLUTION", "GENERIC"]
                else QueryIntent.GENERIC
            )

            badge_cls = {
                "RANKING": "ih-badge-ranking",
                "SOLUTION": "ih-badge-solution",
                "GENERIC": "ih-badge-generic",
            }.get(intent_schema.intent, "ih-badge-generic")

            # Step 2 — data collection
            use_cross = len(selected_platforms) > 1 or (
                len(selected_platforms) == 1 and selected_platforms[0] != Platform.REDDIT
            )

            if use_cross:
                st.write(f"🌐 Fetching reviews from {', '.join(p.value for p in selected_platforms)}…")
                start_time = time.time()
                cross_platform_results = cross_platform_manager.search_cross_platform(
                    query, intent, limit_per_platform=limit, enabled_platforms=selected_platforms
                )
                search_time = time.time() - start_time

                all_reviews = []
                for pname, prev in cross_platform_results["platform_results"].items():
                    all_reviews.extend(prev)
                reviews = all_reviews
                aggregated = cross_platform_results["aggregated"]
                total_reviews = sum(
                    len(v) for v in cross_platform_results["platform_results"].values()
                )
                platform_breakdown = cross_platform_results["platform_results"]
            else:
                st.write("🟠 Searching Reddit…")
                start_time = time.time()
                reviews = reddit_service.scrape(query, limit, subreddit_count)
                search_time = time.time() - start_time
                platform_breakdown = {"reddit": reviews}

            # Step 3 — GPT annotation
            st.write(f"🤖 Annotating {len(reviews)} reviews with GPT…")
            comments = []
            for review in reviews:
                comments.append({
                    "id": review.get("id") if isinstance(review, dict) else review.id,
                    "text": review.get("text") if isinstance(review, dict) else review.text,
                    "upvotes": review.get("upvotes", 0) if isinstance(review, dict) else getattr(review, "upvotes", 0),
                    "permalink": review.get("permalink", "") if isinstance(review, dict) else getattr(review, "permalink", ""),
                })

            annos = llm_service.annotate_comments_with_gpt(
                comments, intent_schema.aspects, intent_schema.entity_type, query
            )
            upvote_map = {c["id"]: c["upvotes"] for c in comments}

            # Step 4 — scoring + summarisation
            st.write("📊 Scoring and summarising…")
            if intent_schema.intent == "RANKING":
                ranking = rank_entities(
                    annos, upvote_map, intent_schema.entity_type, min_mentions=1, query=query
                )
                for item in ranking:
                    item.quotes = [
                        c["text"][:200] + "…"
                        for c in comments
                        if item.name.lower() in c["text"].lower()
                    ][:3]

                ranking_data = [
                    {
                        "name": item.name,
                        "overall_stars": item.overall_stars,
                        "mentions": item.mentions,
                        "quotes": item.quotes,
                    }
                    for item in ranking
                ]
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
                            "quotes": item.quotes,
                        }
                        for item in ranking
                    ],
                }

            elif intent_schema.intent == "SOLUTION":
                clusters = defaultdict(list)
                for anno in annos:
                    if anno.solution_key:
                        clusters[anno.solution_key].append(anno)

                solution_clusters = [
                    {
                        "title": ck,
                        "steps": [],
                        "caveats": [],
                        "evidence_count": len(cv),
                    }
                    for ck, cv in clusters.items()
                    if len(cv) >= 2
                ]
                summary = llm_service.summarize_solutions_with_gpt(query, solution_clusters)
                payload = {
                    "query": query,
                    "intent": intent_schema.intent,
                    "summary": summary,
                    "metadata": {"timestamp": time.time()},
                    "solutions": solution_clusters,
                }

            else:  # GENERIC
                overall, aspect_averages = aggregate_generic(
                    intent_schema.aspects, annos, upvote_map
                )
                positive_annos = sorted(
                    [a for a in annos if a.overall_score >= 3.5],
                    key=lambda a: a.overall_score,
                    reverse=True,
                )
                quotes = []
                for anno in positive_annos[:8]:
                    c = next((x for x in comments if x["id"] == anno.comment_id), None)
                    if c:
                        quotes.append(c["text"][:200] + "…")

                summary = llm_service.summarize_generic_with_gpt(
                    query, aspect_averages, overall, quotes
                )
                payload = {
                    "query": query,
                    "intent": intent_schema.intent,
                    "summary": summary,
                    "metadata": {"timestamp": time.time()},
                    "overall": overall,
                    "aspects": aspect_averages,
                    "quotes": quotes,
                }

            status.update(label=f"Analysis complete — {len(reviews)} reviews in {search_time:.1f}s", state="complete")

        # ── RESULTS ──────────────────────────────────────────────────────────

        if not reviews:
            st.warning("No reviews found. Try a different search term or increase the comment limit.")
        else:
            # Header row: query + intent badge
            aspects_preview = ", ".join(intent_schema.aspects[:5])
            if len(intent_schema.aspects) > 5:
                aspects_preview += "…"
            entity_line = f" · <span style='color:#64748b'>{intent_schema.entity_type}</span>" if intent_schema.entity_type else ""
            st.markdown(
                f"<h2 style='margin-bottom:.25rem'>{query}</h2>"
                f"<div style='margin-bottom:1.2rem'>"
                f"<span class='ih-badge {badge_cls}'>{intent_schema.intent}</span>"
                f"{entity_line}"
                f"<span style='font-size:.78rem;color:#475569;margin-left:.6rem;font-family:JetBrains Mono,monospace'>"
                f"aspects: {aspects_preview}</span></div>",
                unsafe_allow_html=True,
            )

            # ── Metrics row ──
            meaningful = [r for r in reviews if len((r.get("text") or "")) >= 100]
            col3_val = (
                str(len(payload.get("ranking", []))) + " entities"
                if intent_schema.intent == "RANKING"
                else str(len(payload.get("solutions", []))) + " clusters"
                if intent_schema.intent == "SOLUTION"
                else f"{payload.get('overall', 0):.1f} / 5"
            )
            col3_lbl = (
                "Ranked"
                if intent_schema.intent == "RANKING"
                else "Solutions"
                if intent_schema.intent == "SOLUTION"
                else "Overall"
            )
            platforms_str = " + ".join(p.title() for p in platform_breakdown.keys())
            st.markdown(
                f"""<div class="ih-chips">
                  {_metric_chip(str(len(reviews)), "Reviews")}
                  {_metric_chip(str(len(meaningful)), "Detailed")}
                  {_metric_chip(col3_val, col3_lbl)}
                  {_metric_chip(f"{search_time:.1f}s", "Search time")}
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Tabs ──
            intent_tab = {
                "RANKING": "🏆 Rankings",
                "SOLUTION": "🔧 Solutions",
                "GENERIC": "💡 Insights",
            }.get(intent_schema.intent, "💡 Insights")

            tab_overview, tab_results, tab_reviews = st.tabs(
                ["Overview", intent_tab, "Reviews"]
            )

            # ── Overview tab ─────────────────────────────────────────────────
            with tab_overview:
                st.markdown('<div class="ih-section-title">AI Summary</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ih-summary-card">{payload["summary"]}</div>',
                    unsafe_allow_html=True,
                )

                if intent_schema.intent == "GENERIC":
                    ov = payload.get("overall", 3.0)
                    verdict_cls = (
                        "verdict-good" if ov >= 4.0
                        else "verdict-mid" if ov >= 3.0
                        else "verdict-low"
                    )
                    verdict_txt = (
                        "Highly recommended" if ov >= 4.0
                        else "Generally positive" if ov >= 3.0
                        else "Mixed reviews"
                    )
                    c_rating, c_aspects = st.columns([1, 2])
                    with c_rating:
                        st.markdown(
                            f"""<div class="ih-big-rating">
                              <div class="score">{ov:.1f}</div>
                              <div class="denom">out of 5</div>
                              <div class="stars">{_stars(ov)}</div>
                              <div class="verdict {verdict_cls}">{verdict_txt}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with c_aspects:
                        st.markdown('<div class="ih-section-title">Aspect Breakdown</div>', unsafe_allow_html=True)
                        bars = "".join(
                            _aspect_bar(asp, sc)
                            for asp, sc in list(payload.get("aspects", {}).items())
                        )
                        st.markdown(bars or "<p style='color:#475569;font-size:.85rem'>No aspect data</p>", unsafe_allow_html=True)

                if intent_schema.intent == "RANKING" and payload.get("ranking"):
                    st.markdown('<div class="ih-section-title">Top 3 at a Glance</div>', unsafe_allow_html=True)
                    for i, item in enumerate(payload["ranking"][:3], 1):
                        num_cls = ["gold", "silver", "bronze"][i - 1]
                        st.markdown(
                            f"""<div class="ih-rank-card">
                              <div class="ih-rank-num {num_cls}">#{i}</div>
                              <div>
                                <div class="ih-rank-name">{item["name"]}</div>
                                <div class="ih-rank-stars">{_stars(item["overall_stars"])}</div>
                                <div class="ih-rank-meta">{item["overall_stars"]:.1f}/5 · {item["mentions"]} mentions</div>
                              </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                # Platform breakdown (cross-platform only)
                if use_cross:
                    st.markdown('<div class="ih-section-title">Platform Breakdown</div>', unsafe_allow_html=True)
                    pcols = st.columns(len(platform_breakdown))
                    for col, (pname, prevs) in zip(pcols, platform_breakdown.items()):
                        with col:
                            st.markdown(
                                _metric_chip(str(len(prevs)), pname.title()),
                                unsafe_allow_html=True,
                            )

                # Debug section
                if show_debug:
                    with st.expander("🔧 Raw data — first 3 reviews"):
                        import pandas as pd
                        df = pd.DataFrame(reviews[:3])
                        st.dataframe(df[["id", "author", "upvotes", "permalink", "text"]])

                    if not use_cross:
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
                                st.caption(f"Plan unavailable: {e}")

            # ── Results tab ──────────────────────────────────────────────────
            with tab_results:
                if intent_schema.intent == "RANKING":
                    st.markdown('<div class="ih-section-title">Full Ranking</div>', unsafe_allow_html=True)
                    if payload["ranking"]:
                        for i, item in enumerate(payload["ranking"][: SearchConstants.MAX_ENTITIES_TO_DISPLAY], 1):
                            num_cls = (
                                "gold" if i == 1
                                else "silver" if i == 2
                                else "bronze" if i == 3
                                else ""
                            )
                            aspects_html = "".join(
                                _aspect_bar(asp, sc)
                                for asp, sc in list(item.get("aspect_scores", {}).items())[:4]
                            )
                            st.markdown(
                                f"""<div class="ih-rank-card">
                                  <div class="ih-rank-num {num_cls}">#{i}</div>
                                  <div style="flex:1">
                                    <div class="ih-rank-name">{item["name"]}</div>
                                    <div class="ih-rank-stars">{_stars(item["overall_stars"])}</div>
                                    <div class="ih-rank-meta">{item["overall_stars"]:.1f}/5 · {item["mentions"]} mentions · confidence {item.get("confidence", 0):.0%}</div>
                                    <div style="margin-top:.6rem">{aspects_html}</div>
                                  </div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                            if item.get("quotes"):
                                with st.expander(f"Quotes about {item['name']}"):
                                    for q in item["quotes"][:3]:
                                        st.markdown(_quote_block(q), unsafe_allow_html=True)
                    else:
                        st.info("No ranked entities found. Try a more specific query or increase the comment limit.")

                elif intent_schema.intent == "SOLUTION":
                    st.markdown('<div class="ih-section-title">Solution Clusters</div>', unsafe_allow_html=True)
                    if payload["solutions"]:
                        for i, cluster in enumerate(payload["solutions"], 1):
                            st.markdown(
                                f"""<div class="ih-solution-card">
                                  <div class="ih-solution-title">{i}. {cluster["title"]}</div>
                                  <div class="ih-solution-badge">▲ {cluster["evidence_count"]} supporting comments</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                            if cluster.get("steps"):
                                with st.expander("Steps"):
                                    for step in cluster["steps"]:
                                        st.write(f"- {step}")
                            if cluster.get("caveats"):
                                with st.expander("Caveats"):
                                    for caveat in cluster["caveats"]:
                                        st.write(f"- {caveat}")
                    else:
                        st.info("No solution clusters found. Try a more specific query or increase the comment limit.")

                else:  # GENERIC
                    st.markdown('<div class="ih-section-title">Key Insights</div>', unsafe_allow_html=True)
                    quotes = payload.get("quotes", [])
                    if quotes:
                        for q in quotes[:6]:
                            st.markdown(_quote_block(q), unsafe_allow_html=True)
                    else:
                        st.info("No notable quotes found.")

                    st.markdown('<div class="ih-section-title">All Aspects</div>', unsafe_allow_html=True)
                    bars = "".join(
                        _aspect_bar(asp, sc)
                        for asp, sc in payload.get("aspects", {}).items()
                    )
                    st.markdown(bars or "<p style='color:#475569;font-size:.85rem'>No aspect data</p>", unsafe_allow_html=True)

            # ── Reviews tab ──────────────────────────────────────────────────
            with tab_reviews:
                sorted_reviews = sorted(
                    [r for r in reviews if len((r.get("text") or "")) >= 100],
                    key=lambda r: -(r.get("upvotes", 0) or 0),
                )

                st.markdown(
                    f'<div class="ih-section-title">{len(sorted_reviews)} Meaningful Reviews — sorted by upvotes</div>',
                    unsafe_allow_html=True,
                )

                shown = sorted_reviews[:10]
                rest = sorted_reviews[10:]

                for review in shown:
                    link = review.get("url") or (
                        f"https://reddit.com{review.get('permalink', '')}"
                        if review.get("permalink")
                        else ""
                    )
                    st.markdown(
                        _review_card(
                            0,
                            review.get("author", "u/unknown"),
                            review.get("upvotes", 0),
                            _excerpt(review.get("text", "")),
                            link,
                        ),
                        unsafe_allow_html=True,
                    )

                if rest:
                    with st.expander(f"Load {len(rest)} more reviews"):
                        for review in rest:
                            link = review.get("url") or (
                                f"https://reddit.com{review.get('permalink', '')}"
                                if review.get("permalink")
                                else ""
                            )
                            st.markdown(
                                _review_card(
                                    0,
                                    review.get("author", "u/unknown"),
                                    review.get("upvotes", 0),
                                    _excerpt(review.get("text", "")),
                                    link,
                                ),
                                unsafe_allow_html=True,
                            )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {e}")
        st.info("Please try again with a different search term.")

# ── Popular searches ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="ih-section-title">Recommended Searches</div>', unsafe_allow_html=True)

POPULAR = [
    {
        "title": "Top 10 NYC Restaurants",
        "cat": "Food & Dining",
        "q": "top 10 nyc restaurant",
        "image": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&q=80",
    },
    {
        "title": "Tesla Model Y",
        "cat": "Automotive",
        "q": "Tesla Model Y",
        "image": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=400&q=80",
    },
    {
        "title": "Nintendo Switch",
        "cat": "Gaming",
        "q": "Nintendo Switch",
        "image": "https://images.unsplash.com/photo-1578303512597-81e6cc155b3e?w=400&q=80",
    },
    {
        "title": "Best Golf Course — Bay Area",
        "cat": "Sports & Recreation",
        "q": "best golf course in bay area",
        "image": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=400&q=80",
    },
]

pop_cols = st.columns(4, gap="large")
for i, p in enumerate(POPULAR):
    with pop_cols[i]:
        st.markdown(
            f"""<div class="ih-pop-card">
              <img src="{p['image']}" alt="{p['title']}" />
              <div class="ih-pop-body">
                <div class="ih-pop-cat">{p['cat']}</div>
                <div class="ih-pop-title">{p['title']}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Analyze  ›", key=f"pop_{i}", use_container_width=True, type="secondary"):
            st.session_state["pending_query"] = p["q"]
            st.session_state["run_analysis"] = True
            st.rerun()


def main():
    """Entry point — Streamlit runs the module directly."""
    pass
