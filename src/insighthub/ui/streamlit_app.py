"""Streamlit UI for InsightHub — AI consumer intelligence platform."""

import streamlit as st
import logging
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from insighthub.core.scoring import aggregate_generic, rank_entities_with_relaxation
from insighthub.utils.data_prep import export_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _excerpt(s, n=360):
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
        return "#22d3a0"
    if r >= 0.50:
        return "#f59e0b"
    return "#f87171"

def _score_cls(score: float) -> str:
    if score >= 3.6:
        return "score-high"
    if score >= 2.8:
        return "score-mid"
    return "score-low"

def _stars(score: float, max_score: float = 5.0) -> str:
    filled = round(score / max_score * 5)
    return "★" * filled + "☆" * (5 - filled)

def _confidence_label(conf: float) -> tuple:
    if conf >= 0.70:
        return "High confidence", "#22d3a0"
    if conf >= 0.40:
        return "Moderate", "#f59e0b"
    return "Limited data", "#64748b"

def _aspect_bar(label: str, score: float, max_score: float = 5.0) -> str:
    pct = score / max_score * 100
    color = _score_color(score, max_score)
    return (
        f'<div class="ih-aspect-row">'
        f'<span class="ih-aspect-label">{label.replace("_"," ").title()}</span>'
        f'<div class="ih-aspect-track">'
        f'<div class="ih-aspect-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'</div>'
        f'<span class="ih-aspect-score" style="color:{color}">{score:.1f}</span>'
        f'</div>'
    )

def _source_tag(review: dict) -> str:
    url = review.get("url", "") or ""
    permalink = review.get("permalink", "") or ""
    if permalink:
        return '<span class="ih-src ih-src-reddit">Reddit</span>'
    if "youtube" in url.lower() or "youtu.be" in url.lower():
        return '<span class="ih-src ih-src-youtube">YouTube</span>'
    return ""

def _entity_verdict(item: dict) -> str:
    aspects = item.get("aspect_scores", {})
    conf = item.get("confidence", 0.5)
    conf_lbl = "strong consensus" if conf >= 0.7 else "moderate consensus" if conf >= 0.4 else "limited data"
    if not aspects:
        return conf_lbl.capitalize()
    best = max(aspects, key=aspects.get)
    worst = min(aspects, key=aspects.get)
    bs, ws = aspects[best], aspects[worst]
    best_name = best.replace("_", " ")
    worst_name = worst.replace("_", " ")
    if bs >= 4.0 and ws < 2.8:
        return f"Excels at {best_name} · struggles with {worst_name}"
    if bs >= 4.2:
        return f"Standout {best_name} · {conf_lbl}"
    if ws < 2.5:
        return f"Poor {worst_name} is the main concern · {conf_lbl}"
    return f"{conf_lbl.capitalize()}"

def _sentiment_label(score: float) -> tuple:
    if score >= 3.8:
        return "Positive", "#22d3a0", "rgba(34,211,160,0.10)"
    if score >= 2.8:
        return "Mixed", "#f59e0b", "rgba(245,158,11,0.09)"
    return "Critical", "#f87171", "rgba(248,113,113,0.09)"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InsightHub",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, .stApp {
  background: #0d0f16 !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: #cbd5e1;
}
.block-container {
  padding: 0 2rem 5rem !important;
  max-width: 960px !important;
  margin: 0 auto !important;
}
* { box-sizing: border-box; }
h1, h2, h3 {
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  color: #f1f5f9 !important;
  letter-spacing: -0.03em !important;
}

/* ── Animations ── */
@keyframes fadeUp   { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.25; } }
@keyframes scanline { 0% { transform:translateY(-100%); } 100% { transform:translateY(400%); } }

/* ── Homepage ── */
.ih-hero {
  text-align: center;
  padding: 4rem 1rem 1.5rem;
  animation: fadeUp 0.55s ease both;
}
.ih-wordmark {
  font-size: 2rem;
  font-weight: 800;
  color: #f1f5f9;
  letter-spacing: -0.055em;
  line-height: 1;
  margin-bottom: 0.65rem;
}
.ih-wordmark span { color: #6366f1; }
.ih-tagline {
  font-size: 0.9rem;
  color: #64748b;
  line-height: 1.65;
  max-width: 400px;
  margin: 0 auto 1.25rem;
}
.ih-pulse-bar {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.7rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.07em;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px;
  padding: 0.3rem 0.85rem;
  background: rgba(255,255,255,0.025);
}
.ih-pulse-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: #22d3a0;
  animation: pulse-dot 2s ease-in-out infinite;
}

/* ── Pipeline strip (homepage) ── */
.ih-pipeline {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin: 2.25rem auto 0;
  max-width: 520px;
}
.ih-pipe-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  flex: 1;
}
.ih-pipe-icon {
  width: 36px; height: 36px;
  border-radius: 9px;
  background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.95rem;
}
.ih-pipe-lbl {
  font-size: 0.62rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
  text-align: center;
  letter-spacing: 0.04em;
}
.ih-pipe-arrow { color: #334155; font-size: 0.9rem; padding: 0 0.2rem; padding-bottom: 1.1rem; }

/* ── Suggestions ── */
.ih-sug-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
  text-align: center;
  margin: 2.25rem 0 0.8rem;
}

/* ── Result header ── */
.ih-result-header { padding: 2.25rem 0 0; }
.ih-query-title {
  font-size: 1.65rem;
  font-weight: 700;
  color: #f1f5f9;
  letter-spacing: -0.035em;
  margin: 0 0 0.65rem;
  line-height: 1.2;
}
.ih-meta-row {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
  padding-bottom: 1.1rem;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  margin-bottom: 0.1rem;
}
.ih-intent-badge {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 0.22rem 0.6rem;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
}
.ih-badge-RANKING  { background: rgba(99,102,241,0.18); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.ih-badge-SOLUTION { background: rgba(34,211,160,0.12); color: #34d399; border: 1px solid rgba(34,211,160,0.25); }
.ih-badge-GENERIC  { background: rgba(245,158,11,0.12);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.ih-meta-dot { color: #334155; }
.ih-meta-txt {
  font-size: 0.77rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Stats strip ── */
.ih-stats-strip {
  display: flex;
  gap: 0;
  padding: 1rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 1.5rem;
}
.ih-stat {
  flex: 1;
  padding: 0 1.5rem 0 0;
  border-right: 1px solid rgba(255,255,255,0.05);
  margin-right: 1.5rem;
}
.ih-stat:last-child { border-right: none; margin-right: 0; }
.ih-stat-val {
  font-size: 1.45rem;
  font-weight: 700;
  color: #f1f5f9;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.03em;
  line-height: 1;
  margin-bottom: 0.25rem;
}
.ih-stat-lbl {
  font-size: 0.66rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* ── AI Summary card ── */
.ih-summary-card {
  background: rgba(99,102,241,0.06);
  border: 1px solid rgba(99,102,241,0.22);
  border-left: 3px solid #6366f1;
  border-radius: 0 10px 10px 0;
  padding: 1.4rem 1.6rem 1.2rem;
  margin-bottom: 1.75rem;
  position: relative;
  overflow: hidden;
}
.ih-summary-card::before {
  content: '';
  position: absolute;
  top: 0; right: 0; bottom: 0;
  width: 60%;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.03));
  pointer-events: none;
}
.ih-summary-eyebrow {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: #6366f1;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.ih-summary-text {
  font-size: 0.93rem;
  color: #cbd5e1;
  line-height: 1.8;
}
.ih-summary-foot {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 1rem;
  padding-top: 0.85rem;
  border-top: 1px solid rgba(99,102,241,0.15);
  font-size: 0.71rem;
  color: #64748b;
  font-family: 'JetBrains Mono', monospace;
}
.ih-conf-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.ih-foot-right { margin-left: auto; color: #475569; }

/* ── Section header ── */
.ih-section-hdr {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #64748b;
  font-family: 'JetBrains Mono', monospace;
  margin: 1.75rem 0 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.ih-section-hdr::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255,255,255,0.06);
}

/* ── Ranking cards (always-visible, not collapsed) ── */
.ih-rank-card {
  background: rgba(255,255,255,0.028);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 10px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 0.6rem;
  position: relative;
  transition: border-color 0.15s, background 0.15s;
}
.ih-rank-card:hover {
  background: rgba(255,255,255,0.042);
  border-color: rgba(255,255,255,0.14);
}
.ih-rank-card-top3 {
  background: rgba(99,102,241,0.045);
  border-color: rgba(99,102,241,0.18);
}
.ih-rank-card-gold   { border-left: 3px solid #c89b3c !important; }
.ih-rank-card-silver { border-left: 3px solid #6b7a8d !important; }
.ih-rank-card-bronze { border-left: 3px solid #8a6040 !important; }

.ih-rank-head {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
.ih-rank-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  min-width: 1.8rem;
  padding-top: 0.2rem;
  color: #475569;
  flex-shrink: 0;
}
.rank-gold   { color: #c89b3c !important; }
.rank-silver { color: #7a8fa8 !important; }
.rank-bronze { color: #8a6040 !important; }
.ih-rank-info { flex: 1; min-width: 0; }
.ih-rank-name {
  font-size: 1rem;
  font-weight: 700;
  color: #f1f5f9;
  letter-spacing: -0.02em;
  margin-bottom: 0.2rem;
}
.ih-rank-stars { font-size: 0.82rem; color: #c89b3c; letter-spacing: 0.08em; }
.ih-rank-verdict {
  font-size: 0.78rem;
  color: #64748b;
  margin-top: 0.2rem;
  font-style: italic;
  line-height: 1.4;
}
.ih-rank-score-block {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  flex-shrink: 0;
  gap: 0.2rem;
}
.ih-rank-big-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1;
}
.ih-rank-mentions {
  font-size: 0.66rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
  text-align: right;
}
.score-high { color: #22d3a0 !important; }
.score-mid  { color: #f59e0b !important; }
.score-low  { color: #f87171 !important; }

/* ── Aspect bars ── */
.ih-aspects { margin-top: 0.1rem; }
.ih-aspect-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.35rem;
}
.ih-aspect-label { font-size: 0.74rem; color: #64748b; min-width: 120px; }
.ih-aspect-track {
  flex: 1;
  height: 4px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}
.ih-aspect-fill { height: 100%; border-radius: 2px; }
.ih-aspect-score {
  font-size: 0.7rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  min-width: 2rem;
  text-align: right;
}
.ih-rank-foot {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.65rem;
  padding-top: 0.6rem;
  border-top: 1px solid rgba(255,255,255,0.05);
  font-size: 0.68rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
}
.ih-conf-pill {
  padding: 0.12rem 0.5rem;
  border-radius: 3px;
  font-size: 0.63rem;
  font-weight: 600;
  letter-spacing: 0.06em;
}

/* ── Overview rank list (compact) ── */
.ih-ov-rank-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.ih-ov-rank-row:last-child { border-bottom: none; }
.ih-ov-pos { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; min-width: 1.6rem; }
.ih-ov-name { flex: 1; font-size: 0.92rem; font-weight: 600; color: #e2e8f0; }
.ih-ov-stars { font-size: 0.78rem; color: #c89b3c; }
.ih-ov-score { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; min-width: 2.5rem; text-align: right; }

/* ── Generic score block ── */
.ih-score-hero {
  padding: 0.5rem 0 1rem;
}
.ih-score-num {
  font-size: 3rem;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.05em;
  line-height: 1;
}
.ih-score-denom { font-size: 1rem; color: #475569; font-weight: 400; }
.ih-score-stars { font-size: 0.9rem; color: #c89b3c; letter-spacing: 0.09em; margin-top: 0.3rem; }
.ih-score-verdict { font-size: 0.82rem; font-weight: 600; margin-top: 0.3rem; }
.verdict-high { color: #22d3a0; }
.verdict-mid  { color: #f59e0b; }
.verdict-low  { color: #f87171; }

/* ── Solution items ── */
.ih-solution-item {
  display: flex;
  gap: 1rem;
  padding: 0.9rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.ih-solution-item:last-child { border-bottom: none; }
.ih-sol-idx {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  color: #6366f1;
  min-width: 1.6rem;
  padding-top: 0.15rem;
  flex-shrink: 0;
}
.ih-sol-name { font-size: 0.93rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.ih-sol-count { font-size: 0.7rem; color: #22d3a0; font-family: 'JetBrains Mono', monospace; }

/* ── Quote block ── */
.ih-quote {
  border-left: 2px solid rgba(99,102,241,0.4);
  padding: 0.55rem 1rem;
  margin: 0.5rem 0;
  font-size: 0.86rem;
  color: #64748b;
  font-style: italic;
  line-height: 1.68;
  background: rgba(99,102,241,0.04);
  border-radius: 0 6px 6px 0;
}

/* ── Evidence feed ── */
.ih-evidence-group-hdr {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  font-family: 'JetBrains Mono', monospace;
  margin: 1.5rem 0 0.65rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.ih-evidence-group-hdr::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255,255,255,0.06);
}
.ih-evidence-item {
  padding: 0.9rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.ih-evidence-item:last-child { border-bottom: none; }
.ih-ev-meta {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}
.ih-ev-author {
  font-size: 0.75rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
}
.ih-ev-author a { color: #475569; text-decoration: none; }
.ih-ev-author a:hover { color: #818cf8; }
.ih-upvotes { font-size: 0.7rem; color: #334155; font-family: 'JetBrains Mono', monospace; }
.ih-upvotes-high { color: #6366f1 !important; font-weight: 600; }
.ih-ev-text { font-size: 0.875rem; color: #94a3b8; line-height: 1.72; }

/* ── Source tags ── */
.ih-src {
  font-size: 0.6rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.09em;
  padding: 0.13rem 0.45rem; border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  flex-shrink: 0;
}
.ih-src-reddit  { background: rgba(255,69,0,0.12); color: #ff6633; border: 1px solid rgba(255,69,0,0.2); }
.ih-src-youtube { background: rgba(255,0,0,0.1); color: #ff4d4d; border: 1px solid rgba(255,0,0,0.18); }

/* ── Sentiment tag ── */
.ih-sent {
  font-size: 0.6rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.09em;
  padding: 0.13rem 0.45rem; border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  flex-shrink: 0;
}

/* ── Platform breakdown ── */
.ih-plat-strip { display: flex; gap: 2rem; padding: 0.5rem 0 0.25rem; }
.ih-plat-item { font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.ih-plat-count { font-weight: 700; color: #94a3b8; margin-right: 0.3rem; }

/* ── Homepage search area ── */
.ih-search-area { padding: 3.25rem 1rem 1.5rem; text-align: center; }
.ih-search-heading {
  font-size: 2.1rem; font-weight: 800; color: #f1f5f9;
  letter-spacing: -0.045em; line-height: 1.15; margin-bottom: 0.5rem;
}
.ih-search-heading em { color: #6366f1; font-style: normal; }
.ih-search-sub { font-size: 0.875rem; color: #475569; margin-bottom: 1.75rem; line-height: 1.5; }
.ih-live-bar {
  display: inline-flex; align-items: center; gap: 0.5rem;
  font-size: 0.68rem; color: #475569; font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.07em; border: 1px solid rgba(255,255,255,0.07);
  border-radius: 20px; padding: 0.28rem 0.8rem; background: rgba(255,255,255,0.02);
  margin-bottom: 1.5rem;
}
.ih-live-dot { width: 5px; height: 5px; border-radius: 50%; background: #22d3a0; animation: pulse-dot 2s ease-in-out infinite; }

/* ── Homepage section label ── */
.ih-home-lbl {
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.15em; color: #475569; font-family: 'JetBrains Mono', monospace;
  margin: 2rem 0 1rem; display: flex; align-items: center; gap: 0.7rem;
}
.ih-home-lbl::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

/* ── Hero insight card (full-width, image bg) ── */
.ih-hero-card {
  position: relative; border-radius: 12px; overflow: hidden;
  height: 220px; margin-bottom: 0.5rem; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.1);
  transition: border-color 0.2s;
}
.ih-hero-card:hover { border-color: rgba(99,102,241,0.4); }
.ih-hero-card-img {
  position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
  filter: brightness(0.38) saturate(0.85);
}
.ih-hero-card-body {
  position: absolute; inset: 0; padding: 1.5rem 1.75rem;
  background: linear-gradient(135deg, rgba(13,15,22,0.65) 0%, transparent 60%),
              linear-gradient(to top, rgba(13,15,22,0.9) 0%, transparent 55%);
  display: flex; flex-direction: column; justify-content: space-between;
}
.ih-hero-card-top { display: flex; align-items: center; gap: 0.6rem; }
.ih-hero-card-cat {
  font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.12em; font-family: 'JetBrains Mono', monospace;
  padding: 0.18rem 0.55rem; border-radius: 3px;
  background: rgba(99,102,241,0.25); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4);
}
.ih-hero-card-sources { font-size: 0.63rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.ih-hero-card-bottom { }
.ih-hero-card-title { font-size: 1.45rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 0.35rem; }
.ih-hero-card-insight { font-size: 0.82rem; color: #94a3b8; line-height: 1.55; margin-bottom: 0.75rem; max-width: 70%; }
.ih-hero-card-stats { display: flex; align-items: center; gap: 1.25rem; }
.ih-hero-score { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.04em; }
.ih-hero-stars { font-size: 0.8rem; color: #c89b3c; letter-spacing: 0.09em; }
.ih-sentiment-strip { display: flex; gap: 0.75rem; align-items: center; }
.ih-sent-seg { display: flex; align-items: center; gap: 0.3rem; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; }
.ih-sent-bar { width: 28px; height: 3px; border-radius: 2px; }

/* ── Small insight cards (grid) ── */
.ih-insight-card {
  border-radius: 10px; overflow: hidden; position: relative;
  border: 1px solid rgba(255,255,255,0.08);
  transition: border-color 0.15s, transform 0.15s;
  background: rgba(255,255,255,0.025);
}
.ih-insight-card:hover { border-color: rgba(99,102,241,0.35); transform: translateY(-2px); }
.ih-insight-card-img { width: 100%; height: 110px; object-fit: cover; display: block; filter: brightness(0.45) saturate(0.8); }
.ih-insight-card-body { padding: 0.8rem 0.9rem 0.65rem; }
.ih-insight-card-cat {
  font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.11em;
  color: #6366f1; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.3rem;
}
.ih-insight-card-title { font-size: 0.87rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem; letter-spacing: -0.015em; line-height: 1.3; }
.ih-insight-card-insight { font-size: 0.74rem; color: #64748b; line-height: 1.5; margin-bottom: 0.5rem; }
.ih-insight-card-foot { display: flex; align-items: center; gap: 0.6rem; }
.ih-insight-card-score { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; }
.ih-insight-card-meta { font-size: 0.63rem; color: #475569; font-family: 'JetBrains Mono', monospace; }

/* ── Active debates feed ── */
.ih-debate-item {
  display: flex; align-items: flex-start; gap: 1rem; padding: 0.85rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.ih-debate-item:last-child { border-bottom: none; }
.ih-debate-left { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; flex-shrink: 0; }
.ih-debate-icon {
  width: 32px; height: 32px; border-radius: 7px; display: flex;
  align-items: center; justify-content: center; font-size: 0.85rem;
  flex-shrink: 0;
}
.ih-debate-body { flex: 1; min-width: 0; }
.ih-debate-title { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; line-height: 1.35; }
.ih-debate-sub { font-size: 0.72rem; color: #475569; font-family: 'JetBrains Mono', monospace; line-height: 1.4; }
.ih-debate-right { flex-shrink: 0; text-align: right; }
.ih-debate-stat { font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; line-height: 1; }
.ih-debate-stat-lbl { font-size: 0.6rem; color: #475569; font-family: 'JetBrains Mono', monospace; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.08em; }
.ih-debate-tag {
  font-size: 0.55rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  padding: 0.1rem 0.38rem; border-radius: 3px; font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0a0c12 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #64748b !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] h2 { font-size: 0.9rem !important; color: #94a3b8 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] h3 {
  font-size: 0.67rem !important; color: #475569 !important;
  text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700 !important;
}

/* ── Streamlit overrides ── */
.stTextInput > div > div > input {
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;
  font-size: 0.97rem !important;
  padding: 0.75rem 1.1rem !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(99,102,241,0.55) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #334155 !important; }
.stButton > button[kind="primary"] {
  background: #6366f1 !important;
  border: none !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  padding: 0.72rem 1.4rem !important;
  box-shadow: 0 0 0 0 rgba(99,102,241,0) !important;
  transition: background 0.15s, transform 0.1s, box-shadow 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
  background: #818cf8 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(99,102,241,0.25) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
  color: #64748b !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.38rem 0.9rem !important;
  transition: all 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(99,102,241,0.08) !important;
  border-color: rgba(99,102,241,0.35) !important;
  color: #818cf8 !important;
}
/* Underline tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 0 !important;
  gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border-radius: 0 !important;
  color: #475569 !important; font-size: 0.85rem !important;
  font-weight: 500 !important; padding: 0.65rem 1.2rem !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
  background: transparent !important;
  color: #f1f5f9 !important;
  border-bottom-color: #6366f1 !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }
.stExpander {
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 9px !important;
  background: rgba(255,255,255,0.02) !important;
}
.stExpander summary { color: #64748b !important; font-size: 0.83rem !important; }
div[data-testid="stInfo"]    { background: rgba(99,102,241,0.08) !important; border: 1px solid rgba(99,102,241,0.2) !important; border-radius: 9px !important; }
div[data-testid="stSuccess"] { background: rgba(34,211,160,0.07) !important; border: 1px solid rgba(34,211,160,0.2) !important; border-radius: 9px !important; }
div[data-testid="stWarning"] { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.2) !important; border-radius: 9px !important; }
div[data-testid="stError"]   { background: rgba(248,113,113,0.08) !important; border: 1px solid rgba(248,113,113,0.2) !important; border-radius: 9px !important; }
.stProgress > div > div > div { background: #6366f1 !important; border-radius: 2px !important; }
.stCheckbox span { color: #64748b !important; font-size: 0.83rem !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Mobile responsive (iPhone / narrow screens) ── */
@media (max-width: 640px) {
  /* Container: tighten horizontal padding only (preserve top padding for Streamlit header) */
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; padding-bottom: 3rem !important; }

  /* Search area */
  .ih-search-area { padding: 2.75rem 0.25rem 1rem; }
  .ih-search-heading { font-size: 1.45rem; letter-spacing: -0.03em; }
  .ih-search-sub { font-size: 0.82rem; margin-bottom: 1.25rem; }

  /* Stats strip: 2-column grid instead of 5-in-a-row */
  .ih-stats-strip { flex-wrap: wrap; }
  .ih-stat {
    flex: 0 0 50%;
    border-right: none !important;
    margin-right: 0 !important;
    padding: 0.5rem 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .ih-stat:nth-child(odd)  { padding-right: 0.75rem !important; border-right: 1px solid rgba(255,255,255,0.05) !important; }
  .ih-stat:nth-child(even) { padding-left: 0.75rem !important; }
  .ih-stat:nth-last-child(-n+2) { border-bottom: none; }
  .ih-stat-val { font-size: 1.15rem; }

  /* Result header */
  .ih-query-title { font-size: 1.2rem; }

  /* AI summary */
  .ih-summary-card { padding: 1rem 1rem 0.85rem; }
  .ih-summary-text { font-size: 0.87rem; }

  /* Ranking cards: stack score below name on very small screens */
  .ih-rank-head { flex-wrap: wrap; gap: 0.5rem; }
  .ih-rank-score-block { flex-direction: row; align-items: center; gap: 0.5rem; width: 100%; justify-content: flex-start; margin-top: 0.15rem; }
  .ih-rank-big-score { font-size: 1.3rem; }
  .ih-rank-mentions { text-align: left; }

  /* Aspect bars: shrink label */
  .ih-aspect-label { min-width: 72px; font-size: 0.68rem; }

  /* Hero image card */
  .ih-hero-card { height: 185px; }
  .ih-hero-card-body { padding: 1rem 1.1rem; }
  .ih-hero-card-title { font-size: 1.05rem; }
  .ih-hero-card-insight { max-width: 100%; font-size: 0.75rem; display: none; }
  .ih-hero-card-stats { gap: 0.85rem; }
  .ih-hero-score { font-size: 1.2rem; }

  /* Insight grid cards */
  .ih-insight-card-img { height: 90px; }
  .ih-insight-card-title { font-size: 0.8rem; }
  .ih-insight-card-insight { display: none; }

  /* Debates */
  .ih-debate-title { font-size: 0.8rem; }
  .ih-debate-sub { font-size: 0.62rem; }
  .ih-debate-stat { font-size: 0.82rem; }

  /* Evidence */
  .ih-ev-text { font-size: 0.82rem; line-height: 1.62; }

}
</style>
""", unsafe_allow_html=True)

# ── Services ──────────────────────────────────────────────────────────────────

reddit_service = RedditService()
llm_service = LLMServiceFactory.create()
cross_platform_manager = CrossPlatformManager()

# ── Session state ─────────────────────────────────────────────────────────────

if "pending_query" in st.session_state:
    q = st.session_state.pop("pending_query")
    st.session_state["search_input"] = q
    st.session_state["query"] = q

if "search_input" not in st.session_state:
    st.session_state["search_input"] = st.session_state.get("query", "")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")
    st.markdown("### Depth")
    limit = st.slider(
        "Reviews per source",
        SearchConstants.MIN_COMMENTS_UI,
        SearchConstants.MAX_COMMENTS_UI,
        SearchConstants.DEFAULT_COMMENTS_UI,
        step=25,
        help="More reviews = more thorough, but slower",
    )
    st.markdown("---")
    st.markdown("### Sources")
    enable_cross_platform = st.checkbox("Multi-platform", value=True,
        help="Combine Reddit and YouTube for broader coverage")
    if enable_cross_platform:
        platform_options = [Platform.REDDIT.value, Platform.YOUTUBE.value]
        selected_platform_names = st.multiselect("Active sources", platform_options, default=platform_options)
        selected_platforms = [Platform(p) for p in selected_platform_names]
    else:
        selected_platforms = [Platform.REDDIT]

    if not enable_cross_platform or Platform.REDDIT in selected_platforms:
        st.markdown("---")
        st.markdown("### Reddit")
        subreddit_count = st.slider(
            "Subreddits",
            SearchConstants.MIN_SUBREDDITS_UI,
            SearchConstants.MAX_SUBREDDITS_UI,
            SearchConstants.DEFAULT_SUBREDDITS_UI,
            step=1,
        )
    else:
        subreddit_count = SearchConstants.DEFAULT_SUBREDDITS_UI

    st.markdown("---")
    with st.expander("Developer"):
        show_debug = st.checkbox("Show raw data", value=False)

# ── Search area ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="ih-search-area">
  <div class="ih-search-heading">What does the internet<br><em>actually</em> think?</div>
  <p class="ih-search-sub">AI-extracted consensus from Reddit &amp; YouTube — scored, ranked, and explained.</p>
  <div class="ih-live-bar"><div class="ih-live-dot"></div>Live analysis &nbsp;·&nbsp; Reddit + YouTube &nbsp;·&nbsp; GPT-4 powered</div>
</div>
""", unsafe_allow_html=True)

_, col_q, col_btn, _ = st.columns([0.15, 5.6, 1.1, 0.15])
with col_q:
    st.text_input(
        "query", key="search_input", label_visibility="collapsed",
        placeholder="Ask anything — products, places, debates, fixes…",
    )
with col_btn:
    analyze_clicked = st.button("Analyze", use_container_width=True, type="primary")

st.session_state["query"] = st.session_state.get("search_input", "")
run_analysis = st.session_state.pop("run_analysis", False) or analyze_clicked

# ── Homepage editorial content (shown only before any analysis) ───────────────

# Pre-curated editorial intelligence cards — realistic-looking prior analyses
FEATURED = [
    {
        "title": "Tesla Model Y",
        "cat": "RANKING · AUTOMOTIVE",
        "img": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=1200&q=80",
        "insight": "Long-term owners love the tech and range — but quality control debates persist across r/TeslaMotors and YouTube owner reviews.",
        "score": 4.2, "score_color": "#22d3a0",
        "stars": "★★★★☆",
        "pos": 61, "mix": 22, "neg": 17,
        "sources": "143 Reddit + YouTube opinions",
        "query": "Tesla Model Y",
    },
]

GRID_CARDS = [
    {
        "title": "Sony WH-1000XM5",
        "cat": "HEADPHONES",
        "img": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&q=80",
        "insight": "Near-universal praise. Community consensus: best ANC headphones available.",
        "score": 4.6, "score_color": "#22d3a0",
        "meta": "58% positive · 214 opinions",
        "query": "Sony WH-1000XM5 headphones",
    },
    {
        "title": "Best Espresso Machine <$500",
        "cat": "HOME · COFFEE",
        "img": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=600&q=80",
        "insight": "Breville Barista Express dominates. Gaggia Classic Pro is the budget favourite.",
        "score": None, "score_color": "#f59e0b",
        "meta": "Strong consensus · 178 opinions",
        "query": "best espresso machine under $500",
    },
    {
        "title": "Best Golf Courses — Bay Area",
        "cat": "SPORTS · LOCAL",
        "img": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=600&q=80",
        "insight": "Locals debate Harding Park vs TPC Harding. Pasatiempo named hidden gem.",
        "score": None, "score_color": "#6366f1",
        "meta": "Active debate · 92 opinions",
        "query": "best golf course in bay area",
    },
    {
        "title": "MacBook Air M3",
        "cat": "TECH · LAPTOPS",
        "img": "https://images.unsplash.com/photo-1611186871525-9769_...",
        "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&q=80",
        "insight": "Reddit's most recommended laptop. M3 chip upgrade called a meaningful step.",
        "score": 4.7, "score_color": "#22d3a0",
        "meta": "73% positive · 301 opinions",
        "query": "MacBook Air M3",
    },
]

DEBATES = [
    {
        "icon": "🎮", "icon_bg": "rgba(99,102,241,0.15)", "icon_border": "rgba(99,102,241,0.3)",
        "tag": "HOT", "tag_color": "#f87171", "tag_bg": "rgba(248,113,113,0.12)",
        "title": "Nintendo Switch 2 — Is the $449 price tag justified?",
        "sub": "r/NintendoSwitch · r/gaming · YouTube reviews · 340+ opinions analyzed",
        "stat": "51%", "stat_color": "#f87171", "stat_lbl": "think it's overpriced",
        "query": "Nintendo Switch 2",
    },
    {
        "icon": "🎧", "icon_bg": "rgba(34,211,160,0.12)", "icon_border": "rgba(34,211,160,0.25)",
        "tag": "CONSENSUS", "tag_color": "#22d3a0", "tag_bg": "rgba(34,211,160,0.1)",
        "title": "Sony WH-1000XM5 vs Bose QC45 — ANC debate has a clear winner",
        "sub": "r/headphones · r/audiophile · YouTube shootouts · 214 opinions",
        "stat": "4.6", "stat_color": "#22d3a0", "stat_lbl": "Sony wins /5",
        "query": "Sony WH-1000XM5 headphones",
    },
    {
        "icon": "🚗", "icon_bg": "rgba(245,158,11,0.12)", "icon_border": "rgba(245,158,11,0.25)",
        "tag": "DEBATED", "tag_color": "#f59e0b", "tag_bg": "rgba(245,158,11,0.1)",
        "title": "Tesla Model Y long-term reliability — owners divided after 2 years",
        "sub": "r/TeslaMotors · r/electricvehicles · YouTube vlogs · 143 opinions",
        "stat": "61%", "stat_color": "#22d3a0", "stat_lbl": "still recommend",
        "query": "Tesla Model Y",
    },
    {
        "icon": "☕", "icon_bg": "rgba(99,102,241,0.1)", "icon_border": "rgba(99,102,241,0.2)",
        "tag": "TRENDING", "tag_color": "#818cf8", "tag_bg": "rgba(99,102,241,0.12)",
        "title": "Breville vs Gaggia espresso machines — which wins under $500?",
        "sub": "r/espresso · r/Coffee · YouTube reviews · 178 opinions",
        "stat": "#1", "stat_color": "#818cf8", "stat_lbl": "Breville ranked",
        "query": "best espresso machine under $500",
    },
    {
        "icon": "🖥️", "icon_bg": "rgba(34,211,160,0.1)", "icon_border": "rgba(34,211,160,0.2)",
        "tag": "STRONG BUY", "tag_color": "#22d3a0", "tag_bg": "rgba(34,211,160,0.1)",
        "title": "MacBook Air M3 — Reddit's most recommended laptop three months running",
        "sub": "r/apple · r/macbook · r/laptops · YouTube reviews · 301 opinions",
        "stat": "4.7", "stat_color": "#22d3a0", "stat_lbl": "community score",
        "query": "MacBook Air M3",
    },
]

if not run_analysis:
    # ── Hero insight card ─────────────────────────────────────────────────────
    st.markdown('<div class="ih-home-lbl">Featured Intelligence</div>', unsafe_allow_html=True)

    feat = FEATURED[0]
    pos_w = feat["pos"]; mix_w = feat["mix"]; neg_w = feat["neg"]
    st.markdown(f"""
<div class="ih-hero-card">
  <img class="ih-hero-card-img" src="{feat['img']}" alt="{feat['title']}"/>
  <div class="ih-hero-card-body">
    <div class="ih-hero-card-top">
      <span class="ih-hero-card-cat">{feat['cat']}</span>
      <span class="ih-hero-card-sources">{feat['sources']}</span>
    </div>
    <div class="ih-hero-card-bottom">
      <div class="ih-hero-card-title">{feat['title']}</div>
      <div class="ih-hero-card-insight">{feat['insight']}</div>
      <div class="ih-hero-card-stats">
        <div>
          <div class="ih-hero-score {_score_cls(feat['score'])}" style="color:{feat['score_color']}">{feat['score']}<span style="font-size:0.9rem;color:#475569;font-weight:400">/5</span></div>
          <div class="ih-hero-stars">{feat['stars']}</div>
        </div>
        <div class="ih-sentiment-strip">
          <div class="ih-sent-seg"><div class="ih-sent-bar" style="background:#22d3a0;width:{pos_w//3}px"></div><span style="color:#22d3a0">{feat['pos']}% positive</span></div>
          <div class="ih-sent-seg"><div class="ih-sent-bar" style="background:#f59e0b;width:{mix_w//3}px"></div><span style="color:#f59e0b">{feat['mix']}% mixed</span></div>
          <div class="ih-sent-seg"><div class="ih-sent-bar" style="background:#f87171;width:{neg_w//3}px"></div><span style="color:#f87171">{feat['neg']}% critical</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.button(f"Analyze {feat['title']} →", key="hero_btn", type="secondary"):
        st.session_state["pending_query"] = feat["query"]
        st.session_state["run_analysis"] = True
        st.rerun()

    # ── Insight grid ──────────────────────────────────────────────────────────
    st.markdown('<div class="ih-home-lbl" style="margin-top:1.5rem">Recent Analyses</div>', unsafe_allow_html=True)
    grow = [st.columns(2), st.columns(2)]
    for idx, card in enumerate(GRID_CARDS):
        with grow[idx // 2][idx % 2]:
            score_html = (
                f'<div class="ih-insight-card-score {_score_cls(card["score"])}" style="color:{card["score_color"]}">{card["score"]}</div>'
                if card["score"] else
                f'<div class="ih-insight-card-score" style="color:{card["score_color"]}">◈</div>'
            )
            st.markdown(f"""
<div class="ih-insight-card">
  <img class="ih-insight-card-img" src="{card['img']}" alt="{card['title']}"/>
  <div class="ih-insight-card-body">
    <div class="ih-insight-card-cat">{card['cat']}</div>
    <div class="ih-insight-card-title">{card['title']}</div>
    <div class="ih-insight-card-insight">{card['insight']}</div>
    <div class="ih-insight-card-foot">
      {score_html}
      <span class="ih-insight-card-meta">{card['meta']}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            if st.button("Analyze →", key=f"grid_{idx}", use_container_width=True, type="secondary"):
                st.session_state["pending_query"] = card["query"]
                st.session_state["run_analysis"] = True
                st.rerun()

    # ── Active internet debates ───────────────────────────────────────────────
    st.markdown('<div class="ih-home-lbl" style="margin-top:1.75rem">Active Internet Debates</div>', unsafe_allow_html=True)

    for idx, d in enumerate(DEBATES):
        col_card, col_btn = st.columns([13, 1])
        with col_card:
            st.markdown(f"""
<div class="ih-debate-item">
  <div class="ih-debate-left">
    <div class="ih-debate-icon" style="background:{d['icon_bg']};border:1px solid {d['icon_border']}">{d['icon']}</div>
    <span class="ih-debate-tag" style="color:{d['tag_color']};background:{d['tag_bg']}">{d['tag']}</span>
  </div>
  <div class="ih-debate-body">
    <div class="ih-debate-title">{d['title']}</div>
    <div class="ih-debate-sub">{d['sub']}</div>
  </div>
  <div class="ih-debate-right">
    <div class="ih-debate-stat" style="color:{d['stat_color']}">{d['stat']}</div>
    <div class="ih-debate-stat-lbl">{d['stat_lbl']}</div>
  </div>
</div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown('<div style="padding-top:0.75rem"></div>', unsafe_allow_html=True)
            if st.button("→", key=f"deb_{idx}", use_container_width=True, type="secondary"):
                st.session_state["pending_query"] = d["query"]
                st.session_state["run_analysis"] = True
                st.rerun()

# ── Analysis ──────────────────────────────────────────────────────────────────

query = st.session_state.get("query", "")

if run_analysis and query.strip():
    try:
        status = st.status("Analyzing…", expanded=True)
        with status:
            st.write("Detecting query intent…")
            intent_schema = llm_service.detect_intent_and_schema(query)
            intent = (
                QueryIntent(intent_schema.intent)
                if intent_schema.intent in ["RANKING", "SOLUTION", "GENERIC"]
                else QueryIntent.GENERIC
            )

            use_cross = len(selected_platforms) > 1 or (
                len(selected_platforms) == 1 and selected_platforms[0] != Platform.REDDIT
            )

            def _normalize(review):
                return {
                    "id":       review.get("id")        if isinstance(review, dict) else review.id,
                    "text":     review.get("text")       if isinstance(review, dict) else review.text,
                    "upvotes":  review.get("upvotes", 0) if isinstance(review, dict) else getattr(review, "upvotes", 0),
                    "permalink":review.get("permalink","")if isinstance(review, dict) else getattr(review, "permalink",""),
                }

            if use_cross:
                ps = " + ".join(p.value.title() for p in selected_platforms)
                st.write(f"Gathering community reviews from {ps}…")
                start_time = time.time()

                def _scrape_then_annotate(platform):
                    service = cross_platform_manager.platforms[platform]
                    ck = ("scrape_platform_v1", platform.value, query.lower().strip(), limit)
                    raw = cross_platform_manager._scrape_cache.get(ck)
                    if raw is None:
                        raw = service.scrape(query, limit)
                        cross_platform_manager._scrape_cache.set(ck, raw, expire=3600)
                    pc = [_normalize(r) for r in raw]
                    pc = llm_service.filter_relevant_comments(pc, query)
                    pa = llm_service.annotate_comments_with_gpt(pc, intent_schema.aspects, intent_schema.entity_type, query)
                    return platform.value, raw, pc, pa

                platform_breakdown, comments, annos = {}, [], []
                with ThreadPoolExecutor(max_workers=len(selected_platforms)) as ex:
                    futs = {ex.submit(_scrape_then_annotate, p): p for p in selected_platforms}
                    for f in as_completed(futs):
                        pn, raw, pc, pa = f.result()
                        platform_breakdown[pn] = raw
                        comments.extend(pc); annos.extend(pa)

                search_time = time.time() - start_time
                reviews = [r for rlist in platform_breakdown.values() for r in rlist]
                cross_platform_manager.aggregate_results(platform_breakdown, query, intent)

            else:
                st.write("Gathering community reviews from Reddit…")
                start_time = time.time()
                reviews = reddit_service.scrape(query, limit, subreddit_count)
                search_time = time.time() - start_time
                platform_breakdown = {"reddit": reviews}
                comments = [_normalize(r) for r in reviews]
                comments = llm_service.filter_relevant_comments(comments, query)
                st.write(f"Reading and understanding {len(comments)} relevant reviews…")
                annos = llm_service.annotate_comments_with_gpt(
                    comments, intent_schema.aspects, intent_schema.entity_type, query
                )

            upvote_map = {c["id"]: c["upvotes"] for c in comments}
            anno_map   = {a.comment_id: a for a in annos}

            st.write("Building consensus and summarizing…")

            if intent_schema.intent == "RANKING":
                ranking = rank_entities_with_relaxation(annos, upvote_map, intent_schema.entity_type, query=query)
                if intent_schema.entity_type and ranking:
                    valid = set(llm_service.filter_entities_by_type([e.name for e in ranking], intent_schema.entity_type))
                    ranking = [e for e in ranking if e.name in valid]
                ranking = llm_service.validate_entity_locations(ranking, query)
                for item in ranking:
                    item.quotes = [c["text"][:240]+"…" for c in comments if item.name.lower() in c["text"].lower()][:3]

                ranking_data = [{"name": e.name, "overall_stars": e.overall_stars, "mentions": e.mentions, "quotes": e.quotes} for e in ranking]
                summary = llm_service.summarize_ranking_with_gpt(query, ranking_data)
                payload = {
                    "query": query, "intent": intent_schema.intent, "summary": summary,
                    "metadata": {"timestamp": time.time()},
                    "ranking": [{"name": e.name, "overall_stars": e.overall_stars, "aspect_scores": e.aspect_scores,
                                 "mentions": e.mentions, "confidence": e.confidence, "quotes": e.quotes} for e in ranking],
                }

            elif intent_schema.intent == "SOLUTION":
                clusters = defaultdict(list)
                for anno in annos:
                    if anno.solution_key:
                        clusters[anno.solution_key].append(anno)
                sol_clusters = [{"title": ck, "steps": [], "caveats": [], "evidence_count": len(cv)}
                                for ck, cv in clusters.items() if len(cv) >= 2]
                summary = llm_service.summarize_solutions_with_gpt(query, sol_clusters)
                payload = {"query": query, "intent": intent_schema.intent, "summary": summary,
                           "metadata": {"timestamp": time.time()}, "solutions": sol_clusters}

            else:  # GENERIC
                overall, aspect_averages = aggregate_generic(intent_schema.aspects, annos, upvote_map)
                pos_annos = sorted([a for a in annos if a.overall_score >= 3.5], key=lambda a: a.overall_score, reverse=True)
                quotes = []
                for anno in pos_annos[:8]:
                    c = next((x for x in comments if x["id"] == anno.comment_id), None)
                    if c:
                        quotes.append(c["text"][:240]+"…")
                summary = llm_service.summarize_generic_with_gpt(query, aspect_averages, overall, quotes)
                payload = {"query": query, "intent": intent_schema.intent, "summary": summary,
                           "metadata": {"timestamp": time.time()},
                           "overall": overall, "aspects": aspect_averages, "quotes": quotes}

            status.update(label=f"Analysis complete  ·  {len(comments)} relevant / {len(reviews)} scraped  ·  {search_time:.1f}s", state="complete")

        # ── RESULTS ──────────────────────────────────────────────────────────

        if not reviews:
            st.warning("No reviews found. Try a broader search term or increase depth in settings.")
        else:
            annotated_ids = set(anno_map.keys())
            relevant_reviews = [r for r in reviews if (r.get("id") if isinstance(r, dict) else getattr(r, "id", "")) in annotated_ids]
            meaningful = [r for r in relevant_reviews if len((r.get("text") or "")) >= 100]

            # ── Header ────────────────────────────────────────────────────────
            badge_cls = f"ih-badge-{intent_schema.intent}"
            entity_part = (
                f'<span class="ih-meta-dot">·</span><span class="ih-meta-txt">{intent_schema.entity_type}</span>'
                if intent_schema.entity_type else ""
            )
            aspects_preview = ", ".join(intent_schema.aspects[:5])
            if len(intent_schema.aspects) > 5:
                aspects_preview += "…"

            st.markdown(
                f'<div class="ih-result-header">'
                f'<div class="ih-query-title">{query}</div>'
                f'<div class="ih-meta-row">'
                f'<span class="ih-intent-badge {badge_cls}">{intent_schema.intent}</span>'
                f'{entity_part}'
                f'<span class="ih-meta-dot">·</span>'
                f'<span class="ih-meta-txt">{aspects_preview}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # ── Stats strip ───────────────────────────────────────────────────
            plat_label = " + ".join(p.title() for p in platform_breakdown.keys())
            if intent_schema.intent == "RANKING":
                s3_val, s3_lbl = str(len(payload.get("ranking", []))), "Entities ranked"
            elif intent_schema.intent == "SOLUTION":
                s3_val, s3_lbl = str(len(payload.get("solutions", []))), "Solutions found"
            else:
                s3_val, s3_lbl = f"{payload.get('overall', 0):.1f}", "Avg score /5"

            st.markdown(
                f'<div class="ih-stats-strip">'
                f'<div class="ih-stat"><div class="ih-stat-val">{len(comments)}</div><div class="ih-stat-lbl">Reviews analyzed</div></div>'
                f'<div class="ih-stat"><div class="ih-stat-val">{len(meaningful)}</div><div class="ih-stat-lbl">Detailed reviews</div></div>'
                f'<div class="ih-stat"><div class="ih-stat-val">{s3_val}</div><div class="ih-stat-lbl">{s3_lbl}</div></div>'
                f'<div class="ih-stat"><div class="ih-stat-val">{search_time:.1f}s</div><div class="ih-stat-lbl">Analysis time</div></div>'
                f'<div class="ih-stat"><div class="ih-stat-val">{plat_label}</div><div class="ih-stat-lbl">Sources</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── AI Summary card ───────────────────────────────────────────────
            avg_conf = (
                sum(i.get("confidence", 0.5) for i in payload.get("ranking", [{"confidence": 0.5}]))
                / max(len(payload.get("ranking", [1])), 1)
            ) if intent_schema.intent == "RANKING" else 0.6
            conf_lbl, conf_color = _confidence_label(avg_conf)

            st.markdown(
                f'<div class="ih-summary-card">'
                f'<div class="ih-summary-eyebrow">◈ AI Analysis</div>'
                f'<div class="ih-summary-text">{payload["summary"]}</div>'
                f'<div class="ih-summary-foot">'
                f'<div class="ih-conf-dot" style="background:{conf_color}"></div>'
                f'<span style="color:{conf_color}">{conf_lbl}</span>'
                f'<span>·</span>'
                f'<span>{len(meaningful)} substantive reviews</span>'
                f'<span class="ih-foot-right">{len(intent_schema.aspects)} dimensions · {len(platform_breakdown)} source{"s" if len(platform_breakdown)>1 else ""}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # ── Tabs ──────────────────────────────────────────────────────────
            tab_lbl_2 = {"RANKING": "Rankings", "SOLUTION": "Solutions", "GENERIC": "Insights"}.get(intent_schema.intent, "Insights")
            tab_ov, tab_res, tab_ev = st.tabs(["Overview", tab_lbl_2, "Evidence"])

            # ═════════════════════════════════════════════════════════════════
            # OVERVIEW TAB
            # ═════════════════════════════════════════════════════════════════
            with tab_ov:

                if intent_schema.intent == "GENERIC":
                    ov = payload.get("overall", 3.0)
                    v_cls = "verdict-high" if ov >= 4.0 else "verdict-mid" if ov >= 3.0 else "verdict-low"
                    v_txt = "Highly recommended" if ov >= 4.0 else "Generally positive" if ov >= 3.0 else "Mixed reviews"
                    c_score, c_asp = st.columns([1, 2])
                    with c_score:
                        st.markdown(
                            f'<div class="ih-score-hero">'
                            f'<div class="ih-score-num {_score_cls(ov)}">{ov:.1f}'
                            f'<span class="ih-score-denom">/5</span></div>'
                            f'<div class="ih-score-stars">{_stars(ov)}</div>'
                            f'<div class="ih-score-verdict {v_cls}">{v_txt}</div>'
                            f'</div>', unsafe_allow_html=True)
                    with c_asp:
                        st.markdown('<div class="ih-section-hdr">Dimension breakdown</div>', unsafe_allow_html=True)
                        bars = "".join(_aspect_bar(k, v) for k, v in payload.get("aspects", {}).items())
                        st.markdown(bars or '<p style="color:#475569;font-size:.82rem">No aspect data</p>', unsafe_allow_html=True)

                if intent_schema.intent == "RANKING" and payload.get("ranking"):
                    st.markdown('<div class="ih-section-hdr">Top consensus picks</div>', unsafe_allow_html=True)
                    rows = ""
                    for i, item in enumerate(payload["ranking"][:3], 1):
                        pc = ["rank-gold", "rank-silver", "rank-bronze"][i - 1]
                        sc = _score_cls(item["overall_stars"])
                        rows += (
                            f'<div class="ih-ov-rank-row">'
                            f'<div class="ih-ov-pos {pc}">#{i}</div>'
                            f'<div class="ih-ov-name">{item["name"]}</div>'
                            f'<div class="ih-ov-stars">{_stars(item["overall_stars"])}</div>'
                            f'<div class="ih-ov-score {sc}">{item["overall_stars"]:.1f}</div>'
                            f'</div>'
                        )
                    st.markdown(rows, unsafe_allow_html=True)

                if intent_schema.intent == "SOLUTION" and payload.get("solutions"):
                    st.markdown('<div class="ih-section-hdr">Top solutions</div>', unsafe_allow_html=True)
                    rows = "".join(
                        f'<div class="ih-solution-item">'
                        f'<div class="ih-sol-idx">{i}</div>'
                        f'<div><div class="ih-sol-name">{c["title"]}</div>'
                        f'<div class="ih-sol-count">{c["evidence_count"]} supporting comments</div></div>'
                        f'</div>'
                        for i, c in enumerate(payload["solutions"][:4], 1)
                    )
                    st.markdown(rows, unsafe_allow_html=True)

                if use_cross:
                    st.markdown('<div class="ih-section-hdr">Source breakdown</div>', unsafe_allow_html=True)
                    plat_html = '<div class="ih-plat-strip">' + "".join(
                        f'<div class="ih-plat-item"><span class="ih-plat-count">{len(v)}</span>{k.title()} reviews</div>'
                        for k, v in platform_breakdown.items()
                    ) + '</div>'
                    st.markdown(plat_html, unsafe_allow_html=True)

                if show_debug:
                    with st.expander("Raw data — first 3 reviews"):
                        import pandas as pd
                        st.dataframe(pd.DataFrame(reviews[:3])[["id","author","upvotes","permalink","text"]])
                    if not use_cross:
                        with st.expander("Reddit search plan"):
                            try:
                                plan = reddit_service._plan_search(query)
                                st.json({"terms":plan.terms,"subreddits":plan.subreddits,"time_filter":plan.time_filter,
                                         "strategies":plan.strategies,"min_comment_score":plan.min_comment_score})
                            except Exception as e:
                                st.caption(f"Plan unavailable: {e}")

            # ═════════════════════════════════════════════════════════════════
            # RANKINGS / SOLUTIONS / INSIGHTS TAB
            # ═════════════════════════════════════════════════════════════════
            with tab_res:

                if intent_schema.intent == "RANKING":
                    if payload["ranking"]:
                        st.markdown('<div class="ih-section-hdr">Full ranking</div>', unsafe_allow_html=True)
                        for i, item in enumerate(payload["ranking"][: SearchConstants.MAX_ENTITIES_TO_DISPLAY], 1):
                            pc = ("rank-gold" if i == 1 else "rank-silver" if i == 2 else "rank-bronze" if i == 3 else "")
                            card_cls = ("ih-rank-card-top3 " if i <= 3 else "") + (
                                "ih-rank-card-gold" if i == 1 else "ih-rank-card-silver" if i == 2 else "ih-rank-card-bronze" if i == 3 else ""
                            )
                            sc = _score_cls(item["overall_stars"])
                            conf_lbl2, conf_color2 = _confidence_label(item.get("confidence", 0.5))
                            verdict = _entity_verdict(item)

                            # Aspect bars (top 5, sorted high→low)
                            aspect_bars = ""
                            if item.get("aspect_scores"):
                                sorted_asp = sorted(item["aspect_scores"].items(), key=lambda x: x[1], reverse=True)[:5]
                                aspect_bars = '<div class="ih-aspects">' + "".join(_aspect_bar(k, v) for k, v in sorted_asp) + '</div>'

                            conf_bg = conf_color2.replace("#","")
                            conf_pill = (
                                f'<span class="ih-conf-pill" '
                                f'style="color:{conf_color2};background:{conf_color2}1a;border:1px solid {conf_color2}33">'
                                f'{conf_lbl2}</span>'
                            )

                            st.markdown(
                                f'<div class="ih-rank-card {card_cls}">'
                                f'<div class="ih-rank-head">'
                                f'<div class="ih-rank-num {pc}">#{i}</div>'
                                f'<div class="ih-rank-info">'
                                f'<div class="ih-rank-name">{item["name"]}</div>'
                                f'<div class="ih-rank-stars">{_stars(item["overall_stars"])}</div>'
                                f'<div class="ih-rank-verdict">{verdict}</div>'
                                f'</div>'
                                f'<div class="ih-rank-score-block">'
                                f'<div class="ih-rank-big-score {sc}">{item["overall_stars"]:.1f}</div>'
                                f'<div class="ih-rank-mentions">{item["mentions"]} mentions</div>'
                                f'</div>'
                                f'</div>'
                                f'{aspect_bars}'
                                f'<div class="ih-rank-foot">'
                                f'{conf_pill}'
                                f'<span>·</span>'
                                f'<span>{item.get("confidence", 0.5):.0%} confidence score</span>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            if item.get("quotes"):
                                with st.expander(f"Community quotes about {item['name']}"):
                                    for q_text in item["quotes"][:3]:
                                        st.markdown(f'<div class="ih-quote">{q_text}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No ranked entities found. Try a more specific query or increase depth in settings.")

                elif intent_schema.intent == "SOLUTION":
                    if payload["solutions"]:
                        rows = "".join(
                            f'<div class="ih-solution-item">'
                            f'<div class="ih-sol-idx">{i}</div>'
                            f'<div><div class="ih-sol-name">{c["title"]}</div>'
                            f'<div class="ih-sol-count">{c["evidence_count"]} supporting comments</div></div>'
                            f'</div>'
                            for i, c in enumerate(payload["solutions"], 1)
                        )
                        st.markdown(rows, unsafe_allow_html=True)
                    else:
                        st.info("No solution clusters found. Try a more specific query or increase depth in settings.")

                else:  # GENERIC
                    if payload.get("quotes"):
                        st.markdown('<div class="ih-section-hdr">Key community voices</div>', unsafe_allow_html=True)
                        for q_text in payload["quotes"][:6]:
                            st.markdown(f'<div class="ih-quote">{q_text}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No notable quotes found.")

                    st.markdown('<div class="ih-section-hdr">All dimensions</div>', unsafe_allow_html=True)
                    bars = "".join(_aspect_bar(k, v) for k, v in payload.get("aspects", {}).items())
                    st.markdown(bars or '<p style="color:#475569;font-size:.82rem">No aspect data</p>', unsafe_allow_html=True)

            # ═════════════════════════════════════════════════════════════════
            # EVIDENCE TAB
            # ═════════════════════════════════════════════════════════════════
            with tab_ev:
                detailed = [r for r in relevant_reviews if len((r.get("text") or "")) >= 100]

                if not detailed:
                    st.info("No detailed reviews found for this query.")
                else:
                    def _enrich(r):
                        rid = r.get("id") if isinstance(r, dict) else r.id
                        anno = anno_map.get(rid)
                        sc = anno.overall_score if anno and hasattr(anno, "overall_score") else None
                        base = dict(r) if isinstance(r, dict) else {
                            "id": rid, "text": getattr(r, "text", ""),
                            "upvotes": getattr(r, "upvotes", 0),
                            "permalink": getattr(r, "permalink", ""),
                            "url": getattr(r, "url", ""),
                            "author": getattr(r, "author", "anonymous"),
                        }
                        base["_score"] = sc
                        return base

                    enriched = [_enrich(r) for r in detailed]
                    positive = sorted([r for r in enriched if r["_score"] is not None and r["_score"] >= 3.8], key=lambda r: -(r.get("upvotes",0) or 0))
                    critical = sorted([r for r in enriched if r["_score"] is not None and r["_score"] < 2.8],  key=lambda r: -(r.get("upvotes",0) or 0))
                    mixed    = sorted([r for r in enriched if r["_score"] is None or 2.8 <= r["_score"] < 3.8], key=lambda r: -(r.get("upvotes",0) or 0))

                    def _render_ev(review_list, cap=6):
                        if not review_list:
                            return ""
                        html = ""
                        for rev in review_list[:cap]:
                            permalink = rev.get("permalink","") or ""
                            url       = rev.get("url","") or ""
                            author    = rev.get("author") or "anonymous"
                            upvotes   = rev.get("upvotes", 0) or 0
                            sc        = rev.get("_score")
                            src       = _source_tag(rev)
                            upcls     = "ih-upvotes-high" if upvotes >= 100 else ""

                            if permalink:
                                ahtml = f'<a href="https://reddit.com{permalink}" target="_blank">{author}</a>'
                            elif url:
                                ahtml = f'<a href="{url}" target="_blank">{author}</a>'
                            else:
                                ahtml = author

                            sent_html = ""
                            if sc is not None:
                                sl, sc_color, sc_bg = _sentiment_label(sc)
                                sent_html = f'<span class="ih-sent" style="color:{sc_color};background:{sc_bg};border:1px solid {sc_color}22">{sl}</span>'

                            html += (
                                f'<div class="ih-evidence-item">'
                                f'<div class="ih-ev-meta">{src}{sent_html}'
                                f'<span class="ih-ev-author">{ahtml}</span>'
                                f'<span class="ih-upvotes {upcls}">▲ {upvotes:,}</span>'
                                f'</div>'
                                f'<div class="ih-ev-text">{_excerpt(rev.get("text",""))}</div>'
                                f'</div>'
                            )
                        return html

                    st.markdown(
                        f'<div class="ih-section-hdr">{len(enriched)} substantive reviews · grouped by sentiment</div>',
                        unsafe_allow_html=True,
                    )

                    if positive:
                        st.markdown(f'<div class="ih-evidence-group-hdr" style="color:#22d3a0">Positive voices ({len(positive)})</div>', unsafe_allow_html=True)
                        st.markdown(_render_ev(positive, 5), unsafe_allow_html=True)
                        if len(positive) > 5:
                            with st.expander(f"{len(positive)-5} more positive reviews"):
                                st.markdown(_render_ev(positive[5:], 50), unsafe_allow_html=True)

                    if critical:
                        st.markdown(f'<div class="ih-evidence-group-hdr" style="color:#f87171">Critical voices ({len(critical)})</div>', unsafe_allow_html=True)
                        st.markdown(_render_ev(critical, 5), unsafe_allow_html=True)
                        if len(critical) > 5:
                            with st.expander(f"{len(critical)-5} more critical reviews"):
                                st.markdown(_render_ev(critical[5:], 50), unsafe_allow_html=True)

                    if mixed:
                        st.markdown(f'<div class="ih-evidence-group-hdr" style="color:#64748b">Mixed &amp; neutral ({len(mixed)})</div>', unsafe_allow_html=True)
                        st.markdown(_render_ev(mixed, 4), unsafe_allow_html=True)
                        if len(mixed) > 4:
                            with st.expander(f"{len(mixed)-4} more mixed reviews"):
                                st.markdown(_render_ev(mixed[4:], 50), unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {e}")
        st.info("Please try again with a different search term.")

elif run_analysis and not query.strip():
    st.warning("Enter a search term to begin.")


def main():
    """Entry point — Streamlit runs the module directly."""
    pass
