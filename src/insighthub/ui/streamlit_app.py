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
from insighthub.core.constants import SearchConstants, ConfidenceConfig
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

# ── brand marks (inline SVG — self-contained, used consistently across the UI) ──

_RD_ICON = (
    '<span class="brandic"><svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="10" cy="10" r="10" fill="#FF4500"/>'
    '<circle cx="13.9" cy="4.7" r="1.2" fill="#fff"/>'
    '<path d="M10.1 8.2 L11.2 4.9 L13.6 5.3" stroke="#fff" stroke-width="0.9" fill="none"/>'
    '<circle cx="4.4" cy="10.4" r="1.4" fill="#fff"/><circle cx="15.6" cy="10.4" r="1.4" fill="#fff"/>'
    '<ellipse cx="10" cy="11.6" rx="5.3" ry="3.7" fill="#fff"/>'
    '<circle cx="7.9" cy="11" r="1" fill="#FF4500"/><circle cx="12.1" cy="11" r="1" fill="#FF4500"/>'
    '<path d="M7.9 13.3 q2.1 1.5 4.2 0" stroke="#FF4500" stroke-width="0.85" fill="none" stroke-linecap="round"/>'
    '</svg></span>')

_YT_ICON = (
    '<span class="brandic"><svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="1" y="4.5" width="22" height="15" rx="4" fill="#FF0000"/>'
    '<path d="M9.8 9 L15.4 12 L9.8 15 Z" fill="#fff"/></svg></span>')


def _rel_time(ts) -> str:
    """Unix timestamp → compact relative age ('3d ago'); '' when unknown."""
    try:
        d = time.time() - float(ts)
    except (TypeError, ValueError):
        return ""
    if d < 0:
        return ""
    for cut, div, unit in ((3600, 60, "m"), (86400, 3600, "h"), (86400 * 30, 86400, "d"),
                           (86400 * 365, 86400 * 30, "mo")):
        if d < cut:
            return f"{max(1, int(d // div))}{unit} ago"
    return f"{int(d // (86400 * 365))}y ago"


def _fmt_views(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M views"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K views"
    return f"{n} views"


def _source_tag(review: dict) -> str:
    source = review.get("source", "") or ""
    url = review.get("url", "") or ""
    permalink = review.get("permalink", "") or ""
    combined = (source + url + permalink).lower()
    if "youtube" in combined or "youtu.be" in combined:
        return f'<span class="ih-src ih-src-youtube">{_YT_ICON}YouTube</span>'
    if permalink:
        return f'<span class="ih-src ih-src-reddit">{_RD_ICON}Reddit</span>'
    return ""

def _entity_verdict(item: dict) -> str:
    aspects = item.get("aspect_scores", {})
    # Ranking confidence (four-factor), NOT GPT extraction confidence.
    conf = item.get("confidence_score", item.get("confidence", 0.5))
    conf_lbl = "strong consensus" if conf >= 0.6 else "moderate consensus" if conf >= 0.35 else "limited evidence"
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
.ih-rank-card-faint {
  opacity: 0.72;
  border-style: dashed;
  background: rgba(255,255,255,0.015);
}
.ih-also-tag {
  font-size: 0.66rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: none;
  color: #8a93a3;
  background: rgba(138,147,163,0.12);
  border: 1px solid rgba(138,147,163,0.25);
  border-radius: 6px;
  padding: 0.12rem 0.5rem;
  margin-left: 0.55rem;
  vertical-align: middle;
}

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
.ih-plat-sub { color: #475569; }

/* ── Homepage search area (search-first: the hero IS the product) ── */
.ih-search-area { padding: 9rem 1rem 3.25rem; text-align: center; }
.ih-search-heading {
  font-size: 4.1rem; font-weight: 800; color: #f1f5f9;
  letter-spacing: -0.045em; line-height: 1.08; margin-bottom: 1.15rem;
}
.ih-search-heading em { color: #6366f1; font-style: normal; }
.ih-search-sub { font-size: 1.15rem; color: #64748b; margin-bottom: 2rem; line-height: 1.6; }

/* ── Homepage section label ── */
.ih-home-lbl {
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.15em; color: #475569; font-family: 'JetBrains Mono', monospace;
  margin: 2rem 0 1rem; display: flex; align-items: center; gap: 0.7rem;
}
.ih-home-lbl::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

/* ── Example search cards (capability showcase; whole card clickable via
      invisible overlay button, same pattern as the explorer's expick cards) ── */
div[data-testid="stColumn"]:has(div[class*="st-key-exhome"]) { position: relative; }
div[class*="st-key-exhome"] { position: absolute; inset: 0; z-index: 6; margin: 0; width: 100% !important; height: 100% !important; }
div[class*="st-key-exhome"] button { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
.ih-excard {
  height: 208px; border-radius: 14px; overflow: hidden;
  border: 1px solid rgba(255,255,255,0.09);
  background: rgba(255,255,255,0.02);
  display: flex; flex-direction: column;
  transition: border-color 0.15s, transform 0.15s;
}
div[data-testid="stColumn"]:has(div[class*="st-key-exhome"]):hover .ih-excard {
  border-color: rgba(99,102,241,0.45); transform: translateY(-2px);
}
.ih-excard-img {
  flex: none; height: 152px; position: relative;
  background-size: cover; background-position: center;
}
.ih-excard-tag {
  position: absolute; top: 9px; left: 9px; z-index: 1;
  font-size: 0.52rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  font-family: 'JetBrains Mono', monospace;
  padding: 0.14rem 0.42rem; border-radius: 3px;
  background: rgba(13,15,22,0.72); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.35);
}
.ih-excard-title {
  flex: 1; display: flex; align-items: center; padding: 0.3rem 0.75rem;
  font-size: 0.8rem; font-weight: 700; color: #e2e8f0; line-height: 1.3;
  letter-spacing: -0.01em;
}

/* ── How it works (lightweight — must not compete with search) ── */
.ih-hiw-row { display: flex; gap: 0.9rem; margin-top: 0.3rem; }
.ih-hiw-step { flex: 1; text-align: center; padding: 0.4rem 0.4rem; }
.ih-hiw-ic {
  width: 36px; height: 36px; margin: 0 auto 0.5rem; border-radius: 50%;
  background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
  display: flex; align-items: center; justify-content: center; font-size: 0.9rem;
}
.ih-hiw-t { font-size: 0.78rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.25rem; }
.ih-hiw-d { font-size: 0.68rem; color: #64748b; line-height: 1.5; max-width: 210px; margin: 0 auto; }
.ih-hiw-strip {
  display: flex; margin-top: 1.25rem; border-radius: 12px;
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
  padding: 0.85rem 0.5rem;
}
.ih-hiw-fact { flex: 1; display: flex; gap: 0.7rem; padding: 0 1rem; align-items: flex-start; }
.ih-hiw-fact + .ih-hiw-fact { border-left: 1px solid rgba(255,255,255,0.06); }
.ih-hiw-fact-ic { font-size: 1rem; margin-top: 0.05rem; }
.ih-hiw-fact-t { font-size: 0.76rem; font-weight: 700; color: #cbd5e1; margin-bottom: 0.15rem; }
.ih-hiw-fact-d { font-size: 0.67rem; color: #64748b; line-height: 1.5; }

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
  background: rgba(255,255,255,0.045) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 16px !important;
  color: #f1f5f9 !important;
  font-size: 1.2rem !important;
  padding: 1.45rem 1.75rem !important;
  box-shadow: 0 10px 36px rgba(0,0,0,0.35), 0 0 0 1px rgba(99,102,241,0.08) !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(99,102,241,0.6) !important;
  box-shadow: 0 10px 36px rgba(0,0,0,0.35), 0 0 0 3px rgba(99,102,241,0.14) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #334155 !important; }
.stButton > button[kind="primary"] {
  background: #6366f1 !important;
  border: none !important;
  border-radius: 16px !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 1.08rem !important;
  padding: 1.38rem 1.5rem !important;
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

/* ── Progress timeline ── */
@keyframes ih-spinner { to { transform: rotate(360deg); } }
.ih-progress {
  padding: 0.85rem 1.1rem;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  margin-bottom: 1rem;
}
.ih-step {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.22rem 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  line-height: 1.4;
}
.ih-step-icon { font-size: 0.82rem; width: 16px; flex-shrink: 0; }
.ih-step-done .ih-step-icon { color: #22d3a0; }
.ih-step-done .ih-step-text { color: #475569; }
.ih-step-active .ih-step-text { color: #cbd5e1; }
.ih-step-spin {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 1.5px solid rgba(99,102,241,0.25);
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: ih-spinner 0.75s linear infinite;
  flex-shrink: 0;
}
.ih-progress-done {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #22d3a0;
  padding: 0.4rem 0;
  margin-bottom: 0.5rem;
}

/* ── Confidence explanation / source disclosure ── */
.ih-conf-explain {
  font-size: 0.72rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
}
.ih-source-disclosure {
  font-size: 0.72rem;
  color: #475569;
  font-family: 'JetBrains Mono', monospace;
  padding: 0.25rem 0 0.85rem;
  letter-spacing: 0.01em;
}

/* ── Mobile responsive (iPhone / narrow screens) ── */
@media (max-width: 640px) {
  /* Container: tighten horizontal padding only (preserve top padding for Streamlit header) */
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; padding-bottom: 3rem !important; }

  /* Search area */
  .ih-search-area { padding: 4rem 0.25rem 1.5rem; }
  .ih-search-heading { font-size: 2.15rem; letter-spacing: -0.03em; }
  .ih-search-sub { font-size: 0.88rem; margin-bottom: 1.1rem; }

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

  /* Example cards: columns stack full-width */
  .ih-excard { height: 232px; }
  .ih-excard-img { height: 176px; }
  .ih-excard-title { font-size: 0.92rem; }

  /* How it works: stack steps and trust facts vertically */
  .ih-hiw-row { flex-direction: column; gap: 1.1rem; }
  .ih-hiw-strip { flex-direction: column; gap: 0.9rem; }
  .ih-hiw-fact + .ih-hiw-fact { border-left: none; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.9rem; }

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
  <p class="ih-search-sub">AI-extracted consensus from Reddit &amp; YouTube.<br>Scored, ranked, and explained with evidence.</p>
</div>
""", unsafe_allow_html=True)

_, col_q, col_btn, _ = st.columns([0.7, 5.6, 1.0, 0.7])
with col_q:
    st.text_input(
        "query", key="search_input", label_visibility="collapsed",
        placeholder="Ask anything — products, places, debates, fixes…",
    )
with col_btn:
    analyze_clicked = st.button("Analyze", use_container_width=True, type="primary")

st.session_state["query"] = st.session_state.get("search_input", "")
run_analysis = st.session_state.pop("run_analysis", False) or analyze_clicked

# ── Homepage content (shown only before any analysis) ─────────────────────────

# Static example searches — a capability showcase spanning query types, NOT live
# or trending data. Swap for real trending analyses once persistent storage exists.
EXAMPLES = [
    {
        "title": "Best Korean food in New York",
        "tag": "Local",
        "img": "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=600&q=70",
        "query": "best Korean food in New York",
    },
    {
        "title": "Tesla Model Y",
        "tag": "Product",
        "img": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&q=70",
        "query": "Tesla Model Y",
    },
    {
        "title": "Top hotels in Las Vegas",
        "tag": "Travel",
        "img": "https://images.unsplash.com/photo-1581351721010-8cf859cb14a4?w=600&q=70",
        "query": "top hotels in Las Vegas",
    },
    {
        "title": "Best espresso machine under $500",
        "tag": "Shopping",
        "img": "https://images.unsplash.com/photo-1511920170033-f8396924c348?w=600&q=70",
        "query": "best espresso machine under $500",
    },
    {
        "title": "Best golf courses in Bay Area",
        "tag": "Local",
        "img": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=600&q=70",
        "query": "best golf courses in bay area",
    },
    {
        "title": "Nintendo Switch 2",
        "tag": "Consumer Tech",
        "img": "https://images.unsplash.com/photo-1578303512597-81e6cc155b3e?w=600&q=70",
        "query": "Nintendo Switch 2",
    },
]

if not run_analysis and not st.session_state.get("results"):
    # Homepage only: let the landing breathe on wide screens; results pages keep
    # the 960px width they were designed and verified at.
    st.markdown('<style>.block-container{max-width:1280px !important;}</style>', unsafe_allow_html=True)

    # ── Example searches: whole card clickable via invisible overlay button ───
    st.markdown('<div class="ih-home-lbl" style="margin-top:3.5rem">Start with an example</div>', unsafe_allow_html=True)

    def _run_example(q: str):
        st.session_state["pending_query"] = q
        st.session_state["run_analysis"] = True

    ex_cols = st.columns(len(EXAMPLES))
    for idx, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
        col.markdown(f"""
<div class="ih-excard">
  <div class="ih-excard-img" style="background-image:url({ex['img']})"><span class="ih-excard-tag">{ex['tag']}</span></div>
  <div class="ih-excard-title">{ex['title']}</div>
</div>""", unsafe_allow_html=True)
        col.button(ex["title"], key=f"exhome_{idx}", on_click=_run_example, args=(ex["query"],))

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown('<div class="ih-home-lbl" style="margin-top:1.75rem">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="ih-hiw-row">
  <div class="ih-hiw-step"><div class="ih-hiw-ic">📡</div><div class="ih-hiw-t">1. Collect</div><div class="ih-hiw-d">We scan thousands of real discussions across Reddit and YouTube.</div></div>
  <div class="ih-hiw-step"><div class="ih-hiw-ic">✨</div><div class="ih-hiw-t">2. Analyze</div><div class="ih-hiw-d">AI extracts opinions and sentiment, then scores the consensus.</div></div>
  <div class="ih-hiw-step"><div class="ih-hiw-ic">💬</div><div class="ih-hiw-t">3. Explain</div><div class="ih-hiw-d">You get ranked insights backed by real quotes and sources.</div></div>
</div>
<div class="ih-hiw-strip">
  <div class="ih-hiw-fact"><div class="ih-hiw-fact-ic">🛡️</div><div><div class="ih-hiw-fact-t">Real discussions</div><div class="ih-hiw-fact-d">No blogs. No ads. Just real people sharing real opinions.</div></div></div>
  <div class="ih-hiw-fact"><div class="ih-hiw-fact-ic">🎯</div><div><div class="ih-hiw-fact-t">Unbiased AI</div><div class="ih-hiw-fact-d">Our AI finds consensus, not just popular posts or loud opinions.</div></div></div>
  <div class="ih-hiw-fact"><div class="ih-hiw-fact-ic">🔍</div><div><div class="ih-hiw-fact-t">Transparent</div><div class="ih-hiw-fact-d">See sources, quotes, and scores behind every insight.</div></div></div>
</div>""", unsafe_allow_html=True)

# ── Results renderer (tabbed dashboard) ────────────────────────────────────────

_RESULTS_CSS = """
<style>
.ih-metric-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:0.85rem 1rem;text-align:center;}
.ih-metric-val{font-size:1.5rem;font-weight:700;color:#e2e8f0;line-height:1.1;}
.ih-metric-lbl{font-size:0.68rem;color:#8a93a3;text-transform:uppercase;letter-spacing:.04em;margin-top:.25rem;}
.ih-ov-hdr{font-size:.9rem;font-weight:700;color:#cbd5e1;margin:1.4rem 0 .6rem;
  border-left:3px solid #6366f1;padding-left:.5rem;}
.ih-ent-card{background:rgba(99,102,241,0.045);border:1px solid rgba(99,102,241,0.16);
  border-radius:12px;padding:.9rem 1rem;height:100%;position:relative;transition:border-color .15s;}
.ih-ent-card:hover{border-color:rgba(99,102,241,0.4);}
.ih-ent-top{display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem;}
.ih-ent-rank{font-size:.7rem;font-weight:700;color:#fff;background:#6366f1;border-radius:6px;
  padding:.05rem .4rem;}
.ih-ent-rank.g1{background:#c89b3c;} .ih-ent-rank.g2{background:#6b7a8d;} .ih-ent-rank.g3{background:#8a6040;}
.ih-ent-name{font-size:1.02rem;font-weight:700;color:#f1f5f9;margin:.35rem 0 .1rem;}
.ih-ent-big{font-size:1.35rem;font-weight:800;}
.ih-ent-verdict{font-size:.74rem;color:#94a3b8;margin:.2rem 0 .5rem;min-height:2.1em;}
.ih-chip{display:inline-block;font-size:.66rem;color:#a9b2c3;background:rgba(148,163,184,0.12);
  border:1px solid rgba(148,163,184,0.2);border-radius:20px;padding:.08rem .5rem;margin:0 .25rem .25rem 0;}
.ih-cbadge{font-size:.63rem;font-weight:600;border-radius:6px;padding:.08rem .45rem;}
.ih-asp-mini{display:flex;align-items:center;gap:.4rem;margin:.15rem 0;}
.ih-asp-mini .lbl{font-size:.64rem;color:#8a93a3;width:64px;flex:none;text-transform:capitalize;}
.ih-asp-mini .track{flex:1;height:5px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;}
.ih-asp-mini .fill{height:100%;border-radius:3px;}
.ih-asp-mini .v{font-size:.63rem;color:#94a3b8;width:22px;text-align:right;flex:none;}
.ih-sbar{display:flex;align-items:center;gap:.6rem;margin:.35rem 0;}
.ih-sbar .lbl{font-size:.74rem;width:70px;flex:none;}
.ih-sbar .track{flex:1;height:10px;background:rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;}
.ih-sbar .fill{height:100%;border-radius:6px;}
.ih-sbar .v{font-size:.72rem;color:#94a3b8;width:52px;text-align:right;flex:none;}
.ih-srccard{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:1rem;}
.ih-srccard h4{margin:0 0 .5rem;font-size:.82rem;color:#cbd5e1;}
.ih-srccard .row{display:flex;justify-content:space-between;font-size:.76rem;color:#94a3b8;padding:.15rem 0;}
.ih-srccard .row b{color:#e2e8f0;}
.ih-evcard{background:rgba(255,255,255,0.028);border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;padding:.75rem .9rem;margin-bottom:.6rem;}
.ih-evcard-head{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap;margin-bottom:.4rem;}
.ih-evcard-auth{font-size:.75rem;color:#cbd5e1;font-weight:600;}
.ih-evcard-sub{font-size:.7rem;color:#8a93a3;}
.ih-clamp3{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;
  font-size:.8rem;color:#b6c0cf;line-height:1.5;}
.ih-yt{display:flex;gap:.8rem;}
.ih-yt-thumb{width:168px;height:94px;border-radius:8px;object-fit:cover;flex:none;
  border:1px solid rgba(255,255,255,0.1);}
.ih-yt-body{flex:1;min-width:0;}
.ih-badge-src{font-size:.63rem;font-weight:700;border-radius:5px;padding:.06rem .4rem;}
.ih-badge-rd{color:#ff6b4a;background:rgba(255,107,74,0.12);border:1px solid rgba(255,107,74,0.3);}
.ih-badge-yt{color:#ff4a4a;background:rgba(255,74,74,0.12);border:1px solid rgba(255,74,74,0.3);}
.ih-open-link{font-size:.72rem;color:#818cf8;text-decoration:none;font-weight:600;}
.brandic{display:inline-flex;vertical-align:-3px;margin-right:.28rem;}
.brandic svg{width:15px;height:15px;}
</style>
"""


_OVERVIEW_CSS = """
<style>
.ov-metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:.9rem;margin:.4rem 0 1.2rem;}
.ov-mcard{background:#12161d;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1rem 1.1rem;}
.ov-mic{width:30px;height:30px;border-radius:8px;background:rgba(99,102,241,0.16);color:#a5b4fc;
  display:flex;align-items:center;justify-content:center;font-size:.9rem;margin-bottom:.6rem;}
.ov-mval{font-size:1.7rem;font-weight:800;color:#f1f5f9;line-height:1;}
.ov-mlbl{font-size:.8rem;color:#cbd5e1;font-weight:600;margin-top:.35rem;}
.ov-msub{font-size:.68rem;color:#7b8698;margin-top:.1rem;}
.ov-3col{display:grid;grid-template-columns:1fr 1.15fr .85fr;gap:.9rem;margin-bottom:1.6rem;}
.ov-panel{background:#12161d;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.1rem 1.2rem;}
.ov-ptitle{font-size:.9rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;}
.ov-senti{display:flex;align-items:center;gap:1.2rem;}
.ov-donut{width:110px;height:110px;border-radius:50%;position:relative;flex:none;}
.ov-donut-h{position:absolute;inset:26px;background:#12161d;border-radius:50%;}
.ov-legend{display:flex;flex-direction:column;gap:.55rem;font-size:.8rem;color:#b6c0cf;}
.ov-legend>div{display:flex;align-items:center;gap:.5rem;}
.ov-legend b{margin-left:auto;color:#e2e8f0;}
.ov-legend .dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.ov-asp{display:flex;align-items:center;gap:.7rem;margin:.55rem 0;}
.ov-asp .lbl{font-size:.76rem;color:#aab4c2;width:104px;flex:none;}
.ov-asp .track{flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;}
.ov-asp .fill{height:100%;border-radius:5px;}
.ov-asp .v{font-size:.74rem;color:#cbd5e1;width:34px;text-align:right;flex:none;font-weight:600;}
.ov-empty{color:#64748b;font-size:.8rem;}
.ov-srcrow{display:flex;align-items:center;gap:.7rem;margin-bottom:1rem;}
.ov-srcic{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.85rem;flex:none;color:#fff;}
.ov-srcic.rd{background:#ff4500;} .ov-srcic.yt{background:#ff0000;}
.ov-srcname{font-size:.82rem;color:#e2e8f0;font-weight:600;}
.ov-srccnt{font-size:.74rem;color:#8a93a3;} .ov-srccnt b{color:#e2e8f0;font-size:1.05rem;}
.ov-sechdr{font-size:1.15rem;font-weight:800;color:#f1f5f9;margin:.6rem 0 .9rem;}
.ov-picks{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:.85rem;margin-bottom:1.8rem;}
.ov-pick{background:#12161d;border:1px solid rgba(255,255,255,0.07);border-radius:14px;overflow:hidden;
  transition:transform .12s,border-color .12s;}
.ov-pick:hover{transform:translateY(-3px);border-color:rgba(99,102,241,0.35);}
.ov-pimg{height:118px;background-size:cover;background-position:center;position:relative;}
.ov-pph{display:flex;align-items:center;justify-content:center;}
.ov-pph span{font-size:2.2rem;font-weight:800;color:rgba(255,255,255,0.55);}
.ov-rbadge{position:absolute;top:.5rem;left:.5rem;width:24px;height:24px;border-radius:7px;background:#334155;
  color:#fff;font-size:.8rem;font-weight:800;display:flex;align-items:center;justify-content:center;}
.ov-rbadge.g1{background:#e0a83c;} .ov-rbadge.g2{background:#9aa7b5;} .ov-rbadge.g3{background:#b5794a;}
.ov-pbody{padding:.7rem .8rem .85rem;}
.ov-pname{font-size:.92rem;font-weight:700;color:#f1f5f9;line-height:1.2;margin-bottom:.3rem;}
.ov-prate{display:flex;align-items:center;gap:.35rem;color:#f0b429;font-size:.8rem;}
.ov-pscore{color:#e2e8f0;font-weight:700;}
.ov-pcat{font-size:.68rem;color:#818cf8;margin:.25rem 0 .4rem;}
.ov-pdesc{font-size:.72rem;color:#95a1b2;line-height:1.45;display:-webkit-box;-webkit-line-clamp:3;
  -webkit-box-orient:vertical;overflow:hidden;min-height:3.1em;}
.ov-pment{font-size:.68rem;color:#6b7686;margin-top:.5rem;}
.ov-sub2{font-size:.85rem;font-weight:600;color:#cbd5e1;margin:.2rem 0 .7rem;}
.ov-vgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;}
.ov-vid{display:block;text-decoration:none;}
.ov-vthumb{height:86px;border-radius:9px;background-size:cover;background-position:center;position:relative;
  display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,0.08);}
.ov-play{width:30px;height:30px;border-radius:50%;background:rgba(0,0,0,0.6);color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:.72rem;}
.ov-vtitle{font-size:.74rem;color:#e2e8f0;font-weight:600;margin-top:.4rem;line-height:1.3;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.ov-vch{font-size:.68rem;color:#8a93a3;margin-top:.15rem;}
.ov-rdlist{display:flex;flex-direction:column;gap:.55rem;}
.ov-rd{display:flex;gap:.6rem;text-decoration:none;background:#12161d;border:1px solid rgba(255,255,255,0.07);
  border-radius:11px;padding:.65rem .8rem;transition:border-color .12s;}
.ov-rd:hover{border-color:rgba(255,69,0,0.4);}
.ov-rdic{width:26px;height:26px;border-radius:7px;background:#ff4500;color:#fff;font-weight:700;font-size:.72rem;
  display:flex;align-items:center;justify-content:center;flex:none;}
.ov-rdtitle{font-size:.8rem;color:#e2e8f0;font-weight:600;line-height:1.3;}
.ov-rdmeta{font-size:.7rem;color:#8a93a3;margin-top:.15rem;}
.ov-tk{display:flex;gap:.55rem;align-items:flex-start;margin:.5rem 0;font-size:.8rem;color:#c2ccd8;line-height:1.45;}
.ov-tkchk{color:#22c58a;font-weight:800;flex:none;}
.ov-nx{display:flex;align-items:center;gap:.7rem;padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.ov-nxic{width:32px;height:32px;border-radius:9px;background:rgba(99,102,241,0.16);color:#a5b4fc;
  display:flex;align-items:center;justify-content:center;flex:none;}
.ov-nxt{font-size:.82rem;color:#e2e8f0;font-weight:600;}
.ov-nxs{font-size:.7rem;color:#8a93a3;}
.ov-nxarrow{margin-left:auto;color:#6b7686;}
@media(max-width:900px){.ov-metrics{grid-template-columns:repeat(2,1fr);}.ov-3col{grid-template-columns:1fr;}}
</style>
"""


# Master–detail explorer (Overview tab, RANKING intent): compact selector cards
# on top, one fixed evidence-first detail panel below that swaps content in place.
_EXPLORER_CSS = """
<style>
@keyframes fadeUp{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
.ex-hint{color:#8a93a3;font-size:.82rem;margin:-.35rem 0 .8rem;}
/* whole compact card is the click target: invisible button stretched over the column */
div[data-testid="stColumn"]:has(div[class*="st-key-expick"]){position:relative;}
div[class*="st-key-expick"]{position:absolute;inset:0;z-index:6;margin:0;}
div[class*="st-key-expick"] button{width:100%;height:100%;opacity:0;cursor:pointer;}
.ex-card{display:flex;gap:.6rem;align-items:center;background:#12161d;
  border:1px solid rgba(255,255,255,.08);border-radius:13px;padding:.55rem .65rem;
  transition:border-color .12s,transform .12s;}
.ex-card:hover{border-color:rgba(129,140,248,.45);transform:translateY(-1px);}
.ex-card.sel{border-color:#818cf8;background:#151a24;
  box-shadow:0 0 0 1px #818cf8,0 6px 18px rgba(129,140,248,.18);}
.ex-cthumb{width:44px;height:44px;border-radius:10px;background-size:cover;background-position:center;
  flex:none;position:relative;display:flex;align-items:center;justify-content:center;
  font-weight:800;color:rgba(255,255,255,.6);font-size:1.05rem;}
.ex-crank{position:absolute;top:-6px;left:-6px;width:18px;height:18px;border-radius:6px;background:#334155;
  color:#fff;font-size:.6rem;font-weight:800;display:flex;align-items:center;justify-content:center;
  border:1px solid #0e1117;}
.ex-crank.g1{background:#e0a83c;} .ex-crank.g2{background:#9aa7b5;} .ex-crank.g3{background:#b5794a;}
.ex-cbody{min-width:0;}
.ex-cname{font-size:.83rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;line-height:1.25;}
.ex-cmeta{font-size:.7rem;color:#f0b429;margin-top:.12rem;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.ex-cmeta .mm{color:#7b8698;}
/* ── detail panel ── */
.ex-detail{background:#12161d;border:1px solid rgba(255,255,255,.08);border-radius:16px;
  padding:1.35rem 1.6rem 1.5rem;margin:.9rem 0 1.4rem;animation:fadeUp .25s ease;}
.ex-dhead{display:flex;gap:1.1rem;align-items:center;padding-bottom:1.05rem;
  border-bottom:1px solid rgba(255,255,255,.06);}
.ex-dimg{width:84px;height:84px;border-radius:14px;background-size:cover;background-position:center;flex:none;
  border:1px solid rgba(255,255,255,.1);display:flex;align-items:center;justify-content:center;
  font-size:1.9rem;font-weight:800;color:rgba(255,255,255,.6);}
.ex-dname{font-size:1.45rem;font-weight:800;color:#f5f7fa;line-height:1.08;letter-spacing:-.025em;}
.ex-dsub{font-size:.74rem;color:#818cf8;margin:.22rem 0 .4rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;}
.ex-drate{display:flex;align-items:center;gap:.55rem;font-size:.86rem;color:#f0b429;flex-wrap:wrap;}
.ex-drate b{color:#f1f5f9;font-size:1.08rem;}
.ex-drate .ex-meta{color:#8a93a3;}
.ex-cbadge{font-size:.68rem;font-weight:700;padding:.12rem .5rem;border-radius:20px;}
.ex-why{font-size:1.04rem;font-weight:800;color:#e8edf4;margin:1.1rem 0 .1rem;}
.ex-why b{color:#a5b4fc;}
.ex-lbl{font-size:.71rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#94a3b8;margin-bottom:.55rem;}
.ex-sec{margin-top:1.15rem;}
/* Reddit evidence cards */
.ex-detail a,.ex-detail a:hover,.ex-detail a *{text-decoration:none !important;}
.ex-ev{background:#0f1319;border:1px solid rgba(255,255,255,.06);border-radius:12px;
  padding:.7rem .9rem;margin-bottom:.55rem;}
.ex-evtop{display:flex;align-items:center;gap:.45rem;font-size:.73rem;color:#9aa5b4;flex-wrap:wrap;}
.ex-evsub{color:#ff6b4a;font-weight:700;}
.ex-evby{color:#b9c2cf;font-weight:600;}
.ex-evup{color:#ff8a5f;font-weight:800;}
.ex-evdot{color:#4c5665;}
.ex-evtime{color:#7b8698;}
.ex-open{margin-left:auto;flex:none;font-size:.68rem;font-weight:700;border-radius:16px;
  padding:.18rem .6rem;border:1px solid transparent;white-space:nowrap;}
.ex-open.rd{color:#ff6b4a;background:rgba(255,69,0,.10);border-color:rgba(255,69,0,.30);}
.ex-open.rd:hover{background:rgba(255,69,0,.22);}
.ex-open.yt{color:#ff5252;background:rgba(255,0,0,.10);border-color:rgba(255,0,0,.30);}
.ex-open.yt:hover{background:rgba(255,0,0,.22);}
.ex-evmeta{font-size:.69rem;color:#8a93a3;margin-top:.35rem;}
.ex-evtitle{font-size:.88rem;font-weight:700;color:#eef2f7;line-height:1.35;margin-top:.4rem;}
.ex-ev.pos{border-left:3px solid #22d3a0;} .ex-ev.mix{border-left:3px solid #f59e0b;}
.ex-ev.crit{border-left:3px solid #f87171;}
/* nested top-comments inside a conversation card */
.ex-nlbl{font-size:.64rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;
  color:#7b8698;margin:.75rem 0 .35rem;}
.ex-nest{display:flex;flex-direction:column;gap:.45rem;margin-left:.35rem;
  padding-left:.8rem;border-left:2px solid rgba(255,255,255,.09);}
.ex-ncom{background:rgba(255,255,255,.025);border-radius:9px;padding:.5rem .7rem;
  border-left:2px solid #5b6b7f;}
.ex-ncom.pos{border-left-color:#22d3a0;} .ex-ncom.mix{border-left-color:#f59e0b;}
.ex-ncom.crit{border-left-color:#f87171;}
.ex-nhead{display:flex;align-items:center;gap:.45rem;font-size:.7rem;color:#9aa5b4;flex-wrap:wrap;}
.ex-nopen{margin-left:auto;color:#818cf8;font-weight:800;font-size:.8rem;}
.ex-ncom .ex-clamp,.ex-ncom .ex-noclamp{font-size:.81rem;}
/* video header inside a conversation card: strip its standalone chrome */
.ex-ev .ex-vc{background:none;border:none;padding:0;}
/* expand / collapse — pure CSS <details>, no Streamlit rerun */
details.ex-x{margin-top:.45rem;}
details.ex-x summary{list-style:none;cursor:pointer;}
details.ex-x summary::-webkit-details-marker{display:none;}
.ex-clamp{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;
  font-size:.84rem;color:#d4dce6;line-height:1.55;}
details.ex-x[open] .ex-clamp{-webkit-line-clamp:unset;display:block;}
.ex-xbtn{display:inline-block;margin-top:.3rem;font-size:.7rem;font-weight:700;color:#818cf8;}
.ex-xbtn::after{content:"Read more ▾";}
details.ex-x[open] .ex-xbtn::after{content:"Collapse ▴";}
.ex-noclamp{font-size:.84rem;color:#d4dce6;line-height:1.55;margin-top:.45rem;}
/* YouTube evidence cards */
.ex-vgrid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;align-items:start;}
.ex-vc{display:flex;gap:.8rem;background:#0f1319;border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:.7rem .9rem;}
.ex-vthumb{width:172px;height:97px;border-radius:9px;background-size:cover;background-position:center;
  position:relative;flex:none;display:flex;align-items:center;justify-content:center;
  border:1px solid rgba(255,255,255,.08);}
.ex-vdur{position:absolute;right:.35rem;bottom:.35rem;background:rgba(0,0,0,.82);color:#fff;
  font-size:.64rem;font-weight:700;border-radius:4px;padding:.05rem .3rem;}
.ex-vbody{flex:1;min-width:0;}
.ex-vtop{display:flex;align-items:center;gap:.45rem;font-size:.71rem;color:#9aa5b4;flex-wrap:wrap;}
.ex-vch{color:#b9c2cf;font-weight:600;}
.ex-vviews{color:#7b8698;}
.ex-vtitle{font-size:.86rem;font-weight:700;color:#eef2f7;line-height:1.3;margin-top:.25rem;}
.ex-vcomm{font-size:.69rem;font-weight:700;color:#8fa0b5;margin-top:.45rem;
  text-transform:uppercase;letter-spacing:.06em;}
.ex-basis{font-size:.73rem;color:#818cf8;font-weight:600;margin-bottom:.4rem;}
.ex-consensus p{font-size:.92rem;color:#dbe2ea;line-height:1.6;margin:0;}
/* discussion takeaway — the thread's point at a glance */
.ex-tk{display:flex;gap:.55rem;align-items:baseline;background:rgba(129,140,248,.07);
  border:1px solid rgba(129,140,248,.18);border-radius:8px;padding:.45rem .65rem;
  margin-top:.45rem;font-size:.8rem;color:#c7d2fe;line-height:1.45;}
.ex-tklbl{flex:none;font-size:.62rem;font-weight:800;letter-spacing:.08em;
  text-transform:uppercase;color:#818cf8;}
/* viewpoint chip on nested comments */
.ex-vp{font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;
  padding:.1rem .45rem;border-radius:12px;white-space:nowrap;}
/* consensus agreements / disagreements */
.ex-agdg{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:.7rem;}
.ex-agbox,.ex-dgbox{background:#0f1319;border:1px solid rgba(255,255,255,.06);
  border-radius:10px;padding:.6rem .8rem;}
.ex-aglbl{font-size:.64rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
  color:#8fa0b5;margin-bottom:.35rem;}
.ex-agbox ul,.ex-dgbox ul{margin:0;padding:0;}
.ex-agbox li,.ex-dgbox li{list-style:none;display:flex;gap:.45rem;font-size:.8rem;
  color:#c8d2dd;line-height:1.45;margin:.25rem 0;}
.ex-agbox .ic{color:#22d3a0;font-weight:800;flex:none;}
.ex-dgbox .ic{color:#f59e0b;font-weight:800;flex:none;}
/* "in N discussions" support chip on pros/cons themes */
.ex-pcn{margin-left:.4rem;font-size:.66rem;color:#8a93a3;background:rgba(255,255,255,.05);
  border-radius:10px;padding:.05rem .4rem;white-space:nowrap;}
.ex-grid3{display:grid;grid-template-columns:1.15fr 1fr 1fr;gap:1rem;align-items:start;}
.ex-pcbox{background:#0f1319;border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:.85rem 1rem;}
.ex-pcbox ul{margin:0;padding:0;}
.ex-pcbox li{font-size:.82rem;color:#c8d2dd;line-height:1.5;margin:.32rem 0;list-style:none;display:flex;gap:.5rem;}
.ex-pcbox .ic{flex:none;font-weight:800;}
.ex-pros .ic{color:#22d3a0;} .ex-cons .ic{color:#f87171;}
.ex-reps3{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;align-items:start;}
.ex-rep{background:#0f1319;border:1px solid rgba(255,255,255,.06);border-left:3px solid #5b6b7f;
  border-radius:10px;padding:.7rem .9rem;}
.ex-rep.pos{border-left-color:#22d3a0;} .ex-rep.mix{border-left-color:#f59e0b;} .ex-rep.crit{border-left-color:#f87171;}
.ex-rephead{display:flex;align-items:center;gap:.45rem;font-size:.68rem;margin-bottom:.35rem;flex-wrap:wrap;}
.ex-reptag{font-weight:800;text-transform:uppercase;letter-spacing:.05em;}
.ex-repsrc{color:#8a93a3;} .ex-repup{color:#8a93a3;margin-left:auto;}
.ex-reptxt{font-size:.84rem;color:#c8d2dd;line-height:1.52;}
.ex-empty{color:#64748b;font-size:.8rem;font-style:italic;padding:.3rem 0;}
@media(max-width:900px){.ex-grid3{grid-template-columns:1fr;}.ex-reps3{grid-template-columns:1fr;}
  .ex-vgrid{grid-template-columns:1fr;}.ex-agdg{grid-template-columns:1fr;}}
</style>
"""


# Generic entity page (Overview tab, GENERIC intent): buyer's-guide product page
# — hero + consensus first, pros/cons, aspect scores, one mixed evidence feed;
# analytics demoted to the right rail / an expander.
_GENPAGE_CSS = """
<style>
.gp-hero{display:flex;gap:1.4rem;background:#12161d;border:1px solid rgba(255,255,255,.08);
  border-radius:16px;padding:1.25rem;margin-bottom:1rem;}
.gp-himg{width:280px;min-height:210px;border-radius:12px;background-size:cover;background-position:center;
  flex:none;border:1px solid rgba(255,255,255,.1);align-self:stretch;
  display:flex;align-items:center;justify-content:center;font-size:3rem;font-weight:800;
  color:rgba(255,255,255,.55);}
.gp-hbody{flex:1;min-width:0;}
.gp-hname{font-size:1.85rem;font-weight:800;color:#f5f7fa;letter-spacing:-.03em;line-height:1.1;}
.gp-hchips{display:flex;gap:.45rem;margin:.55rem 0 .9rem;flex-wrap:wrap;}
.gp-chip{font-size:.64rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
  padding:.22rem .6rem;border-radius:6px;background:rgba(255,255,255,.06);color:#aab4c2;
  border:1px solid rgba(255,255,255,.08);}
.gp-chip.acc{background:rgba(240,180,41,.12);color:#f0b429;border-color:rgba(240,180,41,.3);}
.gp-hstats{display:flex;align-items:stretch;gap:1.2rem;margin-bottom:1rem;flex-wrap:wrap;}
.gp-hstat .big{font-size:1.5rem;font-weight:800;color:#f1f5f9;line-height:1.15;}
.gp-hstat .big .of{font-size:.85rem;color:#8a93a3;font-weight:600;}
.gp-hstat .stars{color:#f0b429;font-size:.85rem;letter-spacing:.08em;}
.gp-hstat .cap{font-size:.68rem;color:#8a93a3;margin-top:.2rem;}
.gp-hsep{width:1px;background:rgba(255,255,255,.08);}
.gp-conslbl{font-size:.9rem;font-weight:800;color:#e8edf4;margin-bottom:.35rem;}
.gp-cons{font-size:.88rem;color:#c8d2dd;line-height:1.62;margin:0;}
/* what people love / dislike */
.gp-pc{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:1rem;}
.gp-pcbox{background:#12161d;border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:1rem 1.15rem;}
.gp-pchead{display:flex;align-items:center;gap:.5rem;font-size:.92rem;font-weight:800;color:#e8edf4;
  margin-bottom:.55rem;}
.gp-pcic{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-size:.8rem;flex:none;}
.gp-love .gp-pcic{background:rgba(34,211,160,.14);}
.gp-dis .gp-pcic{background:rgba(248,113,113,.13);}
.gp-pcrow{display:flex;align-items:center;gap:.55rem;padding:.42rem 0;
  border-bottom:1px solid rgba(255,255,255,.05);font-size:.83rem;color:#c8d2dd;line-height:1.4;}
.gp-pcrow:last-child{border-bottom:none;}
.gp-pcrow .ic{flex:none;font-weight:800;}
.gp-love .ic{color:#22d3a0;} .gp-dis .ic{color:#f87171;}
.gp-pcrow .pct{margin-left:auto;flex:none;font-size:.68rem;font-weight:800;border-radius:12px;
  padding:.12rem .5rem;}
.gp-love .pct{color:#22d3a0;background:rgba(34,211,160,.12);}
.gp-dis .pct{color:#f87171;background:rgba(248,113,113,.11);}
/* shared panel */
.gp-panel{background:#12161d;border:1px solid rgba(255,255,255,.07);border-radius:14px;
  padding:1.05rem 1.2rem;margin-bottom:.9rem;}
.gp-phead{display:flex;align-items:baseline;gap:.7rem;margin-bottom:.55rem;}
.gp-ptitle{font-size:1.02rem;font-weight:800;color:#f1f5f9;}
.gp-plegend{margin-left:auto;font-size:.66rem;color:#7b8698;}
/* aspect scores */
.gp-aspgrid{display:grid;grid-template-columns:1fr 1fr;gap:.15rem 2rem;}
.gp-asp{display:flex;align-items:center;gap:.7rem;padding:.42rem 0;}
.gp-asp .lbl{width:112px;flex:none;font-size:.8rem;color:#c2ccd8;font-weight:600;text-transform:capitalize;}
.gp-asp .track{display:block;flex:1;height:7px;background:rgba(255,255,255,.06);border-radius:5px;overflow:hidden;}
.gp-asp .fill{display:block;height:100%;border-radius:5px;}
.gp-asp .v{width:30px;flex:none;text-align:right;font-size:.82rem;font-weight:800;}
/* mixed evidence feed */
.gp-ev{display:flex;gap:.9rem;background:#12161d;border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:.8rem .9rem;margin-bottom:.7rem;text-decoration:none !important;
  transition:border-color .12s,transform .12s;}
.gp-ev:hover{border-color:rgba(129,140,248,.4);transform:translateY(-1px);}
.gp-ev *{text-decoration:none !important;}
.gp-evthumb{width:150px;height:88px;border-radius:9px;background-size:cover;background-position:center;
  flex:none;position:relative;display:flex;align-items:center;justify-content:center;
  border:1px solid rgba(255,255,255,.08);}
.gp-evdur{position:absolute;right:.35rem;bottom:.35rem;background:rgba(0,0,0,.82);color:#fff;
  font-size:.62rem;font-weight:700;border-radius:4px;padding:.05rem .3rem;}
.gp-evn{position:absolute;left:.35rem;bottom:.35rem;background:rgba(0,0,0,.72);color:#e2e8f0;
  font-size:.62rem;font-weight:700;border-radius:4px;padding:.05rem .35rem;}
.gp-evbody{flex:1;min-width:0;}
.gp-evtop{display:flex;align-items:center;gap:.45rem;font-size:.71rem;color:#9aa5b4;flex-wrap:wrap;}
.gp-evsrc{font-weight:700;color:#b9c2cf;}
.gp-evtitle{font-size:.92rem;font-weight:700;color:#eef2f7;line-height:1.35;margin-top:.3rem;}
.gp-evsum{font-size:.78rem;color:#95a1b2;line-height:1.5;margin-top:.3rem;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.gp-evasp{display:flex;align-items:center;gap:.35rem;margin-top:.45rem;flex-wrap:wrap;}
.gp-evasp .k{font-size:.64rem;color:#6b7686;}
.gp-evtag{font-size:.64rem;font-weight:700;color:#a5b4fc;background:rgba(129,140,248,.1);
  border:1px solid rgba(129,140,248,.22);border-radius:5px;padding:.08rem .4rem;}
/* right rail */
.gp-glrow{display:flex;align-items:center;gap:.7rem;padding:.45rem 0;}
.gp-glic{width:30px;height:30px;border-radius:8px;background:rgba(99,102,241,.14);color:#a5b4fc;
  display:flex;align-items:center;justify-content:center;font-size:.8rem;flex:none;}
.gp-gllbl{font-size:.72rem;color:#8a93a3;}
.gp-glval{font-size:.92rem;font-weight:800;color:#f1f5f9;line-height:1.1;}
.gp-tkrow{display:flex;gap:.5rem;align-items:flex-start;margin:.55rem 0;font-size:.78rem;
  color:#c2ccd8;line-height:1.5;}
.gp-tkrow .ck{color:#22d3a0;font-weight:800;flex:none;}
.gp-tkrow b{color:#e8edf4;}
.gp-foot{font-size:.7rem;color:#5f6b7c;text-align:center;margin:1.2rem 0 .4rem;}
/* three-up row: love / dislike / aspect scores */
.gp-3col{display:grid;grid-template-columns:1fr 1fr 1.1fr;gap:.9rem;margin-bottom:1rem;align-items:start;}
.gp-3col .gp-pcbox,.gp-3col .gp-panel{margin:0;}
.gp-3col .gp-panel{padding:1rem 1.15rem;}
.gp-3col .gp-aspgrid{grid-template-columns:1fr;gap:0;}
.gp-3col .gp-asp{padding:.28rem 0;}
.gp-3col .gp-asp .lbl{width:96px;font-size:.75rem;}
/* most helpful reviews — AI-synthesized cards */
.gp-revgrid a,.gp-revgrid a:hover,.gp-revgrid a *,
.gp-qgrid a,.gp-qgrid a:hover,.gp-qgrid a *{text-decoration:none !important;}
.gp-revgrid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1.2rem;align-items:stretch;}
.gp-rev{display:flex;flex-direction:column;background:#12161d;border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:.9rem 1rem;}
.gp-revtop{display:flex;align-items:center;gap:.45rem;font-size:.71rem;color:#9aa5b4;flex-wrap:wrap;}
.gp-revthumb{height:122px;border-radius:9px;background-size:cover;background-position:center;
  position:relative;margin-top:.55rem;border:1px solid rgba(255,255,255,.08);}
.gp-revtitle{font-size:.92rem;font-weight:700;color:#eef2f7;line-height:1.35;margin-top:.5rem;}
.gp-revstars{display:flex;align-items:center;gap:.4rem;font-size:.74rem;color:#f0b429;margin-top:.25rem;}
.gp-revsum{font-size:.8rem;color:#b8c2cf;line-height:1.6;margin-top:.45rem;}
.gp-revtklbl{font-size:.64rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
  color:#7b8698;margin:.6rem 0 .25rem;}
.gp-revtk{margin:0;padding:0;}
.gp-revtk li{list-style:none;display:flex;gap:.45rem;font-size:.76rem;color:#c2ccd8;line-height:1.5;
  margin:.18rem 0;}
.gp-revtk li::before{content:"•";color:#818cf8;font-weight:800;}
.gp-revfoot{display:flex;align-items:center;gap:.35rem;margin-top:auto;padding-top:.6rem;flex-wrap:wrap;}
/* why this source matters — selection rationale + consensus influence */
.gp-whybox{background:rgba(129,140,248,.07);border:1px solid rgba(129,140,248,.18);border-radius:8px;
  padding:.5rem .7rem;margin-top:.55rem;font-size:.75rem;color:#c7d2fe;line-height:1.5;}
.gp-whybox .wl{display:block;font-size:.6rem;font-weight:800;letter-spacing:.08em;
  text-transform:uppercase;color:#818cf8;margin-bottom:.2rem;}
.gp-whybox b{color:#dbe3fe;}
/* per-card agreement / debate lines */
.gp-agdg{margin-top:.5rem;font-size:.75rem;line-height:1.55;}
.gp-agdg .ag{color:#b9e6d3;} .gp-agdg .dg{color:#f3d9a4;}
.gp-agdg b{font-weight:800;}
.gp-agdg .ag b{color:#22d3a0;} .gp-agdg .dg b{color:#f59e0b;}
/* expandable original comments — evidence on demand, no rerun */
details.gp-oc{margin-top:.6rem;}
details.gp-oc summary{list-style:none;cursor:pointer;font-size:.72rem;font-weight:700;color:#818cf8;
  padding:.35rem .65rem;border:1px solid rgba(129,140,248,.25);border-radius:8px;
  background:rgba(129,140,248,.06);display:inline-block;}
details.gp-oc summary::-webkit-details-marker{display:none;}
details.gp-oc summary::after{content:" ▾";}
details.gp-oc[open] summary::after{content:" ▴";}
.gp-ocitem{background:rgba(255,255,255,.025);border-left:2px solid #5b6b7f;border-radius:8px;
  padding:.5rem .7rem;margin-top:.45rem;}
.gp-och{display:flex;gap:.45rem;align-items:center;font-size:.68rem;color:#9aa5b4;flex-wrap:wrap;}
.gp-och .up{color:#ff8a5f;font-weight:800;}
.gp-oct{font-size:.79rem;color:#c8d2dd;line-height:1.55;margin-top:.25rem;}
/* themed community highlights */
.gp-theme{margin-bottom:1rem;}
.gp-themehead{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem;flex-wrap:wrap;}
.gp-themename{font-size:.92rem;font-weight:800;color:#e8edf4;}
.gp-themechip{font-size:.66rem;font-weight:800;padding:.14rem .55rem;border-radius:14px;white-space:nowrap;}
.gp-themechip.pos{color:#22d3a0;background:rgba(34,211,160,.12);}
.gp-themechip.neg{color:#f87171;background:rgba(248,113,113,.11);}
.gp-themechip.mix{color:#f59e0b;background:rgba(245,158,11,.11);}
/* community highlights — the comments that shaped the consensus */
.gp-qgrid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;}
.gp-quote{background:#12161d;border:1px solid rgba(255,255,255,.07);border-radius:13px;padding:.85rem 1rem;}
.gp-qmark{color:#818cf8;font-size:1.3rem;font-weight:800;line-height:1;font-family:Georgia,serif;}
.gp-qtxt{font-size:.83rem;color:#d4dce6;line-height:1.55;margin:.3rem 0 .55rem;font-style:italic;}
.gp-qmeta{display:flex;align-items:center;gap:.45rem;font-size:.68rem;color:#8a93a3;flex-wrap:wrap;}
.gp-interp{background:rgba(129,140,248,.06);border:1px solid rgba(129,140,248,.18);border-radius:12px;
  padding:.8rem 1rem;margin:.8rem 0 1.2rem;}
.gp-interplbl{font-size:.7rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
  color:#818cf8;margin-bottom:.3rem;}
.gp-interp p{font-size:.84rem;color:#c8d2dd;line-height:1.6;margin:0;}
@media(max-width:900px){
  .gp-hero{flex-direction:column;} .gp-himg{width:100%;min-height:180px;}
  .gp-pc{grid-template-columns:1fr;} .gp-aspgrid{grid-template-columns:1fr;}
  .gp-3col{grid-template-columns:1fr;} .gp-revgrid{grid-template-columns:1fr;}
  .gp-qgrid{grid-template-columns:1fr;}
  .gp-evthumb{width:110px;height:70px;}
}
</style>
"""


def _conf_badge(score: float) -> str:
    lbl, color = _confidence_label(score)
    return (f'<span class="ih-cbadge" style="color:{color};background:{color}1a;'
            f'border:1px solid {color}33">{lbl}</span>')


def _aspect_mini(name: str, val: float) -> str:
    color = _score_color(val)
    pct = max(4, min(100, val / 5.0 * 100))
    return (f'<div class="ih-asp-mini"><span class="lbl">{name.replace("_"," ")}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.0f}%;background:{color}"></span></span>'
            f'<span class="v">{val:.1f}</span></div>')


def _entity_card_html(rank: int, item: dict) -> str:
    gcls = f"g{rank}" if rank <= 3 else ""
    sc_cls = _score_cls(item["overall_stars"])
    conf = item.get("confidence_score", item.get("confidence", 0.5))
    asp = sorted((item.get("aspect_scores") or {}).items(), key=lambda x: x[1], reverse=True)[:3]
    asp_html = "".join(_aspect_mini(k, v) for k, v in asp)
    return (
        f'<div class="ih-ent-card">'
        f'<div class="ih-ent-top"><span class="ih-ent-rank {gcls}">#{rank}</span>{_conf_badge(conf)}</div>'
        f'<div class="ih-ent-name">{item["name"]}</div>'
        f'<div style="display:flex;align-items:baseline;gap:.5rem">'
        f'<span class="ih-ent-big {sc_cls}">{item["overall_stars"]:.1f}</span>'
        f'<span style="font-size:.7rem;color:#8a93a3">{item["mentions"]} mention'
        f'{"s" if item["mentions"]!=1 else ""}</span></div>'
        f'<div class="ih-ent-verdict">{_entity_verdict(item)}</div>'
        f'{asp_html}'
        f'</div>'
    )


def _grad_for(seed: str) -> str:
    """Deterministic tasteful gradient for image placeholders (no real photo)."""
    import hashlib
    h = int(hashlib.md5((seed or "x").encode()).hexdigest(), 16)
    hue = h % 360
    return f"linear-gradient(135deg,hsl({hue},48%,26%),hsl({(hue+38)%360},52%,16%))"


def render_results(R):
    import re as _re
    from collections import Counter as _Counter
    from datetime import datetime as _dt

    query = R["query"]; intent = R["intent"]; entity_type = R.get("entity_type")
    aspects = R.get("aspects") or []; payload = R["payload"]
    comments = R["comments"]; annos = R["annos"]
    platform_breakdown = R["platform_breakdown"]; search_time = R.get("search_time", 0.0)
    subreddits = R.get("subreddits") or []; use_cross = R.get("use_cross", False)
    show_debug = R.get("show_debug", False)

    st.markdown(_RESULTS_CSS, unsafe_allow_html=True)
    reviews = [r for rl in platform_breakdown.values() for r in rl]
    anno_map = {a.comment_id: a for a in annos}

    def _rid(r): return r.get("id") if isinstance(r, dict) else getattr(r, "id", "")
    def _txt(r): return ((r.get("text") if isinstance(r, dict) else getattr(r, "text", "")) or "")
    def _get(r, k, d=""):
        return (r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)) or d
    def _score(r):
        a = anno_map.get(_rid(r)); return a.overall_score if a and hasattr(a, "overall_score") else None
    def _is_yt(r):
        return "youtube" in (_get(r, "url") + _get(r, "permalink")).lower() or _get(r, "source") == "youtube"
    def _vid(r):
        # url and permalink are concatenated (no separator), so bound the match to
        # exactly 11 chars — a YouTube ID — or the greedy match runs into the next
        # URL's "https" and yields a corrupt id → 404 thumbnail (gray placeholder).
        s = _get(r, "url") + " " + _get(r, "permalink")
        m = _re.search(r"[?&]v=([\w-]{11})", s) or _re.search(r"youtu\.be/([\w-]{11})", s)
        return m.group(1) if m else ""
    def _subr(r):
        m = _re.search(r"/r/([A-Za-z0-9_]+)", _get(r, "permalink"))
        return m.group(1) if m else ""
    def _ents(r):
        a = anno_map.get(_rid(r))
        if not a:
            return []
        return list(dict.fromkeys(e.name for e in (a.entities or []) if getattr(e, "confidence", 0) >= 0.5 and e.name))

    annotated = set(anno_map.keys())
    relevant = [r for r in reviews if _rid(r) in annotated]
    meaningful = [r for r in relevant if len(_txt(r)) >= 100]
    ev_all = [r for r in relevant if len(_txt(r)) >= 60]
    pos = [r for r in ev_all if (_score(r) or 0) >= 3.8]
    crit = [r for r in ev_all if _score(r) is not None and _score(r) < 2.8]
    mixed = [r for r in ev_all if r not in pos and r not in crit]
    ranking = payload.get("ranking", []) if intent == "RANKING" else []
    also = payload.get("also_mentioned", []) if intent == "RANKING" else []
    plat_str = " + ".join(p.title() for p in platform_breakdown.keys())

    # ── header ──
    ts = _dt.fromtimestamp(payload["metadata"]["timestamp"]).strftime("%b %d, %Y at %H:%M")
    ent_part = f'<span class="ih-meta-dot">·</span><span class="ih-meta-txt">{entity_type}</span>' if entity_type else ""
    st.markdown(
        f'<div class="ih-result-header"><div class="ih-query-title">{query}</div>'
        f'<div class="ih-meta-row"><span class="ih-intent-badge ih-badge-{intent}">{intent}</span>{ent_part}'
        f'<span class="ih-meta-dot">·</span><span class="ih-meta-txt">{plat_str} · {ts}</span></div></div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "Rankings", "Reviews", "Sources", "Aspects", "Developer"])

    def _explorer():
        """Master–detail restaurant explorer: pick a card → see all its evidence."""
        names = [it["name"] for it in ranking]
        sel = st.session_state.get("sel_rest")
        if sel not in names:                    # default / self-heal after a new search
            sel = names[0]
        st.session_state["sel_rest"] = sel

        # ---- per-entity evidence helpers (entity name links a review via its anno) ----
        def _supp(name):
            return [r for r in reviews if name in _ents(r)]

        def _ent_ref(r, name):
            a = anno_map.get(_rid(r))
            for e in (getattr(a, "entities", []) or []) if a else []:
                if e.name == name:
                    return e
            return None

        def _ent_score(r, name):
            e = _ent_ref(r, name)
            s = getattr(e, "sentiment_score", None) if e else None
            if s is None:
                s = _score(r)
            return 3.0 if s is None else s

        def _ent_thumb(name):
            for r in _supp(name):
                if _is_yt(r):
                    v = _vid(r)
                    if v:
                        m = _get(r, "meta", {}) or {}
                        return m.get("thumbnail_url") or f"https://i.ytimg.com/vi/{v}/hqdefault.jpg"
            return None

        # Viewpoint chips for nested comments — fixed vocabulary so styling is stable.
        _VP_COLORS = {"Most Helpful": "#818cf8", "Positive Experience": "#22d3a0",
                      "Alternative View": "#f59e0b", "Critical Opinion": "#f87171",
                      "Practical Tip": "#38bdf8"}

        # AI brief for one entity, grounded in the EXACT evidence shown in the panel
        # (cached per query+entity; GPT with heuristic fallback). Returns:
        #   summary        — consensus that names agreements AND disagreements
        #   agreements / disagreements — short bullets across sources
        #   takeaways      — {thread_id: 1-2 sentence gist} for the shown threads
        #   viewpoints     — {comment_id: viewpoint label} for the shown comments
        #   pros / cons    — [{"text", "n"}] recurring themes, n = distinct discussions
        def _entity_brief(name, item, threads, videos):
            ck = f"__brief__::{query}::{name}"
            if ck in st.session_state:
                return st.session_state[ck]
            aspects = item.get("aspect_scores") or {}
            m = item["mentions"]
            fb_pros = [{"text": k.replace("_", " ").title(), "n": None}
                       for k, v in sorted(aspects.items(), key=lambda x: -x[1]) if v >= 3.6][:3]
            fb_cons = [{"text": k.replace("_", " ").title(), "n": None}
                       for k, v in sorted(aspects.items(), key=lambda x: x[1]) if v < 2.9][:3]
            fb = {
                "summary": f'{_entity_verdict(item)}. Based on {m} mention{"s" if m != 1 else ""} across {plat_str}.',
                "agreements": [], "disagreements": [], "takeaways": {}, "viewpoints": {},
                "pros": fb_pros or [{"text": "Generally well regarded", "n": None}],
                "cons": fb_cons or [{"text": "No consistent complaints surfaced", "n": None}],
            }
            out = fb
            try:
                from insighthub.services.llm import _safe_json_loads
                # Digest of exactly what the panel shows, with stable ids GPT echoes back.
                tid_map, lines = {}, []
                for i, t in enumerate(threads, 1):
                    tid_map[f"t{i}"] = t["tid"]
                    lines.append(f'[t{i}] Reddit thread: "{_excerpt(t["title"], 140)}"')
                    if t["body_raw"]:
                        lines.append(f'  post: {_excerpt(t["body_raw"], 400)}')
                    for u in t["comments"]:
                        lines.append(f'  [c:{_rid(u)}] (▲{_get(u, "upvotes", 0) or 0}) {_excerpt(_txt(u), 300)}')
                for i, v in enumerate(videos, 1):
                    ch = f' by {v["ch"]}' if v["ch"] else ""
                    lines.append(f'[v{i}] YouTube video: "{_excerpt(v["title"], 140)}"{ch}')
                    for u in v["comments"]:
                        lines.append(f'  [c:{_rid(u)}] (👍{_get(u, "upvotes", 0) or 0}) {_excerpt(_txt(u), 300)}')
                vp_opts = " | ".join(f'"{k}"' for k in _VP_COLORS)
                sysmsg = ("You analyze community evidence (Reddit threads, YouTube videos and their "
                          "comments) about ONE specific place. Ground EVERY statement only in the "
                          "provided excerpts — no invented details, no generic advice or recommendations. "
                          "Reply with JSON only.")
                usr = (f'Place: {name}\nAspect scores (1-5): {aspects}\n\nEvidence:\n'
                       + "\n".join(lines)
                       + '\n\nReturn JSON:\n{'
                         '"summary":"3-4 sentences summarizing the evidence above: state concretely what '
                         'commenters AGREE on and where they DISAGREE or debate. Reference the discussions '
                         'by topic, never by id. No generic recommendation.",'
                         '"agreements":["short phrase — a point multiple sources agree on",...max 3],'
                         '"disagreements":["short phrase — a point sources disagree or debate about",'
                         '...max 2, empty list if none],'
                         '"takeaways":{"t1":"1-2 sentence takeaway: the main point of that thread about '
                         'this place, so a reader gets it before reading",...one per t# id},'
                         f'"viewpoints":{{"<id after c:>":one of {vp_opts},...one per comment; use '
                         '"Most Helpful" at most once per thread/video},'
                         '"pros":[{"text":"recurring positive theme","n":<distinct threads/videos '
                         'supporting it>},...max 4, themes recurring across discussions first],'
                         '"cons":[{"text":"recurring negative theme or caveat","n":<same>},...max 4]}')
                resp = llm_service.chat(sysmsg, usr, temperature=0.2, max_tokens=900)
                data = _safe_json_loads(resp) if resp else None

                def _pc_norm(seq):
                    outl = []
                    for e in (seq or []):
                        if isinstance(e, dict) and str(e.get("text", "")).strip():
                            n = e.get("n")
                            outl.append({"text": str(e["text"]).strip(),
                                         "n": int(n) if isinstance(n, (int, float)) else None})
                        elif isinstance(e, str) and e.strip():
                            outl.append({"text": e.strip(), "n": None})
                    return outl[:4]

                if isinstance(data, dict) and data.get("summary"):
                    out = {
                        "summary": str(data.get("summary", "")).strip() or fb["summary"],
                        "agreements": [str(x).strip() for x in (data.get("agreements") or []) if str(x).strip()][:3],
                        "disagreements": [str(x).strip() for x in (data.get("disagreements") or []) if str(x).strip()][:2],
                        "takeaways": {tid_map[k]: str(v).strip()
                                      for k, v in (data.get("takeaways") or {}).items()
                                      if k in tid_map and str(v).strip()},
                        "viewpoints": {str(k): str(v).strip()
                                       for k, v in (data.get("viewpoints") or {}).items()
                                       if str(v).strip() in _VP_COLORS},
                        "pros": _pc_norm(data.get("pros")) or fb["pros"],
                        "cons": _pc_norm(data.get("cons")) or fb["cons"],
                    }
            except Exception as _e:
                logger.warning(f"entity brief GPT failed for {name}: {_e}")
            st.session_state[ck] = out
            return out

        # ═══ MASTER: compact selector cards — selectors, not content (one row) ═══
        cat = entity_type.replace("_", " ").title() if entity_type else ""
        hdr = f"Top Ranked {cat}s" if cat else "Top Ranked Results"
        st.markdown(f'<div class="ov-sechdr">{hdr}</div>', unsafe_allow_html=True)
        st.markdown('<div class="ex-hint">Click a card to explore its evidence — the panel below updates in place.</div>',
                    unsafe_allow_html=True)
        picks = ranking[:6]

        def _select(name):
            st.session_state["sel_rest"] = name

        cols = st.columns(len(picks))
        for col, it in zip(cols, picks):
            rank = ranking.index(it) + 1
            is_sel = it["name"] == sel
            rc = f"g{rank}" if rank <= 3 else ""
            photo = it.get("image_url") or _ent_thumb(it["name"])
            thumb = (f'<div class="ex-cthumb" style="background-image:url({photo})">' if photo
                     else f'<div class="ex-cthumb" style="background:{_grad_for(it["name"])}">{it["name"][:1].upper()}')
            thumb += f'<span class="ex-crank {rc}">{rank}</span></div>'
            col.markdown(
                f'<div class="ex-card{" sel" if is_sel else ""}">{thumb}'
                f'<div class="ex-cbody"><div class="ex-cname" title="{it["name"]}">{it["name"]}</div>'
                f'<div class="ex-cmeta">★ {it["overall_stars"]:.1f} <span class="mm">({it["mentions"]})</span></div>'
                f'</div></div>', unsafe_allow_html=True)
            col.button(it["name"], key=f"expick_{rank}", on_click=_select, args=(it["name"],))

        # ═══ DETAIL: one fixed evidence-first panel — content swaps, position doesn't ═══
        import html as _h
        item = next(it for it in ranking if it["name"] == sel)
        rank_no = ranking.index(item) + 1
        supp = _supp(sel)
        conf = item.get("confidence_score", item.get("confidence", 0.5))
        clbl, ccol = _confidence_label(conf)
        photo = item.get("image_url") or _ent_thumb(sel)
        dimg = (f'<div class="ex-dimg" style="background-image:url({photo})"></div>' if photo
                else f'<div class="ex-dimg" style="background:{_grad_for(sel)}">{sel[:1].upper()}</div>')

        head = (
            f'<div class="ex-dhead">{dimg}<div>'
            f'<div class="ex-dname">{sel}</div>'
            f'<div class="ex-dsub">{cat or "Result"}</div>'
            f'<div class="ex-drate">{_stars(item["overall_stars"])}<b>{item["overall_stars"]:.1f}</b>'
            f'<span class="ex-meta">· {item["mentions"]} mention{"s" if item["mentions"] != 1 else ""}</span>'
            f'<span class="ex-cbadge" style="color:{ccol};background:{ccol}1f">{clbl} · {conf:.0%}</span>'
            f'</div></div></div>'
        )

        def _clamp_block(text, btn_cls="", cap=900, fold=380):
            """2-3 line clamped quote with a pure-CSS Expand/Collapse when long.

            fold ≈ chars that fit in the 3 visible lines; below it the toggle
            would be a no-op, so render plain text instead."""
            t = _h.escape(_excerpt(text, cap))
            if len((text or "").strip()) <= fold:
                return f'<div class="ex-noclamp">{t}</div>'
            return (f'<details class="ex-x"><summary><span class="ex-clamp">{t}</span>'
                    f'<span class="ex-xbtn {btn_cls}"></span></summary></details>')

        # ═══ Community evidence — the hero of the page ═══
        rd_units = [r for r in supp if not _is_yt(r)]
        seen_sig, shown_ids = set(), set()

        def _rd_top(r, up, show_author=True, lead=""):
            sub = _subr(r)
            author = str(_get(r, "author", "") or "").removeprefix("u/")
            when = _rel_time(_get(r, "created_utc", None))
            bits = [b for b in (
                lead,
                f'<span class="ex-evsub">r/{sub}</span>' if sub else "",
                f'<span class="ex-evby">u/{_h.escape(author)}</span>' if (author and show_author) else "",
                f'<span class="ex-evup">▲ {up:,}</span>',
                f'<span class="ex-evtime">{when}</span>' if when else "") if b]
            return (_RD_ICON + '<span class="ex-evdot">·</span>'.join(bits)
                    + f'<a class="ex-open rd" href="https://reddit.com{_get(r, "permalink")}" '
                      f'target="_blank">Open Reddit ↗</a>')

        # Thread posts from the FULL corpus — a discussion's title/author/body are
        # context for the comments below even when the OP never names the entity.
        all_posts = {}
        for r in reviews:
            if not _is_yt(r) and _get(r, "unit_type", "comment") == "post":
                all_posts.setdefault(_get(r, "thread_id") or _rid(r), r)

        # -- 1. SELECT the evidence first (deterministic), so the AI brief can be
        # grounded in exactly what the panel shows. Threads ranked by total unit
        # engagement. Nested comments are picked for VIEWPOINT DIVERSITY — the
        # top-upvoted take plus the best alternative/critical takes when they
        # exist — instead of a flat top-upvotes (usually all-positive) list. --
        def _tone(s):
            return ("Positive", "pos") if s >= 3.8 else ("Mixed", "mix") if s >= 2.8 else ("Critical", "crit")

        def _pick_diverse(cands, k=3):
            cands = sorted(cands, key=lambda u: -(_get(u, "upvotes", 0) or 0))
            picked = cands[:1]
            for lo, hi in ((2.8, 3.8), (-1.0, 2.8)):     # best alternative, then best critical
                if len(picked) >= k:
                    break
                nxt = next((u for u in cands if u not in picked
                            and lo <= _ent_score(u, sel) < hi), None)
                if nxt is not None:
                    picked.append(nxt)
            picked += [u for u in cands if u not in picked][: k - len(picked)]
            return picked

        by_tid = {}
        for r in rd_units:
            by_tid.setdefault(_get(r, "thread_id") or _rid(r), []).append(r)

        sel_threads = []
        for tid, units in sorted(by_tid.items(),
                                 key=lambda kv: -sum((_get(u, "upvotes", 0) or 0) for u in kv[1]))[:3]:
            post = all_posts.get(tid)
            src = post if post is not None else units[0]
            title = (_get(src, "post_title") or "").strip()
            if not title:
                continue
            body_raw = ""
            if post is not None:
                shown_ids.add(_rid(post))
                raw = _txt(post).strip()
                raw = raw[len(title):].strip() if raw.startswith(title) else raw
                if len(raw) >= 60:
                    seen_sig.add(_sig(raw))
                    body_raw = raw
            up = int(_get(post, "upvotes", 0) or 0) if post is not None else \
                max((_get(u, "upvotes", 0) or 0) for u in units)
            cands = [u for u in units if _get(u, "unit_type", "comment") != "post"
                     and len(_txt(u).strip()) >= 40 and _sig(_txt(u).strip()) not in seen_sig]
            comments = _pick_diverse(cands)
            for u in comments:
                seen_sig.add(_sig(_txt(u).strip())); shown_ids.add(_rid(u))
            sel_threads.append({"tid": tid, "src": src, "post": post, "title": title,
                                "body_raw": body_raw, "up": up, "comments": comments})

        vids = {}
        for r in supp:
            if not _is_yt(r):
                continue
            v = _vid(r)
            if not v:
                continue
            m = _get(r, "meta", {}) or {}
            vd = vids.setdefault(v, {"vid": v, "title": m.get("video_title", "Video"),
                                     "ch": m.get("channel_title", ""),
                                     "thumb": m.get("thumbnail_url") or f"https://i.ytimg.com/vi/{v}/hqdefault.jpg",
                                     "views": m.get("view_count", 0), "dur": m.get("duration", ""),
                                     "cmts": []})
            if _get(r, "unit_type", "comment") == "comment" and len(_txt(r).strip()) >= 40:
                vd["cmts"].append(r)
        sel_vids = sorted(vids.values(), key=lambda v: -(v["views"] or 0))[:3]
        for v in sel_vids:
            v["comments"] = _pick_diverse([u for u in v["cmts"] if _sig(_txt(u).strip()) not in seen_sig])
            for u in v["comments"]:
                seen_sig.add(_sig(_txt(u).strip())); shown_ids.add(_rid(u))

        # -- 2. AI brief grounded in that exact selection: consensus w/ agreements
        # + disagreements, per-thread takeaways, viewpoint labels, recurring
        # pros/cons themes. --
        brief = _entity_brief(sel, item, sel_threads, sel_vids)

        def _vp_chip(u, idx):
            vp = brief["viewpoints"].get(_rid(u))
            if not vp:                                   # heuristic fallback when GPT is unavailable
                s = _ent_score(u, sel)
                vp = ("Most Helpful" if idx == 0 else
                      "Critical Opinion" if s < 2.8 else
                      "Alternative View" if s < 3.8 else "Positive Experience")
            c = _VP_COLORS.get(vp, "#818cf8")
            return f'<span class="ex-vp" style="color:{c};background:{c}1a;border:1px solid {c}33">{vp}</span>'

        # -- 3. Reddit conversation cards: takeaway first (the thread's point at a
        # glance), then the post and its viewpoint-labeled comments. --
        def _ncom_rd(r, idx):
            body = _txt(r).strip()
            _, cls = _tone(_ent_score(r, sel))
            author = str(_get(r, "author", "") or "").removeprefix("u/")
            when = _rel_time(_get(r, "created_utc", None))
            up = _get(r, "upvotes", 0) or 0
            bits = [b for b in (
                _vp_chip(r, idx),
                f'<span class="ex-evby">u/{_h.escape(author)}</span>' if author else "",
                f'<span class="ex-evup">▲ {up:,}</span>',
                f'<span class="ex-evtime">{when}</span>' if when else "") if b]
            head = ('<span class="ex-evdot">·</span>'.join(bits)
                    + f'<a class="ex-nopen" href="https://reddit.com{_get(r, "permalink")}" '
                      f'target="_blank">↗</a>')
            return f'<div class="ex-ncom {cls}"><div class="ex-nhead">{head}</div>{_clamp_block(body)}</div>'

        disc_cards = []
        for t in sel_threads:
            tk = brief["takeaways"].get(t["tid"], "")
            tk_html = (f'<div class="ex-tk"><span class="ex-tklbl">Takeaway</span>'
                       f'<span>{_h.escape(tk)}</span></div>') if tk else ""
            body = _clamp_block(t["body_raw"]) if t["body_raw"] else ""
            nested = "".join(_ncom_rd(u, i) for i, u in enumerate(t["comments"]))
            nested_html = (f'<div class="ex-nlbl">Community viewpoints</div>'
                           f'<div class="ex-nest">{nested}</div>') if nested else ""
            disc_cards.append(
                f'<div class="ex-ev"><div class="ex-evtop">'
                f'{_rd_top(t["src"], t["up"], show_author=t["post"] is not None)}</div>'
                f'<div class="ex-evtitle">{_h.escape(_excerpt(t["title"], 120))}</div>'
                f'{tk_html}{body}{nested_html}</div>')
        disc_html = (f'<div class="ex-sec"><div class="ex-lbl">{_RD_ICON}Reddit Discussions</div>'
                     f'{"".join(disc_cards)}</div>') if disc_cards else ""

        # -- 4. YouTube conversation cards: video header + viewpoint-labeled
        # viewer comments nested inside. Videos ranked by view count. --
        def _ncom_yt(r, idx):
            body = _txt(r).strip()
            _, cls = _tone(_ent_score(r, sel))
            likes = _get(r, "upvotes", 0) or 0
            author = str(_get(r, "author", "") or "")
            when = _rel_time(_get(r, "created_utc", None))
            bits = [b for b in (
                _vp_chip(r, idx),
                f'<span class="ex-evby">{_h.escape(author)}</span>' if author else "",
                f'<span class="ex-evup">👍 {likes:,}</span>' if likes > 0 else "",
                f'<span class="ex-evtime">{when}</span>' if when else "") if b]
            link = _get(r, "url") or _get(r, "permalink")
            head = ('<span class="ex-evdot">·</span>'.join(bits)
                    + (f'<a class="ex-nopen" href="{link}" target="_blank">↗</a>' if link else ""))
            return f'<div class="ex-ncom {cls}"><div class="ex-nhead">{head}</div>{_clamp_block(body)}</div>'

        vcards = ""
        for v in sel_vids:
            watch = f"https://youtube.com/watch?v={v['vid']}"
            views = _fmt_views(v["views"])
            dur = f'<span class="ex-vdur">{v["dur"]}</span>' if v["dur"] else ""
            bits = [b for b in (
                f'<span class="ex-vch">{_h.escape(v["ch"])}</span>' if v["ch"] else "",
                f'<span class="ex-vviews">{views}</span>' if views else "") if b]
            top = (_YT_ICON + '<span class="ex-evdot">·</span>'.join(bits)
                   + f'<a class="ex-open yt" href="{watch}" target="_blank">Watch on YouTube ↗</a>')
            nested = "".join(_ncom_yt(u, i) for i, u in enumerate(v["comments"]))
            nested_html = (f'<div class="ex-nlbl">Community viewpoints</div>'
                           f'<div class="ex-nest">{nested}</div>') if nested else ""
            vcards += (
                f'<div class="ex-ev"><div class="ex-vc"><a class="ex-vthumb" href="{watch}" target="_blank" '
                f'style="background-image:url({v["thumb"]})">{dur}<span class="ov-play">▶</span></a>'
                f'<div class="ex-vbody"><div class="ex-vtop">{top}</div>'
                f'<div class="ex-vtitle">{_h.escape(v["title"])}</div></div></div>'
                f'{nested_html}</div>')
        vid_html = (f'<div class="ex-sec"><div class="ex-lbl">{_YT_ICON}YouTube Videos</div>'
                    f'{vcards}</div>') if vcards else ""

        # -- 5. AI Consensus — rendered AFTER all evidence; summarizes it explicitly
        # (agreements AND disagreements), over a full evidence-base line. --
        n_threads = len(by_tid)
        n_rdc = sum(1 for r in rd_units if _get(r, "unit_type", "comment") != "post")
        n_ytc = sum(1 for r in supp if _is_yt(r) and _get(r, "unit_type", "comment") == "comment")
        basis_bits = []
        if n_threads:
            basis_bits.append(f'{n_threads} Reddit discussion{"s" if n_threads != 1 else ""}'
                              + (f' with {n_rdc} community comment{"s" if n_rdc != 1 else ""}' if n_rdc else ""))
        if vids:
            basis_bits.append(f'{len(vids)} YouTube video{"s" if len(vids) != 1 else ""}'
                              + (f' with {n_ytc} viewer comment{"s" if n_ytc != 1 else ""}' if n_ytc else ""))
        basis = ("Synthesized from " + " and ".join(basis_bits)) if basis_bits else \
            f'Synthesized from {item["mentions"]} mention{"s" if item["mentions"] != 1 else ""}'
        ag = "".join(f'<li><span class="ic">✓</span><span>{_h.escape(a)}</span></li>'
                     for a in brief["agreements"])
        dg = "".join(f'<li><span class="ic">≠</span><span>{_h.escape(d)}</span></li>'
                     for d in brief["disagreements"])
        agdg = ""
        if ag or dg:
            agdg = ('<div class="ex-agdg">'
                    + (f'<div class="ex-agbox"><div class="ex-aglbl">Where sources agree</div><ul>{ag}</ul></div>' if ag else "")
                    + (f'<div class="ex-dgbox"><div class="ex-aglbl">Where they disagree</div><ul>{dg}</ul></div>' if dg else "")
                    + '</div>')
        cons_html = (f'<div class="ex-sec ex-consensus"><div class="ex-lbl">AI Consensus</div>'
                     f'<div class="ex-basis">{basis}</div><p>{brief["summary"]}</p>{agdg}</div>')

        # -- 6. aspects + pros/cons (recurring themes; "in N discussions" chips
        # when a theme spans multiple sources), three compact columns --
        aspects_d = item.get("aspect_scores") or {}
        bars = "".join(_aspect_bar(k, v) for k, v in sorted(aspects_d.items(), key=lambda x: -x[1]))

        def _pc_li(e, ic):
            n = e.get("n")
            chip = (f'<span class="ex-pcn">in {n} discussions</span>'
                    if isinstance(n, int) and n >= 2 else "")
            return f'<li><span class="ic">{ic}</span><span>{_h.escape(e["text"])}{chip}</span></li>'

        pros = "".join(_pc_li(p, "＋") for p in brief["pros"])
        cons = "".join(_pc_li(c, "－") for c in brief["cons"])
        grid_html = (
            f'<div class="ex-sec"><div class="ex-grid3">'
            f'<div><div class="ex-lbl">Aspect Breakdown</div>{bars or "<div class=ex-empty>No aspect data.</div>"}</div>'
            f'<div class="ex-pcbox ex-pros"><div class="ex-lbl">Pros</div><ul>{pros}</ul></div>'
            f'<div class="ex-pcbox ex-cons"><div class="ex-lbl">Cons</div><ul>{cons}</ul></div>'
            f'</div></div>')

        # Evidence is the hero: Reddit conversations → YouTube conversations
        # → AI consensus (summarizes the evidence) → aspects/pros-cons.
        st.markdown(
            f'<div class="ex-detail">{head}'
            f'<div class="ex-why">Why is <b>{sel}</b> ranked #{rank_no}?</div>'
            f'{disc_html}{vid_html}{cons_html}{grid_html}</div>',
            unsafe_allow_html=True)

    # ═══ OVERVIEW — consumer-product landing ═══
    with tabs[0]:
        st.markdown(_OVERVIEW_CSS, unsafe_allow_html=True)

        st.markdown(_EXPLORER_CSS, unsafe_allow_html=True)

        if intent == "RANKING" and ranking:
            _explorer()
        else:
            # ═══ GENERIC — buyer's-guide product page, not a dashboard. ═══
            # Section order answers the reader's questions in sequence:
            # what is this? → is it good? → why love / dislike it? → which
            # aspects? → show me the evidence. Analytics are metadata and live
            # in the right rail + an expander, never above the fold.
            import html as _h
            st.markdown(_GENPAGE_CSS, unsafe_allow_html=True)

            overall = float(payload.get("overall", 0) or 0)
            aspect_avgs = dict(payload.get("aspects") or {})

            # -- 1. SELECT evidence first (deterministic): ONE mixed feed of
            # discussions — Reddit threads and YouTube videos — ranked by
            # usefulness (relevant units carried, then engagement), so the AI
            # brief is grounded in exactly what the page shows. --
            all_posts = {}
            for r in reviews:
                if not _is_yt(r) and _get(r, "unit_type", "comment") == "post":
                    all_posts.setdefault(_get(r, "thread_id") or _rid(r), r)

            by_tid, by_vid = {}, {}
            for r in relevant:
                if _is_yt(r):
                    v = _vid(r)
                    if v:
                        by_vid.setdefault(v, []).append(r)
                else:
                    by_tid.setdefault(_get(r, "thread_id") or _rid(r), []).append(r)
            n_disc = len(by_tid) + len(by_vid)

            def _quotes(units, k=2):
                us = sorted(units, key=lambda u: -(_get(u, "upvotes", 0) or 0))
                return [u for u in us if len(_txt(u).strip()) >= 60][:k]

            feed = []
            for tid, units in by_tid.items():
                src = all_posts.get(tid) or units[0]
                title = (_get(src, "post_title") or "").strip()
                if not title:
                    continue
                up = max((_get(u, "upvotes", 0) or 0) for u in units)
                feed.append({
                    "kind": "rd", "id": f"rd_{tid}", "title": title, "units": units,
                    "sub": _subr(src) or _subr(units[0]), "up": up, "eng": up,
                    "when": _get(src, "created_utc", None) or _get(units[0], "created_utc", None),
                    "link": f'https://reddit.com{_get(src, "permalink")}',
                    "quotes": _quotes(units)})
            for v, units in by_vid.items():
                m = next(((_get(u, "meta", {}) or {}) for u in units
                          if (_get(u, "meta", {}) or {}).get("video_title")),
                         _get(units[0], "meta", {}) or {})
                cmts = [u for u in units if _get(u, "unit_type", "comment") == "comment"]
                feed.append({
                    "kind": "yt", "id": f"yt_{v}", "units": units,
                    "title": m.get("video_title") or "YouTube video",
                    "ch": m.get("channel_title", ""), "views": m.get("view_count", 0),
                    "dur": m.get("duration", ""),
                    "thumb": m.get("thumbnail_url") or f"https://i.ytimg.com/vi/{v}/hqdefault.jpg",
                    "eng": sum((_get(u, "upvotes", 0) or 0) for u in cmts),
                    "when": _get(units[0], "created_utc", None),
                    "link": f"https://youtube.com/watch?v={v}",
                    "quotes": _quotes(cmts or units)})
            # usefulness, not source: discussions carrying the most relevant
            # units first; ties broken by upvotes / comment likes (views are a
            # different scale, so they don't enter the sort)
            feed.sort(key=lambda c: (-len(c["units"]), -c["eng"]))
            feed = feed[:8]

            # -- 2. ONE cached GPT brief grounded in that selection: display
            # name + category, the "Internet Consensus" paragraph, pros/cons
            # with support share, buyer takeaways, per-discussion gists and
            # related comparisons. All domain judgment stays in GPT; the
            # fallbacks below only reshape data we already have. --
            def _generic_brief():
                ck = f"__gpage__::{query}"
                if ck in st.session_state:
                    return st.session_state[ck]
                summ_plain = _re.sub(r"[#*_]+", " ", payload.get("summary", "") or "")
                summ_plain = _re.sub(r"\s+", " ", summ_plain).strip()
                sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", summ_plain) if len(s.strip()) > 20]
                fb = {
                    "name": query.strip().title() if len(query.split()) <= 6 else query.strip(),
                    "category": (entity_type or "").replace("_", " ").title() or "Community Analysis",
                    "consensus": " ".join(sents[:3]) or f"Community analysis of {query}.",
                    "pros": [{"text": k.replace("_", " ").title(), "pct": None}
                             for k, v in sorted(aspect_avgs.items(), key=lambda x: -x[1]) if v >= 3.6][:5],
                    "cons": [{"text": k.replace("_", " ").title(), "pct": None}
                             for k, v in sorted(aspect_avgs.items(), key=lambda x: x[1]) if v < 3.0][:5],
                    "takeaways": {}, "cards": {}, "comparisons": [],
                }
                out = fb
                try:
                    from insighthub.services.llm import _safe_json_loads
                    id_map, lines = {}, []
                    for i, c in enumerate(feed, 1):
                        id_map[f"d{i}"] = c["id"]
                        if c["kind"] == "rd":
                            lines.append(f'[d{i}] Reddit thread ({len(c["units"])} relevant comments, '
                                         f'▲{c["up"]}): "{_excerpt(c["title"], 140)}"')
                        else:
                            lines.append(f'[d{i}] YouTube video by {c["ch"] or "?"}: '
                                         f'"{_excerpt(c["title"], 140)}"')
                        for u in c["quotes"]:
                            lines.append(f"  quote: {_excerpt(_txt(u), 260)}")
                    sysmsg = ("You analyze community evidence (Reddit threads, YouTube videos and their "
                              "comments) about the subject of a user's query. Ground every statement only "
                              "in the provided excerpts — no invented details. Reply with JSON only.")
                    usr = (f"Query: {query}\nSubject type: {entity_type or 'unknown'}\n"
                           f"Community rating: {overall:.1f}/5\nAspect scores (1-5): {aspect_avgs}\n\n"
                           "Evidence:\n" + "\n".join(lines) + "\n\nReturn JSON:\n{"
                           '"name":"canonical display name of the single subject the query asks about",'
                           '"category":"2-4 word category label for the subject",'
                           '"consensus":"2-3 sentence internet consensus: what it is widely regarded as '
                           'and why (its main strengths), then the recurring concerns. Concrete, grounded, '
                           'no hedging boilerplate.",'
                           '"pros":[{"text":"specific recurring positive","pct":<0-100 share of discussions '
                           'supporting it, or null when unclear>},...max 5],'
                           '"cons":[{"text":"specific recurring concern","pct":<same>},...max 5],'
                           '"takeaways":{"best_for":"one sentence — who it suits best",'
                           '"watch_outs":"one sentence — what to check before choosing it",'
                           '"bottom_line":"one sentence verdict"},'
                           '"cards":{"d1":{"summary":"1-2 sentence gist of that discussion about the '
                           'subject","aspects":["short aspect it covers",...max 3]},...one per d# id},'
                           '"comparisons":["<subject name> vs <named alternative commenters actually '
                           'compare it to>",...max 4, [] if none]}')
                    resp = llm_service.chat(sysmsg, usr, temperature=0.25, max_tokens=1200)
                    data = _safe_json_loads(resp) if resp else None
                    if isinstance(data, dict) and data.get("consensus"):
                        def _pcn(seq):
                            o = []
                            for e in (seq or []):
                                if isinstance(e, dict) and str(e.get("text", "")).strip():
                                    p = e.get("pct")
                                    o.append({"text": str(e["text"]).strip(),
                                              "pct": int(p) if isinstance(p, (int, float)) and 0 < p <= 100 else None})
                                elif isinstance(e, str) and e.strip():
                                    o.append({"text": e.strip(), "pct": None})
                            return o[:5]
                        tk = data.get("takeaways") if isinstance(data.get("takeaways"), dict) else {}
                        out = {
                            "name": str(data.get("name") or "").strip() or fb["name"],
                            "category": str(data.get("category") or "").strip() or fb["category"],
                            "consensus": str(data["consensus"]).strip(),
                            "pros": _pcn(data.get("pros")) or fb["pros"],
                            "cons": _pcn(data.get("cons")) or fb["cons"],
                            "takeaways": {k: str(tk.get(k)).strip() for k in
                                          ("best_for", "watch_outs", "bottom_line")
                                          if str(tk.get(k) or "").strip()},
                            "cards": {id_map[k]: {"summary": str(v.get("summary", "")).strip(),
                                                  "aspects": [str(a).strip() for a in
                                                              (v.get("aspects") or [])[:3] if str(a).strip()]}
                                      for k, v in (data.get("cards") or {}).items()
                                      if k in id_map and isinstance(v, dict)},
                            "comparisons": [str(x).strip() for x in (data.get("comparisons") or [])
                                            if str(x).strip()][:4],
                        }
                except Exception as _e:
                    logger.warning(f"generic page brief GPT failed: {_e}")
                st.session_state[ck] = out
                return out

            brief = _generic_brief()
            gname = brief["name"]

            # -- 3. hero image via the same enrichment chain the ranking page
            # uses (GPT category picks places vs generic; disk-cached) --
            img_ck = f"__gimg__::{query}"
            if img_ck not in st.session_state:
                _hero_url = None
                try:
                    from insighthub.services.image_enrichment import (
                        get_image_service, PLACE_PROVIDERS, GENERIC_PROVIDERS)
                    from insighthub.services.llm import classify_query as _cq
                    _isp = _cq(query) in ("local_discovery", "service_review")
                    _hero_url = get_image_service().get_image_url(
                        gname, query if _isp else (entity_type or "").replace("_", " "),
                        providers=PLACE_PROVIDERS if _isp else GENERIC_PROVIDERS)
                except Exception as _e:
                    logger.warning(f"hero image enrichment skipped: {_e}")
                st.session_state[img_ck] = _hero_url
            hero_img = st.session_state[img_ck] or next(
                (c["thumb"] for c in feed if c["kind"] == "yt"), None)

            main_col, rail = st.columns([2.55, 1.05], gap="medium")

            with main_col:
                # ── hero: what is this + is it good, before anything else ──
                himg = (f'<div class="gp-himg" style="background-image:url({hero_img})"></div>' if hero_img
                        else f'<div class="gp-himg" style="background:{_grad_for(gname)}">'
                             f'{_h.escape(gname[:1].upper())}</div>')
                slbl, scol, _sbg = _sentiment_label(overall)
                rating = (
                    f'<div class="gp-hstat"><div class="big">{overall:.1f}<span class="of">/5</span></div>'
                    f'<div class="stars">{_stars(overall)}</div><div class="cap">Community Rating</div></div>'
                    f'<div class="gp-hsep"></div>'
                    f'<div class="gp-hstat"><div class="big" style="color:{scol}">{slbl}</div>'
                    f'<div class="cap">Overall Sentiment</div></div>'
                    f'<div class="gp-hsep"></div>') if overall else ""
                etype_lbl = (entity_type or "").replace("_", " ")
                etype_chip = (f'<span class="gp-chip">{_h.escape(etype_lbl)}</span>'
                              if etype_lbl and etype_lbl.title() != brief["category"].title() else "")
                st.markdown(
                    f'<div class="gp-hero">{himg}<div class="gp-hbody">'
                    f'<div class="gp-hname">{_h.escape(gname)}</div>'
                    f'<div class="gp-hchips"><span class="gp-chip acc">{_h.escape(brief["category"])}</span>'
                    f'{etype_chip}</div>'
                    f'<div class="gp-hstats">{rating}'
                    f'<div class="gp-hstat"><div class="big">{n_disc}</div>'
                    f'<div class="cap">Discussions Analyzed</div></div></div>'
                    f'<div class="gp-conslbl">Internet Consensus</div>'
                    f'<p class="gp-cons">{_h.escape(brief["consensus"])}</p>'
                    f'</div></div>', unsafe_allow_html=True)

                # ── why people like / dislike it — the highest-value section ──
                def _pcrow(e, ic):
                    pct = (f'<span class="pct">{e["pct"]}%</span>'
                           if isinstance(e.get("pct"), int) else "")
                    return (f'<div class="gp-pcrow"><span class="ic">{ic}</span>'
                            f'<span>{_h.escape(e["text"])}</span>{pct}</div>')
                love = ("".join(_pcrow(p, "✓") for p in brief["pros"])
                        or '<div class="ov-empty">No consistent praise surfaced.</div>')
                dis = ("".join(_pcrow(c, "✕") for c in brief["cons"])
                       or '<div class="ov-empty">No consistent complaints surfaced.</div>')
                love_box = (f'<div class="gp-pcbox gp-love"><div class="gp-pchead">'
                            f'<span class="gp-pcic">👍</span>What People Love</div>{love}</div>')
                dis_box = (f'<div class="gp-pcbox gp-dis"><div class="gp-pchead">'
                           f'<span class="gp-pcic">👎</span>What People Dislike</div>{dis}</div>')

                # aspect scores share the row: where it excels, numeric not donut
                asp_box = ""
                if aspect_avgs:
                    asp_rows = "".join(
                        f'<div class="gp-asp"><span class="lbl">{k.replace("_", " ")}</span>'
                        f'<span class="track"><span class="fill" style="width:{max(4, v / 5 * 100):.0f}%;'
                        f'background:{_score_color(v)}"></span></span>'
                        f'<span class="v" style="color:{_score_color(v)}">{v:.1f}</span></div>'
                        for k, v in sorted(aspect_avgs.items(), key=lambda x: -x[1]))
                    asp_box = (f'<div class="gp-panel"><div class="gp-pchead">Aspect Scores</div>'
                               f'<div class="gp-plegend" style="margin:-.3rem 0 .35rem">'
                               f'1 (poor) → 5 (excellent)</div>'
                               f'<div class="gp-aspgrid">{asp_rows}</div></div>')
                st.markdown(
                    (f'<div class="gp-3col">{love_box}{dis_box}{asp_box}</div>' if asp_box
                     else f'<div class="gp-pc">{love_box}{dis_box}</div>'),
                    unsafe_allow_html=True)

                # -- second cached GPT pass: REVIEW SYNTHESIS. Turns the top
                # discussions into buyer-focused summaries (merging same-theme
                # discussions into one card) and picks the individual comments
                # that shaped the consensus. This is the "read 50 discussions
                # in 5 minutes" content the summaries/aspects can't carry. --
                unit_by_id = {_rid(r): r for r in relevant}

                # Real per-aspect coverage across ALL discussions (not just the
                # top-8 feed): every "mentioned in N discussions" count on this
                # page is derived from these statistics — GPT is only allowed to
                # pick from them, never to invent a number.
                disc_map = dict(by_tid)
                for _v, _us in by_vid.items():
                    disc_map[f"v_{_v}"] = _us
                _ad = {}
                for _did, _us in disc_map.items():
                    for u in _us:
                        a = anno_map.get(_rid(u))
                        for k, s in (getattr(a, "aspect_scores", None) or {}).items():
                            _ad.setdefault(k, {}).setdefault(_did, []).append(s)
                asp_stats = []          # (aspect, n_discussions, avg_score, disc_ids)
                for k, dd in _ad.items():
                    means = [sum(v) / len(v) for v in dd.values()]
                    asp_stats.append((k, len(dd), sum(means) / len(means), set(dd)))
                asp_stats.sort(key=lambda x: (-x[1], x[0]))

                def _sent_word(avg):
                    return ("positive", "pos") if avg >= 3.5 else \
                           ("negative", "neg") if avg < 3.0 else ("mixed", "mix")

                def _review_brief():
                    ck = f"__greviews3__::{query}"
                    if ck in st.session_state:
                        return st.session_state[ck]
                    fb_revs = []
                    for c in feed[:4]:
                        body = max((_txt(u) for u in c["units"]), key=len, default="")
                        fb_revs.append({"id": c["id"], "also": [], "takeaways": [], "aspects": [],
                                        "summary": _excerpt(body, 300),
                                        "why": "", "influence": "", "represents": None,
                                        "agreements": [], "disagreements": []})
                    # fallback themes come straight from the aspect statistics:
                    # real counts, real sentiment, top-upvoted on-aspect quotes
                    fb_themes, _fb_used = [], set()
                    for k, n, avg, dids in asp_stats[:3]:
                        cands = sorted((u for _d in dids for u in disc_map[_d]
                                        if _get(u, "unit_type", "comment") == "comment"
                                        and 60 <= len(_txt(u).strip()) <= 400
                                        and _rid(u) not in _fb_used
                                        and k in (getattr(anno_map.get(_rid(u)), "aspect_scores", None) or {})),
                                       key=lambda u: -(_get(u, "upvotes", 0) or 0))
                        if not cands:
                            continue
                        _fb_used.update(_rid(u) for u in cands[:2])
                        fb_themes.append({"name": k.replace("_", " ").title(),
                                          "sentiment": _sent_word(avg)[0], "count": n,
                                          "quotes": [_rid(u) for u in cands[:2]]})
                    fb = {"reviews": [r for r in fb_revs if r["summary"]],
                          "themes": fb_themes, "interpretation": ""}
                    out = fb
                    try:
                        from insighthub.services.llm import _safe_json_loads
                        id_map, lines = {}, []
                        for i, c in enumerate(feed, 1):
                            id_map[f"d{i}"] = c["id"]
                            if c["kind"] == "rd":
                                lines.append(f'[d{i}] Reddit thread in r/{c["sub"]} (▲{c["up"]}): '
                                             f'"{_excerpt(c["title"], 140)}"')
                            else:
                                lines.append(f'[d{i}] YouTube video by {c["ch"] or "?"}: '
                                             f'"{_excerpt(c["title"], 140)}"')
                            body_u = max(c["units"], key=lambda u: len(_txt(u)))
                            if len(_txt(body_u)) >= 200 and _get(body_u, "unit_type", "comment") != "comment":
                                lines.append(f"  content: {_excerpt(_txt(body_u), 500)}")
                            for u in c["units"]:
                                if _get(u, "unit_type", "comment") == "comment" and len(_txt(u).strip()) >= 60:
                                    lines.append(f'  [c:{_rid(u)}] (▲{_get(u, "upvotes", 0) or 0}) '
                                                 f'{_excerpt(_txt(u), 240)}')
                        stats_line = " | ".join(f"{k}: {n} discussions, avg {avg:.1f}/5"
                                                for k, n, avg, _ds in asp_stats[:8]) or "none"
                        sysmsg = ("You synthesize community reviews (Reddit threads, YouTube videos and "
                                  "their comments) into a buyer's guide about one subject. Ground every "
                                  "statement only in the provided excerpts. Every NUMBER you output must "
                                  "come from the provided statistics or from counting the listed "
                                  "discussions — never invent counts. Reply with JSON only.")
                        usr = (f"Subject: {gname}\nQuery: {query}\n"
                               f"Total discussions analyzed: {n_disc}\n"
                               f"Aspect coverage across ALL discussions: {stats_line}\n\n"
                               "Discussions:\n" + "\n".join(lines)
                               + '\n\nReturn JSON:\n{'
                                 '"reviews":[{"id":"d# this summary is anchored to (the strongest '
                                 'discussion of its theme)",'
                                 '"also":["d# of other discussions repeating the same theme",... [] if none],'
                                 '"summary":"3-5 sentence buyer-focused synthesis of what the reviewer(s) '
                                 'actually concluded: their experience, what they praise, what they '
                                 'complain about. When several discussions share a theme, merge them into '
                                 'this one summary instead of separate cards.",'
                                 '"takeaways":["short key takeaway",...3-4],'
                                 '"aspects":["aspect discussed",...max 3],'
                                 '"agreements":["short point most commenters in this discussion agree '
                                 'on",...max 2],'
                                 '"disagreements":["short point commenters debate or push back on",'
                                 '...max 2, [] if none],'
                                 '"why":"1 sentence: why this source earned its spot — the unique '
                                 'perspective it adds that the other discussions lack",'
                                 '"influence":"short sentence: the consensus point this source most '
                                 "strongly supports or challenges, e.g. 'One of the strongest sources "
                                 "behind the charging-experience consensus.'\","
                                 f'"represents":<int 1-{n_disc}: how many analyzed discussions echo this '
                                 'source\'s main theme, derived from the aspect coverage stats or by '
                                 'counting the listed discussions; null if unclear>},'
                                 '...max 4 cards, most helpful for a buyer first, each anchored to a '
                                 'different discussion],'
                                 '"themes":[{"name":"2-4 word theme the community keeps returning to",'
                                 '"sentiment":"positive"|"negative"|"mixed" (from the avg score of the '
                                 'matching aspect: >=3.5 positive, <3.0 negative, else mixed),'
                                 f'"count":<int 1-{n_disc}: distinct discussions mentioning it — take it '
                                 'from the aspect coverage stats>,'
                                 '"quotes":["<id after c:> of the most representative comment for this '
                                 'theme",...1-2]},'
                                 '...max 4 themes ordered by count desc, each quote id used only once],'
                                 '"interpretation":"1-2 sentences: how these themes combine into the '
                                 'overall consensus — where owners agree and what trade-off they accept"}')
                        resp = llm_service.chat(sysmsg, usr, temperature=0.25, max_tokens=1600)
                        data = _safe_json_loads(resp) if resp else None
                        if isinstance(data, dict) and data.get("reviews"):
                            def _clamp_n(v):
                                try:
                                    return max(1, min(int(v), n_disc))
                                except (TypeError, ValueError):
                                    return None
                            revs, seen_anchor = [], set()
                            for rv in data["reviews"]:
                                if not isinstance(rv, dict):
                                    continue
                                did = id_map.get(str(rv.get("id", "")))
                                summ = str(rv.get("summary") or "").strip()
                                if not did or not summ or did in seen_anchor:
                                    continue
                                seen_anchor.add(did)
                                revs.append({
                                    "id": did,
                                    "also": [id_map[a] for a in map(str, rv.get("also") or [])
                                             if a in id_map and id_map[a] != did],
                                    "summary": summ,
                                    "takeaways": [str(t).strip() for t in (rv.get("takeaways") or [])
                                                  if str(t).strip()][:4],
                                    "aspects": [str(a).strip() for a in (rv.get("aspects") or [])
                                                if str(a).strip()][:3],
                                    "agreements": [str(x).strip() for x in (rv.get("agreements") or [])
                                                   if str(x).strip()][:2],
                                    "disagreements": [str(x).strip() for x in (rv.get("disagreements") or [])
                                                      if str(x).strip()][:2],
                                    "why": str(rv.get("why") or "").strip(),
                                    "influence": str(rv.get("influence") or "").strip(),
                                    "represents": _clamp_n(rv.get("represents")),
                                })
                            themes, _q_used = [], set()
                            for th in (data.get("themes") or []):
                                if not isinstance(th, dict) or not str(th.get("name") or "").strip():
                                    continue
                                qs = [str(q) for q in (th.get("quotes") or [])
                                      if str(q) in unit_by_id and str(q) not in _q_used][:2]
                                if not qs:
                                    continue
                                _q_used.update(qs)
                                themes.append({
                                    "name": str(th["name"]).strip(),
                                    "sentiment": (str(th.get("sentiment") or "").strip().lower()
                                                  if str(th.get("sentiment") or "").strip().lower()
                                                  in ("positive", "negative", "mixed") else "mixed"),
                                    "count": _clamp_n(th.get("count")),
                                    "quotes": qs,
                                })
                            out = {
                                "reviews": revs[:4] or fb["reviews"],
                                "themes": themes[:4] or fb["themes"],
                                "interpretation": str(data.get("interpretation") or "").strip(),
                            }
                    except Exception as _e:
                        logger.warning(f"review synthesis GPT failed: {_e}")
                    st.session_state[ck] = out
                    return out

                revb = _review_brief()
                feed_by_id = {c["id"]: c for c in feed}

                # ── most helpful reviews: the AI-summarized buyer's guide core ──
                rcards = []
                for rv in revb["reviews"][:4]:
                    c = feed_by_id.get(rv["id"])
                    if not c:
                        continue
                    scs = [s for s in (_score(u) for u in c["units"]) if s is not None]
                    avg = sum(scs) / len(scs) if scs else None
                    stars = (f'<div class="gp-revstars">{_stars(avg)}'
                             f'<b style="color:#e2e8f0">{avg:.1f}</b></div>') if avg else ""
                    if c["kind"] == "yt":
                        top = [_YT_ICON + '<span class="gp-evsrc">YouTube</span>',
                               f'<span>{_h.escape(c["ch"])}</span>' if c["ch"] else "",
                               _fmt_views(c["views"])]
                        thumb = (f'<div class="gp-revthumb" style="background-image:url({c["thumb"]})">'
                                 + (f'<span class="gp-evdur">{c["dur"]}</span>' if c["dur"] else "")
                                 + '</div>')
                    else:
                        top = [_RD_ICON + '<span class="gp-evsrc">Reddit</span>',
                               f'<span>r/{_h.escape(c["sub"])}</span>' if c["sub"] else "",
                               f'▲ {c["up"]:,}']
                        thumb = ""
                    top_html = '<span class="ex-evdot">·</span>'.join(t for t in top if t)
                    tks = "".join(f'<li>{_h.escape(t)}</li>' for t in rv["takeaways"])
                    tk_html = (f'<div class="gp-revtklbl">Key takeaways</div>'
                               f'<ul class="gp-revtk">{tks}</ul>') if tks else ""
                    chips = "".join(f'<span class="gp-evtag">{_h.escape(a)}</span>' for a in rv["aspects"])
                    # why this source: selection rationale, coverage, consensus influence
                    rep = rv.get("represents") or (len(rv["also"]) + 1 if rv["also"] else None)
                    why_bits = []
                    if rep and rep > 1:
                        why_bits.append(f'<b>Represents {rep} similar discussions.</b>')
                    for t in (rv.get("why"), rv.get("influence")):
                        t = (t or "").strip()
                        if t:
                            t = t[0].upper() + t[1:]
                            why_bits.append(_h.escape(t if t.endswith((".", "!", "?")) else t + "."))
                    whybox = (f'<div class="gp-whybox"><span class="wl">Why this source</span>'
                              f'{" ".join(why_bits)}</div>') if why_bits else ""
                    # where the discussion agrees vs. debates — one line each
                    agdg = ""
                    if rv.get("agreements") or rv.get("disagreements"):
                        ag = "; ".join(_h.escape(x.rstrip(".")) for x in rv["agreements"])
                        dg = "; ".join(_h.escape(x.rstrip(".")) for x in rv["disagreements"])
                        agdg = ('<div class="gp-agdg">'
                                + (f'<div class="ag"><b>✓ Agree:</b> {ag}</div>' if ag else "")
                                + (f'<div class="dg"><b>⚡ Debated:</b> {dg}</div>' if dg else "")
                                + '</div>')
                    # original comments: supporting evidence on demand, not
                    # primary content — collapsed pure-CSS <details>, no rerun
                    oc_units = sorted((u for u in c["units"]
                                       if _get(u, "unit_type", "comment") == "comment"
                                       and len(_txt(u).strip()) >= 60),
                                      key=lambda u: -(_get(u, "upvotes", 0) or 0))[:4]
                    oc_html = ""
                    if oc_units:
                        items = []
                        for u in oc_units:
                            up = _get(u, "upvotes", 0) or 0
                            author = str(_get(u, "author", "") or "").removeprefix("u/")
                            who = (f'{_h.escape(author)}' if c["kind"] == "yt"
                                   else f'u/{_h.escape(author)}') if author else ""
                            mark = "👍" if c["kind"] == "yt" else "▲"
                            head = '<span class="ex-evdot">·</span>'.join(
                                x for x in (who, f'<span class="up">{mark} {up:,}</span>' if up else "") if x)
                            items.append(f'<div class="gp-ocitem"><div class="gp-och">{head}</div>'
                                         f'<div class="gp-oct">{_h.escape(_excerpt(_txt(u), 320))}</div></div>')
                        oc_html = (f'<details class="gp-oc"><summary>Original comments '
                                   f'({len(oc_units)})</summary>{"".join(items)}</details>')
                    foot = (f'{chips}<a class="ex-open {"yt" if c["kind"] == "yt" else "rd"}" '
                            f'style="margin-left:auto" href="{c["link"]}" target="_blank">Open original ↗</a>')
                    rcards.append(
                        f'<div class="gp-rev"><div class="gp-revtop">{top_html}</div>{thumb}'
                        f'<div class="gp-revtitle">{_h.escape(_excerpt(c["title"], 90))}</div>{stars}'
                        f'<div class="gp-revsum">{_h.escape(rv["summary"])}</div>{whybox}{tk_html}{agdg}'
                        f'{oc_html}<div class="gp-revfoot">{foot}</div></div>')
                if rcards:
                    st.markdown('<div class="gp-phead" style="margin:.4rem 0 .55rem">'
                                '<span class="gp-ptitle">Most Helpful Reviews</span>'
                                '<span class="gp-plegend">AI summaries — themes merged across '
                                'discussions</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="gp-revgrid">{"".join(rcards)}</div>',
                                unsafe_allow_html=True)

                # ── community highlights, organized by THEME: each block names a
                # recurring topic, states how many discussions raised it (real
                # counts from the aspect statistics), then quotes the voices
                # that best represent it — consensus explanation, not a feed ──
                def _quote_card(u):
                    up = _get(u, "upvotes", 0) or 0
                    if _is_yt(u):
                        m = _get(u, "meta", {}) or {}
                        src = _YT_ICON + f'<span>{_h.escape(m.get("channel_title") or "YouTube")}</span>'
                        eng = f'👍 {up:,}' if up else ""
                        link = _get(u, "url")
                    else:
                        sub = _subr(u)
                        src = _RD_ICON + (f'<span>r/{_h.escape(sub)}</span>' if sub else "<span>Reddit</span>")
                        eng = f'▲ {up:,} upvotes' if up else ""
                        link = f'https://reddit.com{_get(u, "permalink")}'
                    meta = '<span class="ex-evdot">·</span>'.join(x for x in (src, eng) if x)
                    return (f'<div class="gp-quote"><div class="gp-qmark">“</div>'
                            f'<div class="gp-qtxt">{_h.escape(_excerpt(_txt(u), 200))}</div>'
                            f'<div class="gp-qmeta">{meta}<a class="ex-nopen" style="margin-left:auto" '
                            f'href="{link}" target="_blank">↗</a></div></div>')

                _chip_words = {"positive": ("Mentioned positively in", "pos"),
                               "negative": ("Mentioned negatively in", "neg"),
                               "mixed": ("Mixed views across", "mix")}
                theme_blocks = []
                for th in (revb.get("themes") or [])[:4]:
                    qcards = [_quote_card(unit_by_id[q]) for q in th["quotes"] if q in unit_by_id]
                    if not qcards:
                        continue
                    chip = ""
                    if th.get("count"):
                        lead, cls = _chip_words.get(th["sentiment"], ("Mentioned in", "mix"))
                        chip = (f'<span class="gp-themechip {cls}">{lead} {th["count"]} '
                                f'discussion{"s" if th["count"] != 1 else ""}</span>')
                    theme_blocks.append(
                        f'<div class="gp-theme"><div class="gp-themehead">'
                        f'<span class="gp-themename">{_h.escape(th["name"])}</span>{chip}</div>'
                        f'<div class="gp-qgrid">{"".join(qcards)}</div></div>')
                if theme_blocks:
                    st.markdown('<div class="gp-phead" style="margin:.2rem 0 .55rem">'
                                '<span class="gp-ptitle">Top Community Highlights</span>'
                                '<span class="gp-plegend">the voices behind the consensus, '
                                'organized by theme</span></div>', unsafe_allow_html=True)
                    st.markdown("".join(theme_blocks), unsafe_allow_html=True)
                    if revb["interpretation"]:
                        st.markdown(f'<div class="gp-interp"><div class="gp-interplbl">AI Interpretation'
                                    f'</div><p>{_h.escape(revb["interpretation"])}</p></div>',
                                    unsafe_allow_html=True)

                # ── all evidence: the unfiltered mixed feed, for diving deeper ──
                st.markdown('<div class="gp-phead" style="margin:.4rem 0 .1rem">'
                            '<span class="gp-ptitle">All Evidence</span></div>',
                            unsafe_allow_html=True)
                srcf = st.radio("Source", ["All", "Reddit", "YouTube"], key="gp_src",
                                horizontal=True, label_visibility="collapsed")
                shown = [c for c in feed if srcf == "All"
                         or (srcf == "Reddit") == (c["kind"] == "rd")]
                ev_cards = []
                for c in shown[:6]:
                    b = brief["cards"].get(c["id"], {})
                    summ = b.get("summary") or (_excerpt(_txt(c["quotes"][0]), 150) if c["quotes"] else "")
                    asp = b.get("aspects") or []
                    if not asp:   # fallback: aspects the discussion's units actually scored
                        cnt = _Counter(k for u in c["units"]
                                       for k in (getattr(anno_map.get(_rid(u)), "aspect_scores", None) or {}))
                        asp = [k.replace("_", " ") for k, _n in cnt.most_common(3)]
                    tags = "".join(f'<span class="gp-evtag">{_h.escape(a)}</span>' for a in asp[:3])
                    tags = (f'<div class="gp-evasp"><span class="k">Key aspects:</span>{tags}</div>'
                            if tags else "")
                    when = _rel_time(c.get("when"))
                    if c["kind"] == "yt":
                        thumb = (f'<div class="gp-evthumb" style="background-image:url({c["thumb"]})">'
                                 + (f'<span class="gp-evdur">{c["dur"]}</span>' if c["dur"] else "")
                                 + '</div>')
                        meta = [f'<span class="gp-evsrc">{_h.escape(c["ch"])}</span>' if c["ch"] else "",
                                _fmt_views(c["views"]), when]
                        icon = _YT_ICON
                    else:
                        thumb = (f'<div class="gp-evthumb" style="background:{_grad_for(c["title"])}">'
                                 f'<span class="gp-evn">💬 {len(c["units"])}</span></div>')
                        meta = [f'<span class="gp-evsrc">r/{_h.escape(c["sub"])}</span>' if c["sub"] else "",
                                f'▲ {c["up"]:,} upvotes', when]
                        icon = _RD_ICON
                    meta_html = '<span class="ex-evdot">·</span>'.join(x for x in meta if x)
                    ev_cards.append(
                        f'<a class="gp-ev" href="{c["link"]}" target="_blank">{thumb}'
                        f'<div class="gp-evbody"><div class="gp-evtop">{icon}{meta_html}</div>'
                        f'<div class="gp-evtitle">{_h.escape(_excerpt(c["title"], 110))}</div>'
                        + (f'<div class="gp-evsum">{_h.escape(summ)}</div>' if summ else "")
                        + f'{tags}</div></a>')
                # parent <div> first: an <a> opening the markdown block would be
                # parsed inside a <p>, and the browser then shreds the first card
                # re-wrapping each child <div> in its own anchor
                st.markdown(f'<div class="gp-feed">{"".join(ev_cards)}</div>' if ev_cards
                            else '<div class="ov-empty">No evidence to show.</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="gp-foot">All evidence with filters lives in the Reviews tab '
                            '· AI analysis of community discussions from Reddit and YouTube — not '
                            'affiliated with the subject.</div>', unsafe_allow_html=True)

            with rail:
                # ── At a Glance: the old metric cards, demoted to metadata ──
                gl_rows = "".join(
                    f'<div class="gp-glrow"><span class="gp-glic">{ic}</span>'
                    f'<div><div class="gp-glval">{val}</div><div class="gp-gllbl">{lbl}</div></div></div>'
                    for ic, val, lbl in (
                        ("◈", n_disc, "Discussions Analyzed"),
                        ("❝", len(meaningful), "Detailed Reviews"),
                        ("⬡", len(platform_breakdown), f"Sources · {plat_str}"),
                        ("◷", f"{search_time:.1f}s", "Analysis Time"),
                        ("↻", ts.split(" at ")[0], "Last Updated")))
                st.markdown(f'<div class="gp-panel"><div class="gp-ptitle" style="margin-bottom:.4rem">'
                            f'At a Glance</div>{gl_rows}</div>', unsafe_allow_html=True)

                src_rows = "".join(
                    f'<div class="ov-srcrow"><span class="ov-srcic {cls}">{icch}</span>'
                    f'<div><div class="ov-srcname">{nme}</div>'
                    f'<div class="ov-srccnt"><b>{cnt}</b> discussions ({cnt / max(1, n_disc):.0%})</div>'
                    f'</div></div>'
                    for nme, cnt, cls, icch in (("Reddit", len(by_tid), "rd", "r/"),
                                                ("YouTube", len(by_vid), "yt", "▶")) if cnt)
                if src_rows:
                    st.markdown(f'<div class="gp-panel"><div class="gp-ptitle" style="margin-bottom:.7rem">'
                                f'Source Breakdown</div>{src_rows}</div>', unsafe_allow_html=True)

                tk_rows = "".join(
                    f'<div class="gp-tkrow"><span class="ck">✓</span>'
                    f'<span><b>{lbl}:</b> {_h.escape(txt)}</span></div>'
                    for lbl, txt in (("Best For", brief["takeaways"].get("best_for")),
                                     ("Watch Outs", brief["takeaways"].get("watch_outs")),
                                     ("Bottom Line", brief["takeaways"].get("bottom_line"))) if txt)
                if overall:
                    tk_rows = (f'<div class="gp-tkrow"><span class="ck">✓</span>'
                               f'<span><b>Overall Rating:</b> {overall:.1f}/5</span></div>') + tk_rows
                if tk_rows:
                    st.markdown(f'<div class="gp-panel"><div class="gp-ptitle" style="margin-bottom:.4rem">'
                                f'Takeaways</div>{tk_rows}</div>', unsafe_allow_html=True)

                # ── related comparisons: the natural next step, replaces CTAs ──
                if brief["comparisons"]:
                    st.markdown('<div class="gp-ptitle" style="margin:.2rem 0 .5rem">Related Comparisons</div>',
                                unsafe_allow_html=True)
                    for _ci, _cq_txt in enumerate(brief["comparisons"]):
                        if st.button(_cq_txt, key=f"gp_cmp_{_ci}", use_container_width=True):
                            st.session_state["pending_query"] = _cq_txt
                            st.session_state["run_analysis"] = True
                            st.rerun()

                with st.expander("Analysis Details"):
                    _scraped = sum(len(v) for v in platform_breakdown.values())
                    st.markdown(
                        f"- **Units scraped:** {_scraped}\n"
                        f"- **Relevant units analyzed:** {len(comments)}\n"
                        f"- **Detailed reviews:** {len(meaningful)}\n"
                        f"- **Platforms:** {plat_str}\n"
                        f"- **Analysis time:** {search_time:.1f}s\n"
                        f"- **Completed:** {ts}"
                        + (f"\n- **Subreddits:** {', '.join('r/' + s for s in subreddits)}"
                           if subreddits else ""))

    # ═══ RANKINGS ═══
    with tabs[1]:
        if not ranking and not also:
            st.info("No ranked entities. Try a broader query or increase depth.")
        if ranking:
            n = st.session_state.get("rank_n", 6)
            for i, item in enumerate(ranking[:n], 1):
                conf = item.get("confidence_score", item.get("confidence", 0.5))
                st.markdown(_entity_card_html(i, item), unsafe_allow_html=True)
                with st.expander("Details · aspects · evidence"):
                    if item.get("aspect_scores"):
                        st.markdown("".join(_aspect_bar(k, v) for k, v in
                                    sorted(item["aspect_scores"].items(), key=lambda x: -x[1])),
                                    unsafe_allow_html=True)
                    st.caption(f"{conf:.0%} ranking confidence · {item['mentions']} mentions · upvote-weighted")
                    for q in (item.get("quotes") or [])[:3]:
                        st.markdown(f'<div class="ih-evcard"><div class="ih-clamp3">{q}</div></div>',
                                    unsafe_allow_html=True)
            if len(ranking) > n:
                if st.button(f"Load more ({len(ranking)-n} left)", key="rank_more"):
                    st.session_state["rank_n"] = n + 6
                    st.rerun()
        if also:
            with st.expander(f"Also mentioned · lower confidence ({len(also)})"):
                for item in also:
                    st.markdown(f'**{item["name"]}** — {item["overall_stars"]:.1f} · {item["mentions"]} mention'
                                f'{"s" if item["mentions"]!=1 else ""} · too few reviews to rank confidently')

    # ═══ EVIDENCE ═══
    with tabs[2]:
        f1, f2, f3, f4 = st.columns(4)
        sent_f = f1.radio("Sentiment", ["All", "Positive", "Mixed", "Critical"], key="ev_sent", horizontal=False)
        src_opts = [p.title() for p in platform_breakdown.keys()]
        src_f = f2.multiselect("Source", src_opts, default=src_opts, key="ev_src")
        ent_names = ["All"] + [it["name"] for it in ranking]
        ent_f = f3.selectbox("Entity", ent_names, key="ev_entity")
        sort_f = f4.selectbox("Sort", ["Relevance", "Upvotes", "Recent"], key="ev_sort")

        pool = {"All": ev_all, "Positive": pos, "Mixed": mixed, "Critical": crit}[sent_f]
        def _srcname(r): return "YouTube" if _is_yt(r) else "Reddit"
        # case-insensitive: the options come from platform keys via .title()
        # ("Youtube") while _srcname says "YouTube" — a direct `in` check
        # silently excluded every YouTube review from this tab.
        _src_sel = {s.lower() for s in src_f}
        pool = [r for r in pool if _srcname(r).lower() in _src_sel]
        if ent_f != "All":
            pool = [r for r in pool if ent_f in _ents(r)]
        if sort_f == "Upvotes":
            pool = sorted(pool, key=lambda r: -(_get(r, "upvotes", 0) or 0))
        elif sort_f == "Recent":
            pool = sorted(pool, key=lambda r: -(_get(r, "created_utc", 0) or 0))

        st.caption(f"{len(pool)} matching reviews")
        n = st.session_state.get("ev_n", 8)
        for r in pool[:n]:
            sc = _score(r)
            sent_badge = ""
            if sc is not None:
                sl, scol, sbg = _sentiment_label(sc)
                sent_badge = f'<span class="ih-cbadge" style="color:{scol};background:{sbg}">{sl}</span>'
            up = _get(r, "upvotes", 0) or 0
            ent_chips = "".join(f'<span class="ih-chip">{e}</span>' for e in _ents(r)[:5])
            snippet = _excerpt(_txt(r), 260)
            if _is_yt(r):
                vid = _vid(r); meta = _get(r, "meta", {}) or {}
                title = meta.get("video_title", "") or "YouTube video"
                chan = meta.get("channel_title", "")
                thumb_src = meta.get("thumbnail_url") or (f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else "")
                thumb = f'<img class="ih-yt-thumb" src="{thumb_src}">' if thumb_src else ""
                link = _get(r, "url") or (f"https://youtube.com/watch?v={vid}" if vid else "#")
                st.markdown(
                    f'<div class="ih-evcard"><div class="ih-yt">{thumb}<div class="ih-yt-body">'
                    f'<div class="ih-evcard-head"><span class="ih-badge-src ih-badge-yt">YouTube</span>{sent_badge}'
                    f'<span class="ih-evcard-sub">👍 {up:,}</span></div>'
                    f'<div class="ih-evcard-auth">{title}</div>'
                    f'<div class="ih-evcard-sub">{chan}</div>'
                    f'<div class="ih-clamp3">{snippet}</div>{ent_chips}'
                    f'<div style="margin-top:.3rem"><a class="ih-open-link" href="{link}" target="_blank">Open Video ↗</a></div>'
                    f'</div></div></div>', unsafe_allow_html=True)
            else:
                sub = _subr(r); perma = _get(r, "permalink")
                link = f"https://reddit.com{perma}" if perma else _get(r, "url", "#")
                sub_txt = f'<span class="ih-evcard-sub">r/{sub}</span>' if sub else ""
                st.markdown(
                    f'<div class="ih-evcard"><div class="ih-evcard-head">'
                    f'<span class="ih-badge-src ih-badge-rd">Reddit</span>{sent_badge}'
                    f'<span class="ih-evcard-auth">{_get(r,"author","anon")}</span>{sub_txt}'
                    f'<span class="ih-evcard-sub">▲ {up:,}</span></div>'
                    f'<div class="ih-clamp3">{snippet}</div>{ent_chips}'
                    f'<div style="margin-top:.3rem"><a class="ih-open-link" href="{link}" target="_blank">Open Reddit ↗</a></div>'
                    f'</div>', unsafe_allow_html=True)
        if len(pool) > n:
            if st.button(f"Load more ({len(pool)-n} left)", key="ev_more"):
                st.session_state["ev_n"] = n + 8
                st.rerun()

    # ═══ SOURCES ═══
    with tabs[3]:
        by = {}
        for c in comments:
            by.setdefault((c.get("source", "reddit") or "reddit").title(), _Counter())[c.get("unit_type", "comment")] += 1
        scols = st.columns(max(1, len(by)))
        for col, (src, cnt) in zip(scols, by.items()):
            rows = "".join(f'<div class="row"><span>{k}s analyzed</span><b>{v}</b></div>' for k, v in cnt.items())
            col.markdown(f'<div class="ih-srccard"><h4>{src}</h4>{rows}</div>', unsafe_allow_html=True)
        scraped = sum(len(v) for v in platform_breakdown.values())
        st.markdown('<div class="ih-ov-hdr">Pipeline funnel</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ih-srccard">'
            f'<div class="row"><span>Scraped (raw units)</span><b>{scraped}</b></div>'
            f'<div class="row"><span>After relevance filter</span><b>{len(comments)}</b></div>'
            f'<div class="row"><span>Final analyzed</span><b>{len(relevant)}</b></div>'
            f'</div>', unsafe_allow_html=True)
        if subreddits:
            st.caption("Subreddits: " + ", ".join(f"r/{s}" for s in subreddits[:8]))

    # ═══ ASPECTS ═══
    with tabs[4]:
        agg2 = {}
        for it in ranking:
            for k, v in (it.get("aspect_scores") or {}).items():
                agg2.setdefault(k, []).append((v, it.get("mentions", 1)))
        rows = [(k, sum(a * b for a, b in vs) / (sum(b for _, b in vs) or 1)) for k, vs in agg2.items()]
        if not rows and payload.get("aspects"):
            rows = list(payload["aspects"].items())
        rows.sort(key=lambda x: -x[1])
        if rows:
            st.markdown("**How each dimension is discussed** (mention-weighted average across ranked entities)")
            st.markdown("".join(_aspect_bar(k, v) for k, v in rows), unsafe_allow_html=True)
            if ranking:
                st.markdown("**Per-entity aspect scores**")
                import pandas as pd
                asp_keys = [k for k, _ in rows]
                df = pd.DataFrame([
                    {"Entity": it["name"], **{k.replace("_", " ").title(): round((it.get("aspect_scores") or {}).get(k, float("nan")), 1)
                                              for k in asp_keys}}
                    for it in ranking[:12]
                ]).set_index("Entity")
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No aspect data available for this query.")

    # ═══ DEVELOPER ═══
    with tabs[5]:
        import pandas as pd
        if intent == "RANKING":
            try:
                from insighthub.core.diagnostics import entity_flow_table
                rows = entity_flow_table(reviews, comments, annos)
                conf_names = {r["name"] for r in ranking}
                also_names = {r["name"] for r in also}
                for row in rows:
                    row["outcome"] = ("ranked" if row["entity"] in conf_names
                                      else "also-mentioned" if row["entity"] in also_names else "dropped")
                if rows:
                    st.markdown("**Entity flow** — retrieved → filtered → extracted → ranked")
                    st.dataframe(pd.DataFrame(rows)[["entity", "retrieved", "filtered", "extracted", "primary", "outcome"]],
                                 use_container_width=True)
            except Exception as _e:
                st.caption(f"Entity flow unavailable: {_e}")
        with st.expander("Ranking JSON"):
            st.json(payload.get("ranking", []))
        with st.expander("Raw data — first 3 reviews"):
            try:
                st.dataframe(pd.DataFrame(reviews[:3])[["id", "author", "upvotes", "permalink", "text"]])
            except Exception:
                st.write(reviews[:3])
        if not use_cross:
            with st.expander("Reddit search plan"):
                try:
                    plan = reddit_service._plan_search(query)
                    st.json({"terms": plan.terms, "subreddits": plan.subreddits,
                             "time_filter": plan.time_filter, "strategies": plan.strategies})
                except Exception as _e:
                    st.caption(f"Plan unavailable: {_e}")


# ── Analysis ──────────────────────────────────────────────────────────────────

query = st.session_state.get("query", "")

if run_analysis and query.strip():
    try:
        # ── Progress timeline ─────────────────────────────────────────────
        use_cross = len(selected_platforms) > 1 or (
            len(selected_platforms) == 1 and selected_platforms[0] != Platform.REDDIT
        )

        _prog = st.empty()
        _done: list = []

        def _update(active: str = ""):
            rows = [
                f'<div class="ih-step ih-step-done">'
                f'<span class="ih-step-icon">✓</span>'
                f'<span class="ih-step-text">{t}</span></div>'
                for t in _done
            ]
            if active:
                rows.append(
                    f'<div class="ih-step ih-step-active">'
                    f'<span class="ih-step-spin"></span>'
                    f'<span class="ih-step-text">{active}</span></div>'
                )
            _prog.markdown(
                '<div class="ih-progress">' + "".join(rows) + '</div>',
                unsafe_allow_html=True,
            )

        # Step 1 — intent detection
        _update("Detecting query type…")
        intent_schema = llm_service.detect_intent_and_schema(query)
        intent = (
            QueryIntent(intent_schema.intent)
            if intent_schema.intent in ["RANKING", "SOLUTION", "GENERIC"]
            else QueryIntent.GENERIC
        )
        _done.append(f"Query type: {intent_schema.intent} · {len(intent_schema.aspects)} dimensions")

        def _normalize(review):
            _g = (lambda k, d=None: review.get(k, d)) if isinstance(review, dict) else (lambda k, d=None: getattr(review, k, d))
            return {
                "id":         _g("id"),
                "text":       _g("text"),
                "upvotes":    _g("upvotes", 0) or 0,
                "permalink":  _g("permalink", ""),
                "post_title": _g("post_title", ""),
                "source":     _g("source", "reddit"),
                "unit_type":  _g("unit_type", "comment"),
                "thread_id":  _g("thread_id", ""),
            }

        if use_cross:
            ps = " + ".join(p.value.title() for p in selected_platforms)
            _update(f"Scanning {ps} for discussions…")
            start_time = time.time()

            def _scrape_then_annotate(platform):
                service = cross_platform_manager.platforms[platform]
                ck = ("scrape_platform_v1", platform.value, query.lower().strip(), limit)
                raw = cross_platform_manager._scrape_cache.get(ck)
                if raw is None:
                    raw = service.scrape(query, limit)
                    cross_platform_manager._scrape_cache.set(ck, raw, expire=3600)
                pc = [_normalize(r) for r in raw]
                pc = llm_service.filter_relevant_comments(pc, query, intent=intent_schema.intent)
                pa = llm_service.annotate_comments_with_gpt(pc, intent_schema.aspects, intent_schema.entity_type, query)
                return platform.value, raw, pc, pa

            platform_breakdown, comments, annos = {}, [], []
            with ThreadPoolExecutor(max_workers=len(selected_platforms)) as ex:
                futs = {ex.submit(_scrape_then_annotate, p): p for p in selected_platforms}
                for f in as_completed(futs):
                    pn, raw, pc, pa = f.result()
                    platform_breakdown[pn] = raw
                    comments.extend(pc)
                    annos.extend(pa)
                    _done.append(f"{pn.title()}: {len(pc)} relevant discussions")
                    _update(f"Processing remaining sources…")

            search_time = time.time() - start_time
            reviews = [r for rlist in platform_breakdown.values() for r in rlist]
            cross_platform_manager.aggregate_results(platform_breakdown, query, intent)

        else:
            # Step 2 — scrape
            _update("Scanning Reddit for discussions…")
            start_time = time.time()
            reviews = reddit_service.scrape(query, limit, subreddit_count)
            search_time = time.time() - start_time
            platform_breakdown = {"reddit": reviews}
            try:
                _plan = reddit_service._last_plan
                if _plan and _plan.subreddits:
                    _sub_preview = ", ".join(f"r/{s}" for s in _plan.subreddits[:3])
                    if len(_plan.subreddits) > 3:
                        _sub_preview += f" +{len(_plan.subreddits) - 3} more"
                    _done.append(f"Search strategy: {_sub_preview}")
            except Exception:
                pass
            _done.append(f"Found {len(reviews)} discussions")
            # Step 3 — filter
            _update("Filtering for relevance…")
            comments = [_normalize(r) for r in reviews]
            comments = llm_service.filter_relevant_comments(comments, query, intent=intent_schema.intent)
            _done.append(f"{len(comments)} relevant discussions")
            # Step 4 — annotate
            _update(f"Extracting opinions and scoring sentiment…")
            annos = llm_service.annotate_comments_with_gpt(
                comments, intent_schema.aspects, intent_schema.entity_type, query
            )

        # Capture subreddits for source disclosure row
        try:
            _subreddits_used = reddit_service._last_plan.subreddits if getattr(reddit_service, "_last_plan", None) else []
        except Exception:
            _subreddits_used = []

        upvote_map = {c["id"]: c["upvotes"] for c in comments}
        source_map = {c["id"]: c.get("source", "reddit") for c in comments}
        anno_map   = {a.comment_id: a for a in annos}

        # Capture relevance stats for the report / debug panel
        from insighthub.core.relevance import pre_filter_comments, extract_query_terms
        _relevance_stats = {
            "query_terms":   extract_query_terms(query),
            "total_scraped": len(reviews),
            "after_filter":  len(comments),
            "filter_rate":   round(1 - len(comments) / max(1, len(reviews)), 3),
        }

        # Step 5 — consensus + summary
        _update("Building consensus and generating summary…")

        if intent_schema.intent == "RANKING":
            from insighthub.services.llm import classify_query as _classify_query
            _qcat = _classify_query(query)
            ranking = rank_entities_with_relaxation(annos, upvote_map, intent_schema.entity_type, query=query, comments=comments, source_map=source_map, query_category=_qcat)
            if intent_schema.entity_type and ranking:
                valid = set(llm_service.filter_entities_by_type([e.name for e in ranking], intent_schema.entity_type))
                ranking = [e for e in ranking if e.name in valid]
            ranking = llm_service.validate_entity_locations(ranking, query)

            # Enrich entities with a real photo (cached, incl. misses). Place
            # categories use Google Places → Yelp so we don't burn quota resolving
            # products; everything else falls back to the free keyless Wikipedia
            # lead image. The category comes from GPT (no hardcoded keyword rules).
            try:
                from insighthub.services.image_enrichment import (
                    get_image_service, PLACE_PROVIDERS, GENERIC_PROVIDERS)
                _img = get_image_service()
                _is_place = _qcat in ("local_discovery", "service_review")
                _chain = PLACE_PROVIDERS if _is_place else GENERIC_PROVIDERS
                # context: places get the query (location matters); generic
                # entities get the GPT-derived entity type, which disambiguates
                # single-word names ("Storm" → the Marvel character, not weather)
                # while keeping cache entries shared across queries
                _ctx = query if _is_place else (intent_schema.entity_type or "").replace("_", " ")
                for e in ranking:
                    e.image_url = _img.get_image_url(e.name, _ctx, providers=_chain)
            except Exception as _e:
                logger.warning(f"Image enrichment skipped: {_e}")

            _insufficient = ConfidenceConfig.TIER_LABELS["insufficient"]
            # Primary picks vs. low-confidence long tail surfaced by the sparse-rescue path.
            confident = [e for e in ranking if e.confidence_tier != _insufficient]
            also_mentioned = [e for e in ranking if e.confidence_tier == _insufficient]
            # Keep the GPT summary focused on the trustworthy picks only.
            summary_source = confident or ranking
            ranking_data = [{"name": e.name, "overall_stars": e.overall_stars, "mentions": e.mentions, "quotes": e.quotes} for e in summary_source]
            summary = llm_service.summarize_ranking_with_gpt(query, ranking_data)
            def _rank_dict(e):
                return {"name": e.name, "overall_stars": e.overall_stars, "aspect_scores": e.aspect_scores,
                        "mentions": e.mentions, "confidence": e.confidence, "quotes": e.quotes,
                        "confidence_tier": e.confidence_tier, "image_url": e.image_url,
                        # Four-factor RANKING confidence (volume/diversity/consistency/
                        # source-fit + corroboration/evidence lift) — distinct from the
                        # GPT extraction confidence above. This is what the UI shows.
                        "confidence_score": e.confidence_score}
            payload = {
                "query": query, "intent": intent_schema.intent, "summary": summary,
                "metadata": {"timestamp": time.time()},
                "ranking": [_rank_dict(e) for e in confident],
                "also_mentioned": [_rank_dict(e) for e in also_mentioned],
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
            overall, aspect_averages = aggregate_generic(intent_schema.aspects, annos, upvote_map, source_map)
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

        # Collapse progress to single completion line
        _prog.markdown(
            f'<div class="ih-progress-done">'
            f'✓ &nbsp; Analysis complete — {len(comments)} discussions processed'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── RESULTS → persist to session_state; rendered by render_results() ──
        if not reviews:
            st.session_state.pop("results", None)
            st.warning("No reviews found. Try a broader search term or increase depth in settings.")
        else:
            # reset progressive-disclosure counters for the new result set
            for _k in ("rank_n", "ev_n"):
                st.session_state.pop(_k, None)
            st.session_state["results"] = {
                "query": query,
                "intent": intent_schema.intent,
                "entity_type": intent_schema.entity_type,
                "aspects": intent_schema.aspects,
                "payload": payload,
                "comments": comments,
                "annos": annos,
                "platform_breakdown": platform_breakdown,
                "search_time": search_time,
                "subreddits": _subreddits_used,
                "use_cross": use_cross,
                "show_debug": show_debug,
            }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {e}")
        st.info("Please try again with a different search term.")

elif run_analysis and not query.strip():
    st.warning("Enter a search term to begin.")

# Render the tabbed dashboard from persisted state. This runs on every rerun
# (including cheap widget/filter interactions) so the Evidence filters and
# Load-more buttons re-render WITHOUT re-running the analysis pipeline.
if st.session_state.get("results"):
    render_results(st.session_state["results"])


def main():
    """Entry point — Streamlit runs the module directly."""
    pass
