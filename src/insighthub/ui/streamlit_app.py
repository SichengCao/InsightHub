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

# Editorial sample cards — illustrate query types, not real analysis data
FEATURED = [
    {
        "title": "Tesla Model Y",
        "cat": "RANKING · AUTOMOTIVE",
        "img": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=1200&q=80",
        "insight": "Long-term owners love the tech and range — but quality control debates persist across r/TeslaMotors and YouTube owner reviews.",
        "query": "Tesla Model Y",
    },
]

GRID_CARDS = [
    {
        "title": "Sony WH-1000XM5",
        "cat": "HEADPHONES",
        "img": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&q=80",
        "insight": "Near-universal praise. Community consensus: best ANC headphones available.",
        "score": None, "score_color": "#6366f1",
        "meta": "Example analysis",
        "query": "Sony WH-1000XM5 headphones",
    },
    {
        "title": "Best Espresso Machine <$500",
        "cat": "HOME · COFFEE",
        "img": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=600&q=80",
        "insight": "Breville Barista Express dominates. Gaggia Classic Pro is the budget favourite.",
        "score": None, "score_color": "#6366f1",
        "meta": "Example analysis",
        "query": "best espresso machine under $500",
    },
    {
        "title": "Best Golf Courses — Bay Area",
        "cat": "SPORTS · LOCAL",
        "img": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=600&q=80",
        "insight": "Locals debate Harding Park vs TPC Harding. Pasatiempo named hidden gem.",
        "score": None, "score_color": "#6366f1",
        "meta": "Example analysis",
        "query": "best golf course in bay area",
    },
    {
        "title": "MacBook Air M3",
        "cat": "TECH · LAPTOPS",
        "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&q=80",
        "insight": "Reddit's most recommended laptop. M3 chip upgrade called a meaningful step.",
        "score": None, "score_color": "#6366f1",
        "meta": "Example analysis",
        "query": "MacBook Air M3",
    },
]

DEBATES = [
    {
        "icon": "🎮", "icon_bg": "rgba(99,102,241,0.15)", "icon_border": "rgba(99,102,241,0.3)",
        "tag": "HOT", "tag_color": "#f87171", "tag_bg": "rgba(248,113,113,0.12)",
        "title": "Nintendo Switch 2 — Is the $449 price tag justified?",
        "sub": "r/NintendoSwitch · r/gaming · YouTube reviews",
        "stat": "→", "stat_color": "#475569", "stat_lbl": "analyze",
        "query": "Nintendo Switch 2",
    },
    {
        "icon": "🎧", "icon_bg": "rgba(34,211,160,0.12)", "icon_border": "rgba(34,211,160,0.25)",
        "tag": "CONSENSUS", "tag_color": "#22d3a0", "tag_bg": "rgba(34,211,160,0.1)",
        "title": "Sony WH-1000XM5 vs Bose QC45 — ANC debate has a clear winner",
        "sub": "r/headphones · r/audiophile · YouTube shootouts",
        "stat": "→", "stat_color": "#475569", "stat_lbl": "analyze",
        "query": "Sony WH-1000XM5 headphones",
    },
    {
        "icon": "🚗", "icon_bg": "rgba(245,158,11,0.12)", "icon_border": "rgba(245,158,11,0.25)",
        "tag": "DEBATED", "tag_color": "#f59e0b", "tag_bg": "rgba(245,158,11,0.1)",
        "title": "Tesla Model Y long-term reliability — owners divided after 2 years",
        "sub": "r/TeslaMotors · r/electricvehicles · YouTube vlogs",
        "stat": "→", "stat_color": "#475569", "stat_lbl": "analyze",
        "query": "Tesla Model Y",
    },
    {
        "icon": "☕", "icon_bg": "rgba(99,102,241,0.1)", "icon_border": "rgba(99,102,241,0.2)",
        "tag": "TRENDING", "tag_color": "#818cf8", "tag_bg": "rgba(99,102,241,0.12)",
        "title": "Breville vs Gaggia espresso machines — which wins under $500?",
        "sub": "r/espresso · r/Coffee · YouTube reviews",
        "stat": "→", "stat_color": "#475569", "stat_lbl": "analyze",
        "query": "best espresso machine under $500",
    },
    {
        "icon": "🖥️", "icon_bg": "rgba(34,211,160,0.1)", "icon_border": "rgba(34,211,160,0.2)",
        "tag": "STRONG BUY", "tag_color": "#22d3a0", "tag_bg": "rgba(34,211,160,0.1)",
        "title": "MacBook Air M3 — Reddit's most recommended laptop three months running",
        "sub": "r/apple · r/macbook · r/laptops · YouTube reviews",
        "stat": "→", "stat_color": "#475569", "stat_lbl": "analyze",
        "query": "MacBook Air M3",
    },
]

if not run_analysis and not st.session_state.get("results"):
    # ── Hero insight card ─────────────────────────────────────────────────────
    st.markdown('<div class="ih-home-lbl">Featured Intelligence</div>', unsafe_allow_html=True)

    feat = FEATURED[0]
    st.markdown(f"""
<div class="ih-hero-card">
  <img class="ih-hero-card-img" src="{feat['img']}" alt="{feat['title']}"/>
  <div class="ih-hero-card-body">
    <div class="ih-hero-card-top">
      <span class="ih-hero-card-cat">{feat['cat']}</span>
      <span class="ih-hero-card-sources" style="color:#475569;font-size:0.7rem;font-style:italic">Example — run your own analysis below</span>
    </div>
    <div class="ih-hero-card-bottom">
      <div class="ih-hero-card-title">{feat['title']}</div>
      <div class="ih-hero-card-insight">{feat['insight']}</div>
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
            # entity → a YouTube video that features it, for card imagery
            _vid_of = {_rid(r): _vid(r) for r in reviews if _is_yt(r)}
            ent_video = {}
            for a in annos:
                v = _vid_of.get(a.comment_id)
                if not v:
                    continue
                for e in (getattr(a, "entities", []) or []):
                    if getattr(e, "confidence", 0) >= 0.5 and e.name and e.name not in ent_video:
                        ent_video[e.name] = v

            _ent_word = (entity_type.split("_")[-1].replace("-", " ").title() + "s") if entity_type else "Entities"

            # ── 1. metric cards ──
            s3v = (len(ranking) if intent == "RANKING"
                   else len(payload.get("solutions", [])) if intent == "SOLUTION"
                   else f'{payload.get("overall", 0):.1f}')
            s3l = (f"{_ent_word} Ranked" if intent == "RANKING"
                   else "Solutions Found" if intent == "SOLUTION" else "Average Score")
            mcards = [("◈", len(comments), "Discussions", "Analyzed"),
                      ("❝", len(meaningful), "Detailed Reviews", "Extracted"),
                      ("★", s3v, s3l, ""),
                      ("◷", f"{search_time:.1f}s", "Analysis Time", "Completed"),
                      ("⬡", len(platform_breakdown), "Platforms", plat_str)]
            st.markdown(
                '<div class="ov-metrics">' + "".join(
                    f'<div class="ov-mcard"><div class="ov-mic">{ic}</div><div class="ov-mval">{val}</div>'
                    f'<div class="ov-mlbl">{lbl}</div><div class="ov-msub">{sub}</div></div>'
                    for ic, val, lbl, sub in mcards) + '</div>',
                unsafe_allow_html=True)

            # ── 2. sentiment donut · top aspects · source breakdown ──
            tot = max(1, len(pos) + len(mixed) + len(crit))
            pp, npct = round(len(pos) / tot * 100), round(len(mixed) / tot * 100)
            cp = max(0, 100 - pp - npct)
            donut = (f'<div class="ov-donut" style="background:conic-gradient(#22c58a 0 {pp}%,'
                     f'#5b6b7f {pp}% {pp+npct}%,#f0655a {pp+npct}% 100%)"><div class="ov-donut-h"></div></div>')
            senti = (f'<div class="ov-panel"><div class="ov-ptitle">Sentiment Overview</div>'
                     f'<div class="ov-senti">{donut}<div class="ov-legend">'
                     f'<div><span class="dot" style="background:#22c58a"></span>Positive<b>{pp}%</b></div>'
                     f'<div><span class="dot" style="background:#5b6b7f"></span>Neutral<b>{npct}%</b></div>'
                     f'<div><span class="dot" style="background:#f0655a"></span>Critical<b>{cp}%</b></div>'
                     f'</div></div></div>')

            agg = {}
            for it in ranking:
                for k, v in (it.get("aspect_scores") or {}).items():
                    agg.setdefault(k, []).append((v, it.get("mentions", 1)))
            asp_avg = [(k, sum(a * b for a, b in vs) / (sum(b for _, b in vs) or 1)) for k, vs in agg.items()]
            if not asp_avg and payload.get("aspects"):
                asp_avg = list(payload["aspects"].items())
            asp_avg.sort(key=lambda x: -x[1])
            _ac = ["#22c58a", "#3b82f6", "#8b5cf6", "#f59e0b", "#eab308"]
            asp_rows = "".join(
                f'<div class="ov-asp"><span class="lbl">{k.replace("_"," ").title()}</span>'
                f'<span class="track"><span class="fill" style="width:{round(v/5*100)}%;background:{_ac[i%5]}"></span></span>'
                f'<span class="v">{round(v/5*100)}%</span></div>'
                for i, (k, v) in enumerate(asp_avg[:5])) or '<div class="ov-empty">No aspect data</div>'
            aspects_panel = f'<div class="ov-panel"><div class="ov-ptitle">Top Aspects Discussed</div>{asp_rows}</div>'

            disc = {}
            for r in reviews:
                disc.setdefault("YouTube" if _is_yt(r) else "Reddit", set()).add(_get(r, "thread_id") or _rid(r))
            src_rows = "".join(
                f'<div class="ov-srcrow"><span class="ov-srcic {"yt" if s=="YouTube" else "rd"}">'
                f'{"▶" if s=="YouTube" else "r/"}</span><div><div class="ov-srcname">{s}</div>'
                f'<div class="ov-srccnt"><b>{len(t)}</b> discussions</div></div></div>'
                for s, t in sorted(disc.items()))
            src_panel = f'<div class="ov-panel"><div class="ov-ptitle">Source Breakdown</div>{src_rows}</div>'
            st.markdown(f'<div class="ov-3col">{senti}{aspects_panel}{src_panel}</div>', unsafe_allow_html=True)

            # ── 3. top ranked picks (visual cards) ──
            if ranking:
                st.markdown(f'<div class="ov-sechdr">Top Ranked {_ent_word}</div>', unsafe_allow_html=True)
                picks = ""
                for i, it in enumerate(ranking[:6], 1):
                    # Image priority: enriched entity photo → a YouTube video that
                    # features it → gradient placeholder only when both truly fail.
                    photo = it.get("image_url")
                    vid = ent_video.get(it["name"])
                    if photo:
                        img = f'<div class="ov-pimg" style="background-image:url({photo})">'
                    elif vid:
                        img = f'<div class="ov-pimg" style="background-image:url(https://i.ytimg.com/vi/{vid}/hqdefault.jpg)">'
                    else:
                        img = f'<div class="ov-pimg ov-pph" style="background:{_grad_for(it["name"])}"><span>{it["name"][:1].upper()}</span>'
                    rc = f"g{i}" if i <= 3 else ""
                    img += f'<span class="ov-rbadge {rc}">{i}</span></div>'
                    cat = entity_type.replace("_", " ").title() if entity_type else ""
                    desc = _excerpt(it["quotes"][0], 88) if it.get("quotes") else _entity_verdict(it)
                    picks += (
                        f'<div class="ov-pick">{img}<div class="ov-pbody">'
                        f'<div class="ov-pname">{it["name"]}</div>'
                        f'<div class="ov-prate">{_stars(it["overall_stars"])}<span class="ov-pscore">{it["overall_stars"]:.1f}</span></div>'
                        f'<div class="ov-pcat">{cat}</div>'
                        f'<div class="ov-pdesc">{desc}</div>'
                        f'<div class="ov-pment">{it["mentions"]} mention{"s" if it["mentions"]!=1 else ""}</div>'
                        f'</div></div>')
                st.markdown(f'<div class="ov-picks">{picks}</div>', unsafe_allow_html=True)

            # ── 4. community highlights: YouTube videos + Reddit discussions ──
            videos = {}
            for r in reviews:
                if _is_yt(r):
                    v = _vid(r)
                    if v and v not in videos:
                        m = _get(r, "meta", {}) or {}
                        videos[v] = {"vid": v, "title": m.get("video_title", "Video"), "ch": m.get("channel_title", ""),
                                     "thumb": m.get("thumbnail_url") or f"https://i.ytimg.com/vi/{v}/hqdefault.jpg"}
            threads = {}
            for r in reviews:
                if not _is_yt(r):
                    tid = _get(r, "thread_id") or _rid(r)
                    up = _get(r, "upvotes", 0) or 0
                    cur = threads.get(tid)
                    if not cur:
                        threads[tid] = {"title": _get(r, "post_title") or _excerpt(_txt(r), 70),
                                        "up": up, "sub": _subr(r), "perma": _get(r, "permalink")}
                    else:
                        cur["up"] = max(cur["up"], up)
            vids = list(videos.values())[:3]
            rthreads = sorted(threads.values(), key=lambda x: -x["up"])[:4]

            if vids or rthreads:
                st.markdown('<div class="ov-sechdr">Community Highlights</div>', unsafe_allow_html=True)
                colv, colr = st.columns(2)
                if vids:
                    vcards = "".join(
                        f'<a class="ov-vid" href="https://youtube.com/watch?v={v["vid"]}" target="_blank">'
                        f'<div class="ov-vthumb" style="background-image:url({v["thumb"]})">'
                        f'<span class="ov-play">▶</span></div>'
                        f'<div class="ov-vtitle">{v["title"]}</div>'
                        f'<div class="ov-vch">{v["ch"]}</div></a>' for v in vids)
                    colv.markdown('<div class="ov-sub2">Top YouTube Videos</div>'
                                  f'<div class="ov-vgrid">{vcards}</div>', unsafe_allow_html=True)
                if rthreads:
                    rrows = "".join(
                        f'<a class="ov-rd" href="https://reddit.com{t["perma"]}" target="_blank">'
                        f'<span class="ov-rdic">r/</span><div><div class="ov-rdtitle">{t["title"]}</div>'
                        f'<div class="ov-rdmeta">{("r/"+t["sub"]+" · ") if t["sub"] else ""}▲ {t["up"]:,} upvotes</div>'
                        f'</div></a>' for t in rthreads)
                    colr.markdown('<div class="ov-sub2">Top Reddit Discussions</div>'
                                  f'<div class="ov-rdlist">{rrows}</div>', unsafe_allow_html=True)

            # ── 5. key takeaways + what's next ──
            sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', payload.get("summary", "")) if len(s.strip()) > 14][:5]
            colk, coln = st.columns(2)
            if sents:
                tk = "".join(f'<div class="ov-tk"><span class="ov-tkchk">✓</span><span>{s}</span></div>' for s in sents)
                colk.markdown(f'<div class="ov-panel"><div class="ov-ptitle">Key Takeaways</div>{tk}</div>',
                              unsafe_allow_html=True)
            nexts = [("❝", "Explore Full Reviews", "Dive into individual reviews and discussions"),
                     ("▤", "Compare " + _ent_word, "Side-by-side of the top-rated picks"),
                     ("⇪", "Save & Share", "Export results or share with friends")]
            nn = "".join(f'<div class="ov-nx"><span class="ov-nxic">{i}</span>'
                         f'<div><div class="ov-nxt">{t}</div><div class="ov-nxs">{s}</div></div>'
                         f'<span class="ov-nxarrow">→</span></div>' for i, t, s in nexts)
            coln.markdown(f'<div class="ov-panel"><div class="ov-ptitle">What\'s Next?</div>{nn}</div>',
                          unsafe_allow_html=True)

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

            # Enrich place-like entities with a real photo (Google Places → Yelp,
            # cached). Gated to place categories so we don't burn quota resolving
            # products; the category comes from GPT (no hardcoded keyword rules).
            if _qcat in ("local_discovery", "service_review"):
                try:
                    from insighthub.services.image_enrichment import get_image_service
                    _img = get_image_service()
                    if _img.enabled:
                        for e in ranking:
                            e.image_url = _img.get_image_url(e.name, query)
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
