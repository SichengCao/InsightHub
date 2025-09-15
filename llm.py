"""LLM service for OpenAI integration."""

import logging
import json
import re
import unicodedata
import string
from textwrap import dedent
from typing import Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from diskcache import Cache
from config import settings
from aspect import get_domain_aspects, aspect_hint_for_query

# ---- Search planner constants ----
SEARCH_PLANNER_PROMPT = dedent("""
You are a search planner for finding the best Reddit comments for ANY user query.
Return ONLY JSON (no prose).

CRITICAL: Focus on the EXACT product/service/thing mentioned in the query. Avoid tangential topics.

Produce:
- "terms": 4–10 short search strings (deduped), MUST include the exact query and specific variants. Quote exact phrases when useful.
- "subreddits": 0–12 subreddit names (no r/ prefix), discovered from the query; include "all" last
- "time_filter": one of ["day","week","month","year","all"]
- "strategies": 2–3 from ["relevance","top","new"]
- "min_comment_score": integer 0..50
- "per_post_top_n": integer 3..12
- "comment_must_patterns": 0–6 lowercase regexes that should appear in relevant comments - MUST include the main product/service name

Examples:
- For "Nintendo Switch": terms should include "Nintendo Switch" exactly, patterns should require "nintendo"
- For "iPhone 15": terms should include "iPhone 15" exactly, patterns should require "iphone"
- For "Tesla Model Y": terms should include "Tesla Model Y" exactly, patterns should require "tesla"
- For "best golf course in bay area": patterns should require "golf" OR "bay area" OR "course" (not all together)
""").strip()

def _safe_json_loads(s: str):
    """Safely parse JSON from LLM response, handling common formatting issues."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

logger = logging.getLogger(__name__)

# Cache for map-reduce results
_cache = Cache(getattr(settings, "cache_dir", ".cache"))

# Domain-aware helper (using imported function from aspect.py)

# Map-reduce pipeline constants
MAP_PROMPT = dedent("""
Return ONLY JSON.

You receive comments with fields: id, text, upvotes, permalink.

Rules:
- Pros/cons bullets must be grounded ONLY in these texts.
- For EVERY pro/con bullet, include supporting `ids` from the input comments.
- Each quote MUST be a verbatim substring (allowing whitespace/quote normalization) of the input text it cites.
- Put all contributing ids into `coverage_ids`. If unsure, leave fields empty.

JSON schema:
{ "pros":[{"text":str,"ids":[str]}],
  "cons":[{"text":str,"ids":[str]}],
  "aspects":[{"name":str,"score":float,"count":int}],
  "quotes":[{"id":str,"quote":str,"permalink":str}],
  "coverage_ids":[str] }
""").strip()

SUMMARY_PROMPT = dedent("""
You will summarize Reddit comments. Be EXTRACTIVE and EVIDENCE-FIRST.

Rules:
- Use only what is present in the input texts.
- For each pro/cons bullet, include supporting comment IDs in an `ids` array.
- Any quote MUST be a verbatim substring of a provided text (ignoring whitespace/typographic quotes).
- If a claim is not supported by multiple comments, either omit it or clearly mark it with just the single `id`.
- Output ONLY JSON with this schema:
{
  "pros": [{"text": str, "ids": [str]}],
  "cons": [{"text": str, "ids": [str]}],
  "aspects": [{"name": str, "score": float, "count": int}],
  "quotes": [{"id": str, "quote": str, "permalink": str}],
  "coverage_ids": [str]
}
""").strip()

REDUCE_PROMPT = dedent("""
Merge multiple partial JSONs about the same topic. Return ONLY valid JSON.

Rules:
- Deduplicate pros/cons by meaning; keep clearest phrasing; union ids.
- Aspects: weighted average by `count`, round to 1 decimal.
- Keep the 6–10 best quotes maximizing coverage and variety.
- Only cite covered ids; no new claims.

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, no code fences.

Output format:
{
  "pros": [{"text": "bullet text", "ids": ["id1", "id2"]}],
  "cons": [{"text": "bullet text", "ids": ["id3", "id4"]}],
  "aspects": [{"name": "AspectName", "score": 3.5, "count": 5}],
  "quotes": [{"id": "comment_id", "quote": "quote text", "permalink": "url"}],
  "coverage_ids": ["id1", "id2", "id3", "id4"],
  "notes": {"dropped_ids": ["id5", "id6"]}
}
""").strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    return re.sub(r"^```(?:json)?|```$", "", s, flags=re.IGNORECASE|re.MULTILINE).strip()

def _safe_json(s: str) -> dict:
    try:
        data = json.loads(_strip_code_fences(s))
    except Exception:
        # Try to fix common JSON issues
        cleaned = _strip_code_fences(s)
        
        # Fix common delimiter issues - more comprehensive
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)   # remove trailing commas before } or ]
        cleaned = re.sub(r'}\s*{', '},{', cleaned)  # Fix missing commas between objects
        
        try:
            data = json.loads(cleaned)
        except Exception:
            # last-ditch: find first {...} block
            m = re.search(r"\{.*\}", cleaned, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = {}
            else:
                data = {}
    
    # Ensure all required keys exist
    for k in ("pros","cons","aspects","quotes","coverage_ids"):
        data.setdefault(k, [] if k != "coverage_ids" else [])
    return data

# Coverage validation functions
import unicodedata
_WS = re.compile(r"\s+")

def _norm_for_substring(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("'","'").replace("'","'").replace(""",'"').replace(""",'"')
    s = _WS.sub(" ", s).strip().lower()
    return s

def _contains_norm(hay: str, needle: str) -> bool:
    H = _norm_for_substring(hay); N = _norm_for_substring(needle)
    return bool(N) and (N in H)

def _validate_and_fix_coverage(final: dict, id2text: dict) -> dict:
    cov = set()
    quotes_ok = []
    for q in final.get("quotes", []) or []:
        rid = str(q.get("id", ""))
        qt  = q.get("quote", "")
        src = id2text.get(rid, "")
        if _contains_norm(src, qt):
            quotes_ok.append(q); cov.add(rid)
    final["quotes"] = quotes_ok

    # fold in IDs attached to pros/cons
    for sec in ("pros","cons"):
        for item in final.get(sec, []) or []:
            for rid in (item.get("ids", []) or []):
                rid = str(rid)
                if rid in id2text:
                    cov.add(rid)
    final["coverage_ids"] = list(cov)
    return final

def _coverage_fallback(final: dict, id2text: dict):
    if final.get("coverage_ids"): 
        return final
    # naive keyword matching: map each pro/con bullet to ids with >=2 token overlaps (len>=4)
    def toks(s): 
        return {t for t in re.findall(r"[a-z]{4,}", (s or "").lower())}
    text_toks = {rid: toks(txt) for rid, txt in id2text.items()}

    cov = set()
    for sec in ("pros","cons"):
        for item in final.get(sec, []) or []:
            if item.get("ids"):
                continue
            bt = toks(item.get("text",""))
            if not bt:
                continue
            hits = []
            for rid, tt in text_toks.items():
                overlap = len(bt & tt)
                if overlap >= 2:
                    hits.append(rid)
            item["ids"] = hits[:5]  # cap
            cov.update(item["ids"])
    final["coverage_ids"] = list(set(final.get("coverage_ids", [])) | cov)
    return final

def _chunk(items, n=12):
    for i in range(0, len(items), n):
        yield items[i:i+n]

# Normalization helpers for substring matching
_WS = re.compile(r"\s+")
def _norm_for_substring(s: str) -> str:
    """Normalize quotes, collapse whitespace, strip control chars; case-insensitive."""
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    s = s.replace("'","'").replace(""",'"').replace(""",'"')
    s = "".join(ch for ch in s if ch not in "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f")
    s = _WS.sub(" ", s).strip().lower()
    return s

def _contains_norm(hay: str, needle: str) -> bool:
    """Check if needle is contained in hay after normalization."""
    H = _norm_for_substring(hay)
    N = _norm_for_substring(needle)
    return bool(N) and (N in H)

def _validate_and_fix_coverage(final: dict, id2text: dict) -> dict:
    """Validate quotes with normalized substring matching and recompute coverage."""
    cov = set()
    quotes = []
    # accept quotes that are substrings after normalization
    for q in final.get("quotes", []) or []:
        rid = str(q.get("id", ""))
        qt  = q.get("quote", "")
        txt = id2text.get(rid, "")
        if _contains_norm(txt, qt):
            quotes.append(q)
            cov.add(rid)
    # also collect ids referenced by pros/cons
    for sec in ("pros","cons"):
        for item in final.get(sec, []) or []:
            for rid in item.get("ids", []) or []:
                rid = str(rid)
                if rid in id2text:
                    cov.add(rid)
    final["quotes"] = quotes
    final["coverage_ids"] = list(cov)
    return final


class LLMServiceFactory:
    """Factory for creating LLM services."""
    
    @staticmethod
    def create():
        """Create appropriate LLM service."""
        if settings.effective_openai_key:
            return OpenAIService()
        else:
            return FallbackLLMService()


class OpenAIService:
    """OpenAI-based LLM service."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.effective_openai_key)
        # Prefer a small-but-strong JSON-friendly model
        self.model = getattr(settings, "openai_model", None) or "gpt-4o-mini"
        logger.info("OpenAI service initialized")
    
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Generic chat method for LLM interactions."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def plan_reddit_search(self, query: str) -> dict:
        """Plan Reddit search strategy using LLM."""
        try:
            sys = "You output strict JSON to plan Reddit searches."
            user = f"Query: {query}\n\n{SEARCH_PLANNER_PROMPT}"
            resp = self.chat(sys, user, temperature=0.2, max_tokens=400)
            plan = _safe_json_loads(resp)

            # Defensive clamps
            plan.setdefault("terms", [query])
            plan["terms"] = list(dict.fromkeys([t for t in plan["terms"] if isinstance(t, str) and t.strip()]))[:10]

            subs = [s.replace("r/","").strip() for s in plan.get("subreddits", []) if isinstance(s, str)]
            if "all" not in [s.lower() for s in subs]:
                subs.append("all")
            plan["subreddits"] = subs[:12]

            plan["time_filter"] = plan.get("time_filter") or "year"
            plan["strategies"] = [s for s in plan.get("strategies", ["relevance","top"]) if s in ("relevance","top","new")] or ["relevance","top"]
            plan["min_comment_score"] = max(0, int(plan.get("min_comment_score", 1)))
            plan["per_post_top_n"] = min(12, max(3, int(plan.get("per_post_top_n", 8))))

            pats = plan.get("comment_must_patterns") or []
            plan["comment_must_patterns"] = [p for p in pats if isinstance(p, str) and p.strip()][:6]
            return plan
        except Exception as e:
            logger.error(f"Reddit search planning failed: {e}")
            # Return a safe fallback plan
            return {
                "terms": [query],
                "subreddits": ["all"],
                "time_filter": "year",
                "strategies": ["relevance", "top"],
                "min_comment_score": 1,
                "per_post_top_n": 8,
                "comment_must_patterns": [r"\b(love|hate|recommend|avoid|worth|issue|problem|help)\b"]
            }
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def analyze_comment(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Analyze comment sentiment and aspects."""
        try:
            prompt = f"""
            Analyze the sentiment of this review text and provide a rating from 1-5 stars:
            
            Text: "{text}"
            
            Return JSON with: {{"sentiment": "POSITIVE/NEGATIVE/NEUTRAL", "stars": 1-5, "reasoning": "brief explanation"}}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
                timeout=10
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Convert to our format
            sentiment = result.get("sentiment", "NEUTRAL")
            stars = float(result.get("stars", 3.0))
            
            # Map sentiment to compound score
            if sentiment == "POSITIVE":
                compound = 0.5
            elif sentiment == "NEGATIVE":
                compound = -0.5
            else:
                compound = 0.0
            
            return {
                "compound": compound,
                "label": sentiment,
                "stars": stars
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return {
                "compound": 0.0,
                "label": "NEUTRAL",
                "stars": 3.0
            }
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def generate_pros_cons(self, reviews: List[Any], query: str) -> Dict[str, str]:
        """Generate pros and cons from reviews."""
        try:
            # Collect review texts - filter for meaningful length
            meaningful_reviews = [r for r in reviews if len(r.get("text") if isinstance(r, dict) else r.text) > 100]
            review_texts = [(r.get("text") if isinstance(r, dict) else r.text)[:300] for r in meaningful_reviews[:10]]  # Use more reviews
            reviews_summary = "\n".join([f"- {text}" for text in review_texts])
            
            hint = aspect_hint_for_query(query)
            prompt = f"""
            {hint}
            
            Analyze these detailed reviews for "{query}" and provide specific insights:
            
            Reviews:
            {reviews_summary}
            
            Instructions:
            - Extract SPECIFIC positive and negative aspects mentioned in the reviews
            - Write a precise summary that reflects the actual content and sentiment patterns
            - Include specific details, numbers, and concrete examples from the reviews
            - Avoid generic statements - be specific to what users actually said
            
            Return JSON with:
            {{
                "pros": ["5 specific positive aspects with details from reviews"],
                "cons": ["5 specific negative aspects with details from reviews"],
                "summary": "precise paragraph summarizing the actual review content, specific user experiences, and concrete findings. Include specific details and examples from the reviews."
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
                timeout=20
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI pros/cons generation failed: {e}")
            # Analyze actual review content for fallback
            positive_aspects = []
            negative_aspects = []
            
            for review in reviews[:10]:  # Analyze first 10 reviews
                text_lower = review.text.lower()
                if "battery" in text_lower or "charge" in text_lower:
                    if "good" in text_lower or "great" in text_lower:
                        positive_aspects.append("Battery life and charging performance")
                    elif "poor" in text_lower or "bad" in text_lower:
                        negative_aspects.append("Battery life and charging issues")
                
                if "camera" in text_lower or "photo" in text_lower:
                    if "excellent" in text_lower or "amazing" in text_lower:
                        positive_aspects.append("Camera quality and photo capabilities")
                    elif "mediocre" in text_lower or "disappointing" in text_lower:
                        negative_aspects.append("Camera quality concerns")
                
                if "price" in text_lower or "expensive" in text_lower:
                    negative_aspects.append("Pricing and value concerns")
                
                if "design" in text_lower or "build" in text_lower:
                    if "premium" in text_lower or "quality" in text_lower:
                        positive_aspects.append("Design and build quality")
            
            # Remove duplicates and limit to 5 each
            positive_aspects = list(set(positive_aspects))[:5]
            negative_aspects = list(set(negative_aspects))[:5]
            
            # Generate specific summary
            pos_count = sum(1 for r in reviews if r.sentiment_label == 'POSITIVE')
            neg_count = sum(1 for r in reviews if r.sentiment_label == 'NEGATIVE')
            avg_rating = sum(r.stars for r in reviews) / len(reviews) if reviews else 3.0
            
            summary = f"Analysis of {len(reviews)} reviews reveals {pos_count} positive, {neg_count} negative experiences with an average rating of {avg_rating:.1f}/5 stars. "
            if positive_aspects:
                summary += f"Users specifically praised: {', '.join(positive_aspects[:3])}. "
            if negative_aspects:
                summary += f"Main concerns include: {', '.join(negative_aspects[:3])}. "
            summary += f"The {query} shows {'generally positive' if avg_rating > 3.5 else 'mixed' if avg_rating > 2.5 else 'negative'} sentiment overall."
            
            return {
                "pros": positive_aspects if positive_aspects else ["Quality features", "User experience", "Performance", "Design", "Value"],
                "cons": negative_aspects if negative_aspects else ["Pricing concerns", "Service issues", "Reliability questions", "Limited features", "Support problems"],
                "summary": summary
            }
    
    def _map_phase(self, comments, query=""):
        """Map phase: extract evidence from comment chunks."""
        partials = []
        
        # Get aspect hint for the query
        hint = aspect_hint_for_query(query)
        map_prompt = (hint + "\n" + MAP_PROMPT).strip() if hint else MAP_PROMPT
        
        for group in _chunk(comments, n=12):
            payload = [{
                "id": str(getattr(c, "id", f"idx-{i}")) if not isinstance(c, dict) else str(c.get("id", f"idx-{i}")),
                "text": (getattr(c, "text", getattr(c, "body", "")) if not isinstance(c, dict) else c.get("text", ""))[:1400],
                "upvotes": int(getattr(c, "upvotes", getattr(c, "score", 0)) if not isinstance(c, dict) else c.get("upvotes", c.get("score", 0)) or 0),
                "permalink": (
                    (getattr(c, "url", "") if not isinstance(c, dict) else c.get("url", "")) or
                    (f"https://reddit.com{getattr(c, 'permalink', '')}" if not isinstance(c, dict) else (f"https://reddit.com{c.get('permalink','')}" if c.get("permalink") else ""))
                )
            } for i, c in enumerate(group)]
            
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"user","content": map_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)}],
                    max_tokens=800, 
                    temperature=0.2
                )
                partials.append(_safe_json(resp.choices[0].message.content))
            except Exception as e:
                logger.error(f"Map phase failed for group: {e}")
                # Add empty partial on failure
                partials.append({"pros": [], "cons": [], "aspects": [], "quotes": [], "coverage_ids": []})
        return partials
    
    def _reduce_phase(self, partials):
        """Reduce phase: merge partial results."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content": REDUCE_PROMPT + "\n\n" + json.dumps(partials, ensure_ascii=False)}],
                max_tokens=800, 
                temperature=0.1
            )
            final = _safe_json(resp.choices[0].message.content)
            # clamp aspect scores without biasing upward
            for a in final.get("aspects", []):
                try:
                    a["score"] = max(1.0, min(5.0, float(a.get("score", 3.0))))
                except Exception:
                    a["score"] = 3.0
            return final
        except Exception as e:
            logger.error(f"Reduce phase failed: {e}")
            # Fallback: merge partials manually
            merged = {"pros": [], "cons": [], "aspects": [], "quotes": [], "coverage_ids": []}
            for p in partials:
                merged["pros"].extend(p.get("pros", []))
                merged["cons"].extend(p.get("cons", []))
                merged["aspects"].extend(p.get("aspects", []))
                merged["quotes"].extend(p.get("quotes", []))
                merged["coverage_ids"].extend(p.get("coverage_ids", []))
            return merged
    
    def summarize_comments_map_reduce(self, comments, query=""):
        """Evidence-first map-reduce summarization with caching and validation."""
        if not comments:
            return {"pros": [], "cons": [], "aspects": [], "quotes": [], "coverage_ids": []}
        
        def _get(c, k, default=None):
            return (c.get(k, default) if isinstance(c, dict) else getattr(c, k, default))
        
        # Create cache key from comment IDs
        key = ("mapreduce", self.model, query, tuple(sorted(_get(c, "id", str(i)) for i, c in enumerate(comments))))
        hit = _cache.get(key)
        if hit: 
            return hit
        
        # IMPORTANT: keys must match payload ids exactly
        id2text = {str(_get(c,"id", i)): (_get(c,"text","") or _get(c,"body","") or "") for i, c in enumerate(comments)}
        
        parts = self._map_phase(comments, query)
        final = self._reduce_phase(parts)
        final = _validate_and_fix_coverage(final, id2text)
        final = _coverage_fallback(final, id2text)  # last resort so coverage isn't 0 in mock mode
        
        # Clamp aspect scores
        for a in final.get("aspects", []):
            try:
                a["score"] = max(1.0, min(5.0, float(a.get("score", 3.0))))
            except Exception:
                a["score"] = 3.0
        
        _cache.set(key, final, expire=12 * 3600)  # 12h TTL
        return final
    
    def summarize_concatenated_comments(self, concatenated_text: str, query: str, reviews: list) -> dict:
        """Analyze concatenated comments for better quality and faster processing."""
        try:
            # Create a mapping of text snippets to review IDs for coverage tracking
            text_to_id = {}
            for review in reviews:
                text = review.get("text") if isinstance(review, dict) else review.text
                if text:
                    # Use first 50 chars as identifier
                    snippet = text[:50].strip()
                    review_id = review.get("id") if isinstance(review, dict) else review.id
                    text_to_id[snippet] = review_id
            
            prompt = f"""
            Analyze these Reddit comments about "{query}" and extract key insights.
            
            Return ONLY JSON with this exact schema:
            {{
                "summary": "comprehensive paragraph summarizing key findings, user sentiment patterns, and main themes from the comments",
                "pros": [{{"text": "specific positive aspect", "ids": ["comment_ids_that_support_this"]}}],
                "cons": [{{"text": "specific negative aspect", "ids": ["comment_ids_that_support_this"]}}],
                "quotes": [{{"id": "comment_id", "quote": "exact_quote_from_text", "permalink": "reddit_permalink"}}],
                "coverage_ids": ["all_comment_ids_referenced"]
            }}
            
            Rules:
            - Write a comprehensive summary paragraph that captures the overall sentiment, key themes, and main user experiences
            - Extract 3-5 specific pros and cons with concrete details
            - For each pro/con, include the comment IDs that support it
            - Include 3-5 representative quotes with exact text from comments
            - All quotes must be verbatim from the provided text
            - Focus on actionable insights and specific user experiences
            
            Comments:
            {concatenated_text[:8000]}  # Limit to avoid token limits
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.3,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Validate and fix coverage
            coverage_ids = set()
            for pro in result.get("pros", []):
                for cid in pro.get("ids", []):
                    coverage_ids.add(str(cid))
            for con in result.get("cons", []):
                for cid in con.get("ids", []):
                    coverage_ids.add(str(cid))
            for quote in result.get("quotes", []):
                coverage_ids.add(str(quote.get("id", "")))
            
            result["coverage_ids"] = list(coverage_ids)
            
            return result
            
        except Exception as e:
            logger.error(f"Concatenated analysis failed: {e}")
            return {"pros": [], "cons": [], "quotes": [], "coverage_ids": []}


class FallbackLLMService:
    """Fallback LLM service using simple rules."""
    
    def __init__(self):
        logger.info("Using fallback LLM service")
    
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Fallback chat method - returns empty string."""
        logger.warning("Fallback LLM service chat called - no actual LLM available")
        return ""
    
    def plan_reddit_search(self, query: str) -> dict:
        """Fallback search planning."""
        logger.warning("Using fallback Reddit search planning")
        return {
            "terms": [query],
            "subreddits": ["all"],
            "time_filter": "year",
            "strategies": ["relevance", "top"],
            "min_comment_score": 1,
            "per_post_top_n": 8,
            "comment_must_patterns": [r"\b(love|hate|recommend|avoid|worth|issue|problem|help)\b"]
        }
    
    def analyze_comment(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Simple rule-based analysis."""
        text_lower = text.lower()
        
        # Simple keyword-based sentiment
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "disappointing", "poor"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            compound = 0.3
            label = "POSITIVE"
            stars = 4.0
        elif neg_count > pos_count:
            compound = -0.3
            label = "NEGATIVE"
            stars = 2.0
        else:
            compound = 0.0
            label = "NEUTRAL"
            stars = 3.0
        
        return {
            "compound": compound,
            "label": label,
            "stars": stars
        }
    
    def generate_pros_cons(self, reviews: List[Any], query: str) -> Dict[str, str]:
        """Simple pros/cons extraction."""
        return {
            "pros": f"Users generally appreciate the {query} for its features and quality.",
            "cons": f"Some users have concerns about the {query} pricing and availability."
        }
    
    def summarize_comments_map_reduce(self, comments, query=""):
        """Fallback map-reduce summarization."""
        if not comments:
            return {"pros": [], "cons": [], "aspects": [], "quotes": [], "coverage_ids": []}
        
        # Simple fallback: extract basic pros/cons
        pros = []
        cons = []
        aspects = []
        quotes = []
        coverage_ids = []
        
        for i, comment in enumerate(comments[:10]):  # Limit to first 10
            text = getattr(comment, "text", getattr(comment, "body", ""))
            comment_id = getattr(comment, "id", str(i))
            coverage_ids.append(comment_id)
            
            # Simple keyword extraction
            text_lower = text.lower()
            if any(word in text_lower for word in ["good", "great", "excellent", "amazing", "love"]):
                pros.append({"text": f"Positive feedback: {text[:100]}...", "ids": [comment_id]})
            elif any(word in text_lower for word in ["bad", "terrible", "awful", "hate", "worst"]):
                cons.append({"text": f"Negative feedback: {text[:100]}...", "ids": [comment_id]})
            
            # Simple aspect detection
            if "battery" in text_lower:
                aspects.append({"name": "Battery", "score": 3.5, "count": 1})
            if "camera" in text_lower:
                aspects.append({"name": "Camera", "score": 3.5, "count": 1})
            if "price" in text_lower:
                aspects.append({"name": "Price", "score": 3.0, "count": 1})
            
            # Add quote
            quotes.append({
                "id": comment_id,
                "quote": text[:200] + "..." if len(text) > 200 else text,
                "permalink": getattr(comment, "permalink", "")
            })
        
        return {
            "pros": pros[:6],
            "cons": cons[:6], 
            "aspects": aspects[:8],
            "quotes": quotes[:8],
            "coverage_ids": coverage_ids
        }
    
    def summarize_concatenated_comments(self, concatenated_text: str, query: str, reviews: list) -> dict:
        """Fallback concatenated analysis - delegates to map-reduce."""
        logger.warning("Fallback LLM service concatenated analysis called - using map-reduce fallback")
        return self.summarize_comments_map_reduce(reviews, query)
