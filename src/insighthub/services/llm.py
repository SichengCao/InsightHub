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
import hashlib
from ..core.config import settings
from ..core.aspect import get_domain_aspects, aspect_hint_for_query
from ..core.models import IntentSchema, GPTCommentAnno, EntityRef

# ---- Search planner constants ----
from ..core.constants import PromptConstants, SearchConstants, CacheConstants, ErrorConstants, FileConstants
PLANNER_PROMPT_VERSION = PromptConstants.PLANNER_PROMPT_VERSION

SEARCH_PLANNER_PROMPT = dedent("""
You are a search planner for finding the best Reddit comments for ANY user query.
Return ONLY JSON (no prose). Use English keys; keep values in the query's language.

INPUT (via user message): the raw user query (e.g., "best golf course in bay area").

Rules:
- Focus on the EXACT product/service/topic in the query, but add SHORT, meaningful variants:
  * brand/model aliases, common abbreviations, and for places: local nicknames (e.g., "SF Bay Area","East Bay","Peninsula","Silicon Valley","Marin").
- SUBREDDIT SELECTION STRATEGY:
  * For location queries: prioritize LOCAL subreddits (city/region names) over general topic subreddits
    Example: "NYC restaurants" → ['AskNYC', 'NewYorkCity', 'FoodNYC'] NOT ['food', 'restaurants']
  * For product queries: prioritize BRAND-SPECIFIC subreddits over general category subreddits  
    Example: "iPhone 15" → ['iPhone', 'Apple'] NOT ['smartphones', 'technology']
  * For services: prioritize SERVICE-SPECIFIC subreddits
  * NEVER include "all" subreddit - it returns too much irrelevant content
  * Prefer active, relevant subreddits with 10K+ members
  * Choose the MOST relevant subreddits - prioritize quality over quantity
  * For NYC queries: start with ['AskNYC', 'NewYorkCity', 'FoodNYC'] and add more relevant subreddits as needed
- All arrays MUST be case-insensitively deduped and length-bounded per spec.
- Never include "r/" prefixes in subreddit names.
- For comment_must_patterns: Include key product/service terms that should appear in relevant comments.
- For "Tesla Model Y" → include patterns like "\\btesla\\b", "\\bmodel y\\b", "\\bev\\b"
- For "iPhone 15" → include patterns like "\\biphone\\b", "\\bapple\\b", "\\bphone\\b"
- For "best golf course" → include patterns like "\\bgolf\\b", "\\bcourse\\b"
- Only return empty list if no clear product/service terms exist.

Also infer an internal intent to guide your choices:
- If the query is a "best/top/which/where" compare: treat as RANKING.
- If "how to/fix/solution/workaround": treat as SOLUTION.
- Otherwise: GENERIC.

Produce JSON with exactly these keys:
- "terms": 2–4 short search strings (must include the exact raw query once).
  * Make terms SPECIFIC and TARGETED to avoid irrelevant results
  * For location queries: include specific place names, avoid generic terms like "best food"
  * For product queries: include specific model names, avoid generic category terms
  * PRIORITIZE specificity over recall - better to get fewer, more relevant results
- "subreddits": 2–6 names (no "r/" prefix). AVOID "all" unless absolutely necessary. PRIORITIZE the most relevant and active subreddits.
- "time_filter": one of ["day","week","month","year","all"].
- "strategies": 1–2 from ["relevance","top","new"].
- "min_comment_score": integer 0..50.
- "per_post_top_n": integer 3..8.
- "comment_must_patterns": 0–2 lowercase regexes, simple words with \\b boundaries (example: "\\\\bpace\\\\b").

Heuristics by inferred intent:
- RANKING → strategies: ["top","relevance"]; time_filter: "year" (if evergreen) else "all"; per_post_top_n: 5–8; min_comment_score: 3–8 (higher for quality).
- SOLUTION → strategies: ["new","relevance"]; time_filter: "week" or "month"; per_post_top_n: 6–10; min_comment_score: 0–3.
- GENERIC → strategies: ["relevance","top"]; time_filter: "month" or "year"; per_post_top_n: 5–8; min_comment_score: 2–5 (higher for quality).

Output strict JSON only. No comments, no trailing commas.
""").strip()

def _safe_json_loads(s: str):
    """Safely parse JSON from LLM response, handling common formatting issues."""
    try:
        return json.loads(s)
    except Exception:
        # Try to extract JSON array first
        array_match = re.search(r'\[.*\]', s, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except:
                pass
        
        # Try to extract JSON object
        obj_match = re.search(r"\{.*\}", s, re.S)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except:
                pass
        
        # If all else fails, try to clean the string
        cleaned = s.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except:
            raise ValueError(f"Could not parse JSON from: {s[:200]}...")

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

# Coverage validation functions (using the first _norm_for_substring function)

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
        # Use different models for different tasks to optimize cost
        self.model = getattr(settings, "openai_model", None) or "gpt-4o-mini"
        self.fast_model = "gpt-3.5-turbo"  # For simple tasks
        self.smart_model = "gpt-4o-mini"   # For complex analysis
        self.cache = Cache(FileConstants.CACHE_DIR)  # Cache for API responses
        logger.info("OpenAI service initialized with caching")
    
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Generic chat method for LLM interactions with caching."""
        # Create cache key from input parameters including prompt version
        cache_key = hashlib.md5(f"{system}|{user}|{temperature}|{max_tokens}|{PLANNER_PROMPT_VERSION}".encode()).hexdigest()
        
        # Check cache first
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for LLM request: {cache_key[:CacheConstants.CACHE_KEY_LENGTH]}...")
            return cached_response
        
        # Retry logic with exponential backoff
        max_retries = ErrorConstants.MAX_RETRY_ATTEMPTS
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=ErrorConstants.REQUEST_TIMEOUT  # Increased timeout for larger batches
                )
                result = response.choices[0].message.content.strip()
                
                # Cache the response for configured TTL
                self.cache.set(cache_key, result, expire=3600*CacheConstants.CACHE_TTL_HOURS)
                logger.debug(f"Cached LLM response: {cache_key[:CacheConstants.CACHE_KEY_LENGTH]}...")
                
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = ErrorConstants.RETRY_BASE_DELAY ** attempt  # Exponential backoff
                    logger.warning(f"Chat attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    logger.error(f"Chat failed after {max_retries} attempts: {e}")
                    raise
    
    def plan_reddit_search(self, query: str, max_subreddits: int = 4) -> dict:
        """Plan Reddit search strategy using LLM."""
        try:
            sys = "You output strict JSON to plan Reddit searches."
            user = f"Query: {query}\n\n{SEARCH_PLANNER_PROMPT}"
            resp = self.chat(sys, user, temperature=0.2, max_tokens=400)
            plan = _safe_json_loads(resp)

            # Defensive clamps with intent-aware defaults
            plan.setdefault("terms", [query])
            plan["terms"] = list(dict.fromkeys([t for t in plan["terms"] if isinstance(t, str) and t.strip()]))[:SearchConstants.MAX_SEARCH_TERMS]  # Comprehensive search: up to 10 terms

            subs = [s.replace("r/","").strip() for s in plan.get("subreddits", []) if isinstance(s, str)]
            # Remove "all" subreddit to avoid irrelevant content - prioritize specific subreddits
            subs = [s for s in subs if s.lower() != "all"]
            
            # Expand subreddit list if it's too short to meet user's preference
            if len(subs) < max_subreddits:
                # Add common general subreddits based on query type
                query_lower = query.lower()
                additional_subs = []
                
                if any(word in query_lower for word in ['restaurant', 'food', 'dining', 'eat']):
                    additional_subs = ['food', 'restaurants', 'AskReddit', 'cooking', 'FoodPorn']
                elif any(word in query_lower for word in ['iphone', 'apple', 'phone']):
                    additional_subs = ['technology', 'Apple', 'smartphones', 'AskReddit', 'gadgets']
                elif any(word in query_lower for word in ['tesla', 'car', 'vehicle', 'electric']):
                    additional_subs = ['cars', 'ElectricVehicles', 'TeslaMotors', 'AskReddit', 'technology']
                elif any(word in query_lower for word in ['movie', 'film', 'cinema']):
                    additional_subs = ['movies', 'films', 'AskReddit', 'entertainment', 'NetflixBestOf']
                elif any(word in query_lower for word in ['camera', 'photography', 'photo', 'lens']):
                    additional_subs = ['photography', 'cameras', 'CameraGear', 'AskPhotography', 'PhotoCritique']
                elif any(word in query_lower for word in ['golf', 'course', 'club']):
                    additional_subs = ['golf', 'golfcoursereview', 'AskReddit', 'sports']
                else:
                    additional_subs = ['AskReddit']
                
                # Add additional subreddits up to max_subreddits
                for sub in additional_subs:
                    if len(subs) >= max_subreddits:
                        break
                    if sub.lower() not in [s.lower() for s in subs]:
                        subs.append(sub)
            
            plan["subreddits"] = subs[:max_subreddits]  # User-configurable subreddit count

            # Intent-aware defaults
            plan["time_filter"] = plan.get("time_filter") or "month"
            plan["strategies"] = [s for s in plan.get("strategies", ["relevance","top"]) if s in ("relevance","top","new")] or ["relevance","top"]  # Comprehensive search: 2 strategies
            plan["min_comment_score"] = max(0, int(plan.get("min_comment_score", 1)))  # Default configuration: score 1
            plan["per_post_top_n"] = min(12, max(3, int(plan.get("per_post_top_n", 8))))  # Default configuration: 8 comments per post

            pats = plan.get("comment_must_patterns") or []
            plan["comment_must_patterns"] = [p for p in pats if isinstance(p, str) and p.strip()][:SearchConstants.MAX_COMMENT_PATTERNS]  # Comprehensive search: up to 6 patterns
            return plan
        except Exception as e:
            logger.error(f"Reddit search planning failed: {e}")
            # Return a safe fallback plan with intent-aware defaults
            return {
                "terms": [query],
                "subreddits": ["AskReddit"],  # More focused than "all"
                "time_filter": "month",  # More balanced default
                "strategies": ["relevance", "top"],
                "min_comment_score": 3,  # Higher quality default
                "per_post_top_n": 6,  # Balanced default
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
    
    def generate_aspects_for_query(self, query: str, sample_comments: list) -> dict:
        """Generate relevant aspects for a query using LLM analysis of sample comments."""
        try:
            context_comments = sample_comments[:5] if sample_comments else []
            context_text = "\n".join([
                f"- {comment.get('text', '')[:200] if isinstance(comment, dict) else comment.text[:200]}"
                for comment in context_comments
            ])
            
            prompt = f"""
            Analyze the query "{query}" and the sample comments below to generate the most relevant aspects for this topic.
            
            Sample comments:
            {context_text}
            
            Return ONLY JSON with this exact schema:
            {{
                "aspects": [
                    {{
                        "name": "aspect_name",
                        "keywords": ["keyword1", "keyword2", "keyword3"],
                        "description": "brief description of what this aspect covers"
                    }}
                ]
            }}
            
            Rules:
            - Generate 6-8 relevant aspects based on the query topic and comment content
            - Use specific, actionable aspect names (e.g., "Course Quality", "Battery Life", "Customer Service")
            - Include 3-5 relevant keywords for each aspect that would appear in reviews
            - Focus on aspects that users actually discuss in reviews
            - Make aspects specific to the query domain (golf courses, tech products, restaurants, etc.)
            - Avoid generic aspects unless they're truly relevant
            """
            
            response = self.chat(
                system="You are an expert at analyzing user reviews and identifying key aspects that matter to consumers.",
                user=prompt,
                temperature=0.0,
                max_tokens=800
            )
            
            # Robust JSON parsing
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                result = json.loads(response)
            
            aspects = result.get("aspects", [])
            validated_aspects = {}
            for aspect in aspects:
                name = aspect.get("name", "").strip()
                keywords = aspect.get("keywords", [])
                if name and keywords:
                    clean_keywords = [k.strip().lower() for k in keywords if k.strip()]
                    if clean_keywords:
                        validated_aspects[name] = clean_keywords
            
            logger.info(f"Successfully generated {len(validated_aspects)} aspects")
            return validated_aspects
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM aspect generation response: {e}")
            logger.warning(f"Response was: {response[:200]}...")
            return {}
                
        except Exception as e:
            logger.error(f"LLM aspect generation failed: {e}")
            return {}
    
    def detect_intent_and_schema(self, query: str) -> IntentSchema:
        """Detect query intent and generate aspect schema."""
        try:
            prompt = f"""
            Analyze the query "{query}" and determine the user's intent and generate relevant aspects.
            
            Return ONLY JSON with this exact schema:
            {{
                "intent": "RANKING|SOLUTION|GENERIC",
                "entity_type": "entity_type_for_ranking_or_null",
                "aspects": ["aspect1", "aspect2", "aspect3"]
            }}
            
            Intent Rules:
            - RANKING: User wants to compare/rank specific entities (e.g., "best golf courses", "top restaurants", "iPhone vs Samsung")
            - SOLUTION: User wants solutions/fixes/how-to (e.g., "fix wind noise", "how to improve battery", "troubleshooting")
            - GENERIC: General product/service review analysis (e.g., "iPhone 15", "Tesla Model Y", "restaurant reviews")
            
            Entity Type Rules (for RANKING queries):
            - For restaurant queries: use "restaurant" (not "locations")
            - For product queries: use "product" or specific type like "phone", "car", "movie"
            - For service queries: use "service" or specific type like "hotel", "gym"
            - For location queries: use "location" or specific type like "city", "neighborhood"
            - Examples:
              * "best restaurant in NYC" → entity_type: "restaurant"
              * "best iPhone" → entity_type: "phone" 
              * "best hotel in Paris" → entity_type: "hotel"
              * "best neighborhoods in SF" → entity_type: "neighborhood"
            
            Aspect Rules:
            - Generate 4-8 relevant aspects based on the query domain
            - Use specific, actionable aspect names
            - Focus on aspects users actually discuss in reviews
            - Make aspects specific to the query domain
            """
            
            response = self.chat(
                system="You are an expert at analyzing user queries and determining their intent for review analysis.",
                user=prompt,
                temperature=0.2,
                max_tokens=600
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                result = json.loads(response)
            
            intent = result.get("intent", "GENERIC")
            entity_type = result.get("entity_type")
            aspects = result.get("aspects", [])
            
            if intent not in ["RANKING", "SOLUTION", "GENERIC"]:
                intent = "GENERIC"
            
            clean_aspects = [a.strip() for a in aspects if a.strip()][:8]
            
            # Fix common entity_type mismatches
            if "restaurant" in query.lower() and entity_type == "locations":
                entity_type = "restaurant"
            elif "hotel" in query.lower() and entity_type == "locations":
                entity_type = "hotel"
            elif "gym" in query.lower() and entity_type == "services":
                entity_type = "gym"
            
            logger.info(f"Detected intent: {intent}, entity_type: {entity_type}, aspects: {len(clean_aspects)}")
            return IntentSchema(intent=intent, entity_type=entity_type, aspects=clean_aspects)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse intent detection response: {e}")
            fallback_aspects = ["Quality", "Performance", "Value", "User Experience"]
            return IntentSchema(intent="GENERIC", entity_type=None, aspects=fallback_aspects)
                
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            fallback_aspects = ["Quality", "Performance", "Value", "User Experience"]
            return IntentSchema(intent="GENERIC", entity_type=None, aspects=fallback_aspects)
    
    # OLD METHOD REMOVED - using new GPT-only pipeline method below
    def _old_annotate_comments_with_gpt(self, query: str, aspects: List[str], comments: List[Dict], entity_type: str = None) -> List[GPTCommentAnno]:
        """Annotate comments with GPT for scoring and entity extraction."""
        annotations = []
        
        # Process in batches of 40 comments
        batch_size = 40
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            batch_annotations = self._annotate_batch(query, aspects, batch, entity_type)
            annotations.extend(batch_annotations)
        
        return annotations
    
    def _annotate_batch(self, query: str, aspects: List[str], comments: List[Dict], entity_type: str = None) -> List[GPTCommentAnno]:
        """Annotate a batch of comments."""
        try:
            # Prepare comments for annotation
            comments_text = []
            for comment in comments:
                text = comment.get("text", "")[:500]  # Limit text length
                comments_text.append(f"ID: {comment.get('id', '')}\nText: {text}")
            
            comments_str = "\n\n".join(comments_text)
            
            entity_type_instruction = ""
            if entity_type:
                entity_type_instruction = f"\nIMPORTANT: For entity extraction, use entity_type = \"{entity_type}\" for all relevant entities."
            
            prompt = f"""
            Analyze these Reddit comments about "{query}" and provide detailed annotations.
            
            Comments:
            {comments_str}
            
            Return ONLY JSON with this exact schema:
            {{
                "annotations": [
                    {{
                        "id": "comment_id",
                        "overall_stars": 3.5,
                        "aspects": {{"aspect_name": 4.0, "another_aspect": 2.5}},
                        "entities": [
                            {{"name": "entity_name", "entity_type": "type", "confidence": 0.8}}
                        ],
                        "cluster_key": "solution_cluster_key_or_null"
                    }}
                ]
            }}
            
            Rules:
            - Rate overall_stars from 1.0 to 5.0 based on sentiment
            - Rate each aspect from 1.0 to 5.0 based on how well the comment addresses it
            - Extract entities mentioned (products, places, people, etc.) with confidence 0.0-1.0
            - For SOLUTION queries, provide cluster_key for grouping similar solutions
            - For RANKING queries, extract entities with the specific entity_type requested
            - For GENERIC queries, cluster_key should be null{entity_type_instruction}
            """
            
            response = self.chat(
                system="You are an expert at analyzing user comments and extracting structured information.",
                user=prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                result = json.loads(response)
            
            annotations = []
            for anno_data in result.get("annotations", []):
                try:
                    entities = []
                    for entity_data in anno_data.get("entities", []):
                        entity = EntityRef(
                            name=entity_data.get("name", ""),
                            entity_type=entity_data.get("entity_type", ""),
                            mentions=1,
                            confidence=entity_data.get("confidence", 0.5)
                        )
                        entities.append(entity)
                    
                    annotation = GPTCommentAnno(
                        id=anno_data.get("id", ""),
                        overall_stars=float(anno_data.get("overall_stars", 3.0)),
                        aspects=anno_data.get("aspects", {}),
                        entities=entities,
                        cluster_key=anno_data.get("cluster_key")
                    )
                    annotations.append(annotation)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse annotation: {e}")
                    continue
            
            logger.info(f"Successfully annotated {len(annotations)} comments")
            return annotations
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse annotation response: {e}")
            return []
                
        except Exception as e:
            logger.error(f"Comment annotation failed: {e}")
            return []
    
    def summarize_ranking_with_gpt(self, query: str, ranking_items: List[Dict]) -> str:
        """Generate ranking summary focused on top entities."""
        try:
            if not ranking_items:
                return f"No ranked entities found for {query}. Try adjusting your search terms or filters."
            
            # Prepare ranking data
            top_items = ranking_items[:SearchConstants.MAX_ENTITIES_FOR_SUMMARY]  # Top entities for comprehensive ranking
            ranking_text = "\n".join([
                f"{i+1}. **{item['name']}** - {item['overall_stars']:.1f}/5 stars ({item['mentions']} mentions)"
                for i, item in enumerate(top_items)
            ])
            
            # Get quotes from top entities for detailed insights
            quotes_text = ""
            for item in top_items[:SearchConstants.MAX_ENTITIES_FOR_QUOTES]:
                if item.get('quotes'):
                    quotes_text += f"\n**{item['name']}:**\n"
                    for quote in item['quotes'][:2]:
                        quotes_text += f"- \"{quote[:150]}...\"\n"
            
            prompt = f"""
            Generate a ranking summary for "{query}" based on the analysis results.
            
            Top Ranked Entities:
            {ranking_text}
            
            Key Insights:
            {quotes_text}
            
            Provide a summary that includes:
            1. Brief overview of the ranking results
            2. Highlight the top 3 entities with specific details
            3. Mention key insights from user reviews
            4. Provide a concise recommendation
            
            Write in a natural, engaging style focused on the ranking results. Do NOT use pros/cons format.
            """
            
            response = self.chat(
                system="You are an expert reviewer who creates clear, engaging ranking summaries.",
                user=prompt,
                temperature=0.4,
                max_tokens=600
            )
            
            return response.strip()
                
        except Exception as e:
            logger.error(f"Ranking summarization failed: {e}")
            return f"Ranking analysis completed for {query}. Found {len(ranking_items)} ranked entities."

    def summarize_generic_with_gpt(self, query: str, aspects: Dict[str, float], overall: float, quotes: List[str]) -> str:
        """Generate detailed summary by analyzing review content."""
        try:
            aspects_text = "\n".join([f"- {name}: {score:.1f}/5" for name, score in aspects.items()])
            quotes_text = "\n".join([f"- \"{quote[:150]}...\"" for quote in quotes[:6]])  # Longer quotes for better analysis
            
            prompt = f"""
            Analyze these Reddit reviews for "{query}" and create a detailed, insightful summary.
            
            Overall Rating: {overall:.1f}/5
            
            Aspect Analysis:
            {aspects_text}
            
            Key User Reviews:
            {quotes_text}
            
            Create a comprehensive summary that:
            1. **Overall Assessment**: What's the general consensus about {query}?
            2. **Detailed Analysis**: Analyze the actual review content - what specific experiences do users mention?
            3. **Strengths**: What do users consistently praise? Be specific about features/benefits mentioned.
            4. **Concerns**: What issues do users repeatedly mention? Focus on real user experiences.
            5. **Context & Insights**: Any interesting patterns, comparisons, or insights from the reviews?
            6. **Recommendation**: Who should consider this and why? Based on actual user feedback.
            
            Write in an engaging, analytical style that captures the real user sentiment and experiences. 
            Focus on what users actually say, not just the scores.
            """
            
            response = self.chat(
                system="You are an expert analyst who creates detailed, insightful reviews by analyzing real user experiences and sentiment patterns.",
                user=prompt,
                temperature=0.4,
                max_tokens=800
            )
            
            return response.strip()
                
        except Exception as e:
            logger.error(f"Generic summarization failed: {e}")
            return f"Analysis completed for {query}. Overall rating: {overall:.1f}/5. See detailed aspect scores and quotes for more information."
    
    def summarize_solutions_with_gpt(self, query: str, clusters: List[Dict]) -> str:
        """Generate solution summary with clusters."""
        try:
            clusters_text = ""
            for i, cluster in enumerate(clusters, 1):
                cluster_text = f"""
                Solution {i}: {cluster.get('title', 'Untitled Solution')}
                Steps: {cluster.get('steps', [])}
                Caveats: {cluster.get('caveats', [])}
                Evidence: {cluster.get('evidence_count', 0)} comments
                """
                clusters_text += cluster_text + "\n"
            
            prompt = f"""
            Generate a comprehensive solution summary for "{query}" based on the solution clusters.
            
            Solution Clusters:
            {clusters_text}
            
            Provide a summary that includes:
            1. Overview of the problem and available solutions
            2. Summary of each solution cluster with key steps
            3. Important caveats and considerations
            4. Recommendations for which solutions to try first
            
            Write in a helpful, actionable style for users seeking solutions.
            """
            
            response = self.chat(
                system="You are an expert problem-solver who provides clear, actionable solutions.",
                user=prompt,
                temperature=0.4,
                max_tokens=1000
            )
            
            return response.strip()
                
        except Exception as e:
            logger.error(f"Solution summarization failed: {e}")
            return f"Found {len(clusters)} solution clusters for {query}. See detailed solutions for specific steps and caveats."

    # ===== GPT-ONLY PIPELINE METHODS =====
    
    def detect_intent_and_schema(self, query: str, sample_comments: List[Dict] = None) -> IntentSchema:
        """Detect query intent and generate relevant aspect schema."""
        try:
            # Prepare sample text for analysis
            sample_text = ""
            if sample_comments:
                sample_text = "\n".join([c.get("text", "")[:200] for c in sample_comments[:5]])
            
            prompt = f"""Analyze this query and determine its intent and relevant aspects:

Query: "{query}"

Sample comments (if available):
{sample_text}

Determine the INTENT:
- RANKING: User wants to compare/rank specific items (e.g., "best iPhone", "iPhone vs Samsung")
- SOLUTION: User wants solutions to a problem (e.g., "how to fix iPhone battery", "iPhone problems")
- GENERIC: General discussion/reviews (e.g., "iPhone reviews", "iPhone experience")

Generate relevant ASPECTS for this query (3-8 aspects):
- For tech products: performance, battery, camera, design, price, software, etc.
- For services: quality, customer service, value, reliability, etc.
- For locations: atmosphere, accessibility, value, quality, etc.

Return JSON:
{{
    "intent": "RANKING|SOLUTION|GENERIC",
    "aspects": ["aspect1", "aspect2", "aspect3"],
    "entity_type": "products|services|locations|etc" (for RANKING queries)
}}"""

            response = self.chat(
                system="You are an expert at analyzing user queries and determining intent and relevant aspects.",
                user=prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            try:
                result = _safe_json_loads(response)
            except Exception as e:
                logger.error(f"Intent detection JSON parsing failed: {e}")
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspect_names = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspect_names = ["quality", "value", "performance"]
                
                return IntentSchema(
                    intent="GENERIC",
                    aspects=aspect_names,
                    entity_type="products"
                )
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.error(f"Intent detection failed: Expected dict, got {type(result)}")
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspect_names = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspect_names = ["quality", "value", "performance"]
                
                return IntentSchema(
                    intent="GENERIC",
                    aspects=aspect_names,
                    entity_type="products"
                )
            
            # Validate and set defaults
            intent = result.get("intent", "GENERIC")
            if intent not in ["RANKING", "SOLUTION", "GENERIC"]:
                intent = "GENERIC"
            
            aspects = result.get("aspects", [])
            if not aspects:
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspects = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspects = ["quality", "value", "performance"]
            
            entity_type = result.get("entity_type", "products")
            
            # Fix common entity_type mismatches
            if "restaurant" in query.lower() and entity_type == "locations":
                entity_type = "restaurant"
            elif "hotel" in query.lower() and entity_type == "locations":
                entity_type = "hotel"
            elif "gym" in query.lower() and entity_type == "services":
                entity_type = "gym"
            
            return IntentSchema(
                intent=intent,
                aspects=aspects,
                entity_type=entity_type
            )
            
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return IntentSchema(
                intent="GENERIC",
                aspects=["quality", "value", "performance"],
                entity_type="products"
            )

    def annotate_comments_with_gpt(self, comments: List[Dict], aspects: List[str], entity_type: str = None, query: str = None) -> List[GPTCommentAnno]:
        """
        Annotate comments using GPT with caching and batch processing.
        
        This method implements a sophisticated batch processing pipeline:
        1. Process comments in optimal batch sizes to avoid token limits
        2. Generate comprehensive prompts with context and examples
        3. Extract structured data (sentiment, aspects, entities) from GPT responses
        4. Apply validation and fallback logic for failed batches
        5. Cache results for performance optimization
        
        Args:
            comments: List of comment objects or dictionaries
            aspects: List of aspect names to score
            entity_type: Expected entity type for extraction (e.g., "restaurant", "phone")
            query: Original user query for context-aware filtering
            
        Returns:
            List of GPTCommentAnno objects with extracted information:
            - overall_stars: 1-5 sentiment rating
            - aspect_scores: Dict of aspect -> score mappings
            - entities: List of extracted entities with confidence
        """
        import time
        start_time = time.time()
        annotations = []
        
        # Process in batches to avoid token limits and timeouts
        # Smaller batches = more reliable, larger batches = faster processing
        batch_size = SearchConstants.LLM_BATCH_SIZE  # Balanced batch size for speed and stability
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            
            try:
                # Prepare batch text
                batch_text = ""
                for j, comment in enumerate(batch):
                    # Handle both dict and object formats
                    if isinstance(comment, dict):
                        text = comment.get('text', '')
                    else:
                        text = getattr(comment, 'text', '')
                    # Truncate long comments to save tokens (keep first 300 chars)
                    truncated_text = text[:300] + "..." if len(text) > 300 else text
                    batch_text += f"Comment {j+1}: {truncated_text}\n\n"
                
                prompt = f"""Analyze these Reddit comments and provide structured annotations.

USER QUERY: "{query or 'General analysis'}"

Comments:
{batch_text}

Aspects to score: {', '.join(aspects)}

IMPORTANT FILTERING RULES:
- For queries with specific years (e.g., "2025", "2024"), ONLY extract entities that match that time period
  * If comment mentions "2020 movie" but query asks for "2025", set overall_score=1 and extract NO entities
  * If comment mentions "Edge of Tomorrow (2014)" but query asks for "2025", set overall_score=1 and extract NO entities
- For queries with specific locations (e.g., "NYC", "San Francisco", "Bay Area", "Los Angeles"), ONLY extract entities in that location  
  * If comment mentions a different location than the query specifies, set overall_score=1 and extract NO entities
  * Examples: "Chicago restaurant" for "NYC" query, "Indiana golf" for "Bay Area" query, "NYC golf" for "Los Angeles" query
  * Use geographic reasoning: if query asks for location X, only extract entities in location X
- Only extract entities that are actually comparable and relevant to the query location
- CRITICAL: If content is from wrong time period or location, set overall_score=1 and entities=[]

For each comment, provide:
1. Overall star rating (1-5)
2. Per-aspect scores (1-5 for each aspect)
3. Entities mentioned (for ranking queries)
4. Solution cluster key (for solution queries)

Return JSON array:
[
    {{
        "overall_score": 4,
        "aspect_scores": {{"quality": 4, "value": 3, "performance": 5}},
        "entities": [{{"name": "iPhone 15", "type": "product", "confidence": 0.9}}],
        "solution_key": "battery_replacement"
    }}
]"""

                response = self.chat(
                    system="You are an expert at analyzing Reddit comments for sentiment, aspects, and entities.",
                    user=prompt,
                    temperature=0.2,
                    max_tokens=1500
                )
                
                batch_annotations = _safe_json_loads(response)
                
                # Convert to GPTCommentAnno objects
                for j, annotation_data in enumerate(batch_annotations):
                    if j < len(batch):
                        comment = batch[j]
                        
                        # Extract comment ID
                        if isinstance(comment, dict):
                            comment_id = comment.get("id", "")
                        else:
                            comment_id = getattr(comment, "id", "")
                        
                        # Extract entities
                        entities = []
                        for entity_data in annotation_data.get("entities", []):
                            entities.append(EntityRef(
                                name=entity_data.get("name", ""),
                                entity_type=entity_data.get("type", entity_type or "unknown"),
                                confidence=entity_data.get("confidence", 0.5)
                            ))
                        
                        annotations.append(GPTCommentAnno(
                            comment_id=comment_id,
                            overall_score=annotation_data.get("overall_score", 3),
                            aspect_scores=annotation_data.get("aspect_scores", {}),
                            entities=entities,
                            solution_key=annotation_data.get("solution_key", "")
                        ))
                
            except Exception as e:
                logger.error(f"GPT annotation batch failed: {e}")
                # Add default annotations for failed batch
                for comment in batch:
                    # Extract comment ID
                    if isinstance(comment, dict):
                        comment_id = comment.get("id", "")
                    else:
                        comment_id = getattr(comment, "id", "")
                    
                    annotations.append(GPTCommentAnno(
                        comment_id=comment_id,
                        overall_score=3,
                        aspect_scores={aspect: 3 for aspect in aspects},
                        entities=[],
                        solution_key=""
                    ))
        
        return annotations

    def generate_dynamic_aspects(self, query: str, sample_comments: List[Dict] = None) -> List[str]:
        """Generate dynamic aspects based on query and sample comments."""
        try:
            # Prepare sample text
            sample_text = ""
            if sample_comments:
                sample_text = "\n".join([c.get("text", "")[:200] for c in sample_comments[:3]])
            
            prompt = f"""Generate relevant aspects for analyzing this query:

Query: "{query}"

Sample comments:
{sample_text}

Generate 4-6 relevant aspects that people would care about for this topic.
Return as a JSON array: ["aspect1", "aspect2", "aspect3"]"""

            response = self.chat(
                system="You are an expert at identifying relevant aspects for any topic.",
                user=prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            aspects = _safe_json_loads(response)
            if isinstance(aspects, list) and all(isinstance(a, str) for a in aspects):
                return aspects[:6]  # Limit to 6 aspects
            
        except Exception as e:
            logger.error(f"Dynamic aspect generation failed: {e}")
        
        # Fallback to domain-specific aspects
        return get_domain_aspects(query)


class FallbackLLMService:
    """Fallback LLM service using simple rules."""
    
    def __init__(self):
        logger.info("Using fallback LLM service")
    
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Fallback chat method - returns empty string."""
        logger.warning("Fallback LLM service chat called - no actual LLM available")
        return ""
    
    def plan_reddit_search(self, query: str) -> dict:
        """Fallback search planning with intent awareness."""
        logger.warning("Using fallback Reddit search planning")
        
        # Intent detection for better defaults
        query_lower = query.lower()
        if any(word in query_lower for word in ["best", "top", "which", "where"]):
            # RANKING intent
            time_filter = "year"
            min_score = 5
            per_post = 6
        elif any(word in query_lower for word in ["how to", "fix", "solution", "workaround"]):
            # SOLUTION intent
            time_filter = "month"
            min_score = 2
            per_post = 8
        else:
            # GENERIC intent
            time_filter = "month"
            min_score = 3
            per_post = 6
        
        return {
            "terms": [query],
            "subreddits": ["all"],
            "time_filter": time_filter,
            "strategies": ["relevance", "top"],
            "min_comment_score": min_score,
            "per_post_top_n": per_post,
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

    # ===== GPT-ONLY PIPELINE METHODS (FALLBACK VERSIONS) =====
    
    def detect_intent_and_schema(self, query: str, sample_comments: List[Dict] = None) -> IntentSchema:
        """Fallback intent detection - returns GENERIC with basic aspects."""
        logger.warning("Fallback LLM service intent detection called - using basic fallback")
        return IntentSchema(
            intent="GENERIC",
            aspects=["quality", "value", "performance"],
            entity_type="products"
        )

    def annotate_comments_with_gpt(self, comments: List[Dict], aspects: List[str], entity_type: str = None, query: str = None) -> List[GPTCommentAnno]:
        """Fallback comment annotation - returns dummy annotations."""
        logger.warning("Fallback LLM service comment annotation called - using dummy annotations")
        annotations = []
        for comment in comments:
            # Extract comment ID safely
            if isinstance(comment, dict):
                comment_id = comment.get("id", "")
            else:
                comment_id = getattr(comment, "id", "")
            
            annotations.append(GPTCommentAnno(
                comment_id=comment_id,
                overall_score=3.0,
                aspect_scores={aspect: 3.0 for aspect in aspects},
                entities=[],
                solution_key=""
            ))
        return annotations

    def generate_dynamic_aspects(self, query: str, sample_comments: List[Dict] = None) -> List[str]:
        """Fallback dynamic aspect generation - uses domain-specific aspects."""
        logger.warning("Fallback LLM service dynamic aspect generation called - using domain aspects")
        return get_domain_aspects(query)

    # ===== GPT-ONLY PIPELINE METHODS =====
    
    def detect_intent_and_schema(self, query: str, sample_comments: List[Dict] = None) -> IntentSchema:
        """Detect query intent and generate relevant aspect schema."""
        try:
            # Prepare sample text for analysis
            sample_text = ""
            if sample_comments:
                sample_text = "\n".join([c.get("text", "")[:200] for c in sample_comments[:5]])
            
            prompt = f"""Analyze this query and determine its intent and relevant aspects:

Query: "{query}"

Sample comments (if available):
{sample_text}

Determine the INTENT:
- RANKING: User wants to compare/rank specific items (e.g., "best iPhone", "iPhone vs Samsung")
- SOLUTION: User wants solutions to a problem (e.g., "how to fix iPhone battery", "iPhone problems")
- GENERIC: General discussion/reviews (e.g., "iPhone reviews", "iPhone experience")

Generate relevant ASPECTS for this query (3-8 aspects):
- For tech products: performance, battery, camera, design, price, software, etc.
- For services: quality, customer service, value, reliability, etc.
- For locations: atmosphere, accessibility, value, quality, etc.

Return JSON:
{{
    "intent": "RANKING|SOLUTION|GENERIC",
    "aspects": ["aspect1", "aspect2", "aspect3"],
    "entity_type": "products|services|locations|etc" (for RANKING queries)
}}"""

            response = self.chat(
                system="You are an expert at analyzing user queries and determining intent and relevant aspects.",
                user=prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            try:
                result = _safe_json_loads(response)
            except Exception as e:
                logger.error(f"Intent detection JSON parsing failed: {e}")
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspect_names = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspect_names = ["quality", "value", "performance"]
                
                return IntentSchema(
                    intent="GENERIC",
                    aspects=aspect_names,
                    entity_type="products"
                )
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.error(f"Intent detection failed: Expected dict, got {type(result)}")
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspect_names = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspect_names = ["quality", "value", "performance"]
                
                return IntentSchema(
                    intent="GENERIC",
                    aspects=aspect_names,
                    entity_type="products"
                )
            
            # Validate and set defaults
            intent = result.get("intent", "GENERIC")
            if intent not in ["RANKING", "SOLUTION", "GENERIC"]:
                intent = "GENERIC"
            
            aspects = result.get("aspects", [])
            if not aspects:
                # Generate dynamic aspects instead of hardcoded fallback
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspects = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspects = ["quality", "value", "performance"]
            
            entity_type = result.get("entity_type", "products")
            
            # Fix common entity_type mismatches
            if "restaurant" in query.lower() and entity_type == "locations":
                entity_type = "restaurant"
            elif "hotel" in query.lower() and entity_type == "locations":
                entity_type = "hotel"
            elif "gym" in query.lower() and entity_type == "services":
                entity_type = "gym"
            
            return IntentSchema(
                intent=intent,
                aspects=aspects,
                entity_type=entity_type
            )
            
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return IntentSchema(
                intent="GENERIC",
                aspects=["quality", "value", "performance"],
                entity_type="products"
            )

    def annotate_comments_with_gpt(self, comments: List[Dict], aspects: List[str], entity_type: str = None, query: str = None) -> List[GPTCommentAnno]:
        """Annotate comments with GPT for scoring and entity extraction."""
        annotations = []
        
        # Smart comment selection: prioritize high-quality comments
        if len(comments) > 30:
            # Sort by upvotes and take top 30 most relevant comments
            sorted_comments = sorted(comments, key=lambda c: c.get('upvotes', 0) if isinstance(c, dict) else getattr(c, 'upvotes', 0), reverse=True)
            comments = sorted_comments[:30]
            logger.info(f"Selected top 30 comments from {len(sorted_comments)} total comments")
        
        # Process comments for annotation
        logger.info(f"annotate_comments_with_gpt called with {len(comments)} comments")
        if comments:
            logger.info(f"First comment type: {type(comments[0])}, content: {comments[0]}")
        
        # Adaptive batch sizing: start with 8, reduce if timeouts occur
        batch_size = 8
        timeout_count = 0
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            
            try:
                # Prepare batch text
                batch_text = ""
                for j, comment in enumerate(batch):
                    # Handle both dict and object formats
                    if isinstance(comment, dict):
                        text = comment.get('text', '')
                    else:
                        text = getattr(comment, 'text', '')
                    # Truncate long comments to save tokens (keep first 300 chars)
                    truncated_text = text[:300] + "..." if len(text) > 300 else text
                    batch_text += f"Comment {j+1}: {truncated_text}\n\n"
                
                prompt = f"""Analyze these Reddit comments and provide structured annotations.

USER QUERY: "{query or 'General analysis'}"

Comments:
{batch_text}

Aspects to score: {', '.join(aspects)}

IMPORTANT FILTERING RULES:
- For queries with specific years (e.g., "2025", "2024"), ONLY extract entities that match that time period
  * If comment mentions "2020 movie" but query asks for "2025", set overall_score=1 and extract NO entities
  * If comment mentions "Edge of Tomorrow (2014)" but query asks for "2025", set overall_score=1 and extract NO entities
- For queries with specific locations (e.g., "NYC", "San Francisco", "Bay Area", "Los Angeles"), ONLY extract entities in that location  
  * If comment mentions a different location than the query specifies, set overall_score=1 and extract NO entities
  * Examples: "Chicago restaurant" for "NYC" query, "Indiana golf" for "Bay Area" query, "NYC golf" for "Los Angeles" query
  * Use geographic reasoning: if query asks for location X, only extract entities in location X
- Only extract entities that are actually comparable and relevant to the query location
- CRITICAL: If content is from wrong time period or location, set overall_score=1 and entities=[]

For each comment, provide:
1. Overall star rating (1-5)
2. Per-aspect scores (1-5 for each aspect)
3. Entities mentioned (for ranking queries)
4. Solution cluster key (for solution queries)

Return JSON array:
[
    {{
        "overall_score": 4,
        "aspect_scores": {{"quality": 4, "value": 3, "performance": 5}},
        "entities": [{{"name": "iPhone 15", "type": "product", "confidence": 0.9}}],
        "solution_key": "battery_replacement"
    }}
]"""

                response = self.chat(
                    system="You are an expert at analyzing Reddit comments for sentiment, aspects, and entities.",
                    user=prompt,
                    temperature=0.2,
                    max_tokens=1500
                )
                
                batch_annotations = _safe_json_loads(response)
                
                # Convert to GPTCommentAnno objects
                for j, annotation_data in enumerate(batch_annotations):
                    if j < len(batch):
                        comment = batch[j]
                        
                        # Extract comment ID
                        if isinstance(comment, dict):
                            comment_id = comment.get("id", "")
                        else:
                            comment_id = getattr(comment, "id", "")
                        
                        # Extract entities
                        entities = []
                        for entity_data in annotation_data.get("entities", []):
                            entities.append(EntityRef(
                                name=entity_data.get("name", ""),
                                entity_type=entity_data.get("type", entity_type or "unknown"),
                                confidence=entity_data.get("confidence", 0.5)
                            ))
                        
                        annotations.append(GPTCommentAnno(
                            comment_id=comment_id,
                            overall_score=annotation_data.get("overall_score", 3),
                            aspect_scores=annotation_data.get("aspect_scores", {}),
                            entities=entities,
                            solution_key=annotation_data.get("solution_key", "")
                        ))
                
            except Exception as e:
                logger.error(f"GPT annotation batch failed: {e}")
                # Add default annotations for failed batch
                for comment in batch:
                    # Extract comment ID
                    if isinstance(comment, dict):
                        comment_id = comment.get("id", "")
                    else:
                        comment_id = getattr(comment, "id", "")
                    
                    annotations.append(GPTCommentAnno(
                        comment_id=comment_id,
                        overall_score=3,
                        aspect_scores={aspect: 3 for aspect in aspects},
                        entities=[],
                        solution_key=""
                    ))
        
        return annotations

    def generate_dynamic_aspects(self, query: str, sample_comments: List[Dict] = None) -> List[str]:
        """Generate dynamic aspects based on query and sample comments."""
        try:
            # Prepare sample text
            sample_text = ""
            if sample_comments:
                sample_text = "\n".join([c.get("text", "")[:200] for c in sample_comments[:3]])
            
            prompt = f"""Generate relevant aspects for analyzing this query:

Query: "{query}"

Sample comments:
{sample_text}

Generate 4-6 relevant aspects that people would care about for this topic.
Return as a JSON array: ["aspect1", "aspect2", "aspect3"]"""

            response = self.chat(
                system="You are an expert at identifying relevant aspects for any topic.",
                user=prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            aspects = _safe_json_loads(response)
            if isinstance(aspects, list) and all(isinstance(a, str) for a in aspects):
                return aspects[:6]  # Limit to 6 aspects
            
        except Exception as e:
            logger.error(f"Dynamic aspect generation failed: {e}")
        
        # Fallback to domain-specific aspects
        return get_domain_aspects(query)

