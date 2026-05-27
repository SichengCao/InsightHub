"""LLM service for OpenAI integration."""

import logging
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Example: "NYC restaurants" -> ['AskNYC', 'NewYorkCity', 'FoodNYC'] NOT ['food', 'restaurants']
  * For product queries: prioritize BRAND-SPECIFIC subreddits over general category subreddits  
    Example: "iPhone 15" -> ['iPhone', 'Apple'] NOT ['smartphones', 'technology']
  * For services: prioritize SERVICE-SPECIFIC subreddits
  * NEVER include "all" subreddit - it returns too much irrelevant content
  * Prefer active, relevant subreddits with 10K+ members
  * Choose the MOST relevant subreddits - prioritize quality over quantity
  * For NYC queries: start with ['AskNYC', 'NewYorkCity', 'FoodNYC'] and add more relevant subreddits as needed
- All arrays MUST be case-insensitively deduped and length-bounded per spec.
- Never include "r/" prefixes in subreddit names.
- For comment_must_patterns: Always return an empty list []. GPT sentiment analysis will handle comment quality filtering.

Also infer an internal intent to guide your choices:
- If the query is a "best/top/which/where" compare: treat as RANKING.
- If "how to/fix/solution/workaround": treat as SOLUTION.
- Otherwise: GENERIC.

Produce JSON with exactly these keys:
- "terms": 2-4 short search strings (must include the exact raw query once).
  * Make terms SPECIFIC and TARGETED to avoid irrelevant results
  * For location queries: include specific place names, avoid generic terms like "best food"
  * For product queries: include specific model names, avoid generic category terms
  * PRIORITIZE specificity over recall - better to get fewer, more relevant results
- "subreddits": 2-4 names (no "r/" prefix). AVOID "all" unless absolutely necessary. PRIORITIZE the most relevant and active subreddits.
- "time_filter": one of ["day","week","month","year","all"].
- "strategies": 1-3 from ["relevance","top","new"].
- "min_comment_score": integer 0..50.
- "per_post_top_n": integer 3..8.
- "comment_must_patterns": Always empty list [].

Heuristics by inferred intent:
- RANKING -> strategies: ["top","relevance"]; time_filter: "all" (recommendations are evergreen -- never use "week" or "month"); per_post_top_n: 5-8; min_comment_score: 3-5 (higher for quality).
- SOLUTION -> strategies: ["new","relevance"]; time_filter: "week" or "month"; per_post_top_n: 6-10; min_comment_score: 0-2.
- GENERIC -> strategies: ["relevance","top"]; time_filter: "month" or "year"; per_post_top_n: 5-8; min_comment_score: 2-4 (higher for quality).

Output strict JSON only. No comments, no trailing commas.
""").strip()

def _safe_json_loads(s: str):
    """Safely parse JSON from LLM response, handling common formatting issues."""
    # First, strip markdown code blocks if present
    cleaned = s.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Try direct parsing on cleaned string
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to extract JSON array
        array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except:
                pass
        
        # Try to extract JSON object
        obj_match = re.search(r"\{.*\}", cleaned, re.S)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except:
                pass
        
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
- Keep the 6-10 best quotes maximizing coverage and variety.
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
    
    def filter_entities_by_type(self, names: list, entity_type: str) -> list:
        """Return only names whose primary identity matches entity_type, via a single GPT call."""
        if not names or not entity_type:
            return names
        label = entity_type.replace("_", " ")
        try:
            verdict = self.chat(
                system="You are a fact-checker. Answer ONLY with a JSON array of booleans.",
                user=(
                    f"For each name below, return true if it is primarily known as a {label} "
                    f"and false if it is not. Use your world knowledge, not just the name itself.\n"
                    f"Names: {names}\nReturn ONLY a JSON array like [true, false, true, ...]"
                ),
                temperature=0.0,
                max_tokens=len(names) * 10 + 20,
            )
            import re as _re, json as _json
            flags = _json.loads(_re.search(r'\[.*\]', verdict, _re.S).group(0))
            kept = [n for n, ok in zip(names, flags) if ok]
            logger.info(f"Entity type filter: {len(kept)}/{len(names)} kept for {entity_type}")
            return kept
        except Exception as e:
            logger.warning(f"Entity type filter failed ({e}), returning all names")
            return names

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
            subs = [s for s in subs if s.lower() != "all"]
            plan["subreddits"] = subs[:max_subreddits]

            # Intent-aware defaults
            plan["time_filter"] = plan.get("time_filter") or "month"
            plan["strategies"] = [s for s in plan.get("strategies", ["relevance","top"]) if s in ("relevance","top","new")] or ["relevance","top"]  # Comprehensive search: 2 strategies
            plan["min_comment_score"] = max(0, int(plan.get("min_comment_score", 1)))  # Default configuration: score 1
            plan["per_post_top_n"] = min(12, max(3, int(plan.get("per_post_top_n", 8))))  # Default configuration: 8 comments per post

            # Always use empty patterns - GPT sentiment analysis handles quality filtering
            plan["comment_must_patterns"] = []
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
                "comment_must_patterns": []  # GPT sentiment analysis handles quality
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
            {concatenated_text[:8000]}
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
    
    def summarize_ranking_with_gpt(self, query: str, ranking_items: List[Dict]) -> str:
        """Generate ranking summary focused on top entities with deduplication."""
        try:
            if not ranking_items:
                return f"No ranked entities found for {query}. Try adjusting your search terms or filters."
            
            # Let GPT handle deduplication and smart ranking
            entities_text = "\n".join([
                f"- **{item['name']}** - {item['overall_stars']:.1f}/5 stars ({item['mentions']} mentions)"
                for item in ranking_items[:15]  # Show more entities for better deduplication
            ])
            
            # Get quotes from all entities
            quotes_text = ""
            for item in ranking_items[:10]:
                if item.get('quotes'):
                    quotes_text += f"\n**{item['name']}**: {', '.join(item['quotes'][:2])}\n"
            
            prompt = f"""
            Generate a ranking summary for "{query}" based on the analysis results.
            
            Raw Entity Data (may contain duplicates and generic descriptions to filter):
            {entities_text}
            
            Key User Insights:
            {quotes_text}
            
            CRITICAL FILTERING RULES:
            1. **ONLY include specific named entities** - exclude generic descriptions that are not proper nouns, brand names, or specific identifiers that users could search for or purchase
            
            2. **Merge duplicate entities** - combine similar names and handle variations in spelling/formatting
            
            3. **Include ONLY concrete named entities** - focus on specific names that represent actual searchable/purchasable entities rather than descriptive phrases
            
            Provide a summary that includes:
            1. Brief overview of the ranking results (only specific named entities)
            2. Highlight the top 3-5 DISTINCT named entities with specific details
            3. Mention key insights from user reviews for top entities
            4. Provide a concise recommendation
            
            If no proper named entities are found, explain that the search needs more specific results.
            Write in a natural, engaging style. Only rank actual named entities.
            """
            
            response = self.chat(
                system="You are an expert reviewer who creates clear, engaging ranking summaries. CRITICAL: ONLY include specific named entities in rankings - actual proper nouns, brand names, model numbers, or specific business/location names that users can search for or purchase. DO NOT include generic descriptive phrases, qualifying adjectives, or category descriptions. Use your understanding to distinguish between concrete entities versus vague descriptors. Always filter out non-specific terms and focus on actual named entities.",
                user=prompt,
                temperature=0.3,  # Lower temperature for more consistent deduplication
                max_tokens=800
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
    
    def detect_intent_and_schema(self,query: str, sample_comments: List[Dict] = None) -> IntentSchema:
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
- For services/venues: quality, value, atmosphere, accessibility, conditions, etc.

For RANKING queries, set entity_type to the SPECIFIC venue/product type being compared.
NEVER use "locations", "location", "places", "area", "food", or bare cuisine names.
entity_type is the VENUE/SHOP TYPE, not the item served:
  * "best ramen in Tokyo"             -> entity_type: "ramen_restaurant"
  * "best Korean restaurant in NYC"   -> entity_type: "Korean_restaurant"
  * "best pizza in NYC"               -> entity_type: "pizza_restaurant"
  * "best coffee in Seattle"          -> entity_type: "coffee_shop"
  * "best golf course in bay area"    -> entity_type: "golf_course"
  * "best hotel in Paris"             -> entity_type: "hotel"
  * "best iPhone model"               -> entity_type: "phone"

Return JSON:
{{
    "intent": "RANKING|SOLUTION|GENERIC",
    "aspects": ["aspect1", "aspect2", "aspect3"],
    "entity_type": "specific_entity_type_or_null"
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
                    entity_type=None
                )

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.error(f"Intent detection failed: Expected dict, got {type(result)}")
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspect_names = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspect_names = ["quality", "value", "performance"]

                return IntentSchema(
                    intent="GENERIC",
                    aspects=aspect_names,
                    entity_type=None
                )

            # Validate and set defaults
            intent = result.get("intent", "GENERIC")
            if intent not in ["RANKING", "SOLUTION", "GENERIC"]:
                intent = "GENERIC"

            aspects = result.get("aspects", [])
            if not aspects:
                try:
                    dynamic_aspects = self.generate_dynamic_aspects(query)
                    aspects = list(dynamic_aspects.keys()) if dynamic_aspects else ["quality", "value", "performance"]
                except:
                    aspects = ["quality", "value", "performance"]

            entity_type = result.get("entity_type")

            # Null out values the prompt explicitly forbids — enforcement of prompt contract,
            # not domain logic. GPT was told: "NEVER use 'locations', 'location', 'places',
            # 'area', 'food', or bare cuisine names."
            _FORBIDDEN = frozenset({"locations", "location", "places", "place", "area", "areas", "food", "services"})
            if entity_type and entity_type.lower() in _FORBIDDEN:
                entity_type = None

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
                entity_type=None
            )

    def filter_relevant_comments(self, comments: List[Dict], query: str, threshold: float = 0.35) -> List[Dict]:
        """Drop comments that are not topically relevant to the query using GPT batch evaluation."""
        if not comments:
            return comments

        batch_size = 20
        batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]

        system_msg = (
            "You are a relevance evaluator. For each comment decide whether it contains direct, "
            "firsthand discussion of the topic in the user's query — specific opinions, experiences, "
            "or insights about the entities, venues, or products being asked about. "
            "Return ONLY a JSON array of objects, one per comment, in input order."
        )

        def _evaluate_batch(batch_idx: int, batch: list) -> tuple:
            batch_text = ""
            for j, comment in enumerate(batch):
                text = comment.get("text", "")
                truncated = text[:200] + "..." if len(text) > 200 else text
                batch_text += f"Comment {j + 1}: {truncated}\n\n"

            prompt = (
                f'User query: "{query}"\n\n'
                f"{batch_text}"
                f'For each comment return:\n'
                f'- relevant: true if the comment directly addresses the query topic, false if off-topic or generic\n'
                f'- relevance_score: 0.0 to 1.0\n'
                f'- reason: one short phrase\n\n'
                f'Return a JSON array [{{"relevant": bool, "relevance_score": float, "reason": str}}, ...] '
                f'with exactly {len(batch)} elements in input order.'
            )
            try:
                response = self.chat(system=system_msg, user=prompt, temperature=0.0, max_tokens=len(batch) * 35 + 20)
                results = _safe_json_loads(response)
                if isinstance(results, list) and len(results) == len(batch):
                    return batch_idx, results
            except Exception as e:
                logger.warning(f"Relevance filter batch {batch_idx} failed: {e}")
            # fail-open: keep all comments in this batch
            return batch_idx, [{"relevant": True, "relevance_score": 1.0}] * len(batch)

        results_by_idx: dict = {}
        max_workers = min(8, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate_batch, idx, batch): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                batch_idx, batch_results = future.result()
                results_by_idx[batch_idx] = batch_results

        kept, dropped = [], 0
        for idx, batch in enumerate(batches):
            verdicts = results_by_idx.get(idx, [{"relevant": True}] * len(batch))
            for comment, verdict in zip(batch, verdicts):
                if not isinstance(verdict, dict):
                    kept.append(comment)
                    continue
                score = verdict.get("relevance_score", 1.0)
                is_relevant = verdict.get("relevant", True)
                if is_relevant and score >= threshold:
                    kept.append(comment)
                else:
                    dropped += 1
                    logger.debug(f"Dropped irrelevant comment: {verdict.get('reason', '?')} | score={score:.2f}")

        logger.info(f"Relevance filter: {len(kept)}/{len(comments)} kept (dropped {dropped})")
        return kept

    def validate_entity_locations(self, entities: list, query: str) -> list:
        """Remove ranked entities whose geographic location does not match the query's location context."""
        if not entities:
            return entities
        names = [e.name for e in entities]
        try:
            verdict = self.chat(
                system="You are a geography fact-checker. Answer ONLY with a JSON array of booleans.",
                user=(
                    f'User query: "{query}"\n'
                    f"For each entity/venue below, return true if it is located in the geographic area "
                    f"specified by the query, and false if it is in a DIFFERENT location. "
                    f"If the query has no specific location, return true for all.\n"
                    f"Use your world knowledge, not just the name.\n"
                    f"Entities: {names}\n"
                    f"Return ONLY a JSON array like [true, false, true, ...]"
                ),
                temperature=0.0,
                max_tokens=len(names) * 10 + 30,
            )
            import re as _re, json as _json
            flags = _json.loads(_re.search(r'\[.*\]', verdict, _re.S).group(0))
            kept = [e for e, ok in zip(entities, flags) if ok]
            logger.info(f"Location filter: {len(kept)}/{len(entities)} entities kept for '{query}'")
            return kept
        except Exception as e:
            logger.warning(f"Location validation failed ({e}), returning all entities")
            return entities

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
        batch_size = SearchConstants.LLM_BATCH_SIZE
        batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]

        # Build the static parts of the prompt once (shared across all batches).
        entity_type_line = (
            f"\nEXTRACT ONLY: {entity_type} entities -- skip people, organizations, "
            f"and anything that is not a {entity_type}."
        ) if entity_type else ""
        aspects_str = ', '.join(aspects) if aspects else 'quality, value, experience'
        query_str = query or 'General analysis'
        entity_or_venue = entity_type or 'venue/product'

        system_msg = (
            "You are an expert at analyzing Reddit comments for sentiment, aspects, and entities. "
            "CRITICAL: (1) Assign each entity its OWN sentiment_score based on how the author feels "
            "about that specific entity. (2) Extract ONLY specific named venues, products, or businesses -- "
            "NEVER extract city names, regions, or geographic labels as entities. "
            "(3) ALL entity names must be in English or romanized Latin script -- never output Korean, "
            "Chinese, Japanese, Arabic, or any other non-Latin script characters in the 'name' field. "
            "Romanize if necessary. "
            "(4) Return ONLY valid JSON array, no markdown code blocks."
        )

        def _process_batch(batch_idx: int, batch: list) -> tuple:
            """Annotate one batch; returns (batch_idx, List[GPTCommentAnno])."""
            try:
                batch_text = ""
                for j, comment in enumerate(batch):
                    text = comment.get('text', '') if isinstance(comment, dict) else getattr(comment, 'text', '')
                    truncated = text[:150] + "..." if len(text) > 150 else text
                    batch_text += f"Comment {j+1}: {truncated}\n\n"

                prompt = f"""Analyze these Reddit comments and provide structured annotations.

USER QUERY: "{query_str}"{entity_type_line}

Comments:
{batch_text}

Aspects to score: {aspects_str}

ENTITY EXTRACTION RULES (critical for accuracy):
- Extract specific named VENUES, PRODUCTS, or BUSINESSES that the author directly experienced.
  These are proper nouns representing reviewable things: golf courses, restaurants, phones, hotels.
- If EXTRACT ONLY is specified above, only extract entities of that exact category.
  Skip people, instructors, media channels, brands, events, or anything not of that category.
  Example: EXTRACT ONLY golf_course -> extract "Harding Park" but NOT "Tom Hsieh" or "Fried Egg Golf".
  If the category includes a cuisine (e.g. Korean_restaurant), only extract restaurants whose
  PRIMARY identity, menu, and cuisine tradition match that cuisine. The restaurant must be
  recognizable as that type of restaurant by name or description -- serving a few dishes inspired
  by that cuisine is NOT enough. If you are uncertain, exclude it.
- NEVER extract geographic labels as entities: city abbreviations (LA, NY, NYC, SD, SF),
  metro area names, states, regions, or vague references like "Bay Area courses" or "SoCal spots".
  Geographic labels describe WHERE, not WHAT is being reviewed -- they are not entities.
- For comparison mentions (e.g. "better than Samsung"), extract the compared entity with
  is_primary=false and its own sentiment_score reflecting the author's view of it.
- For EACH entity provide a SEPARATE sentiment_score (1-5) based solely on how the author
  feels about THAT specific entity -- not the comment's overall tone.
  Example: "iPhone 15 blows away the Samsung S24" -> iPhone 15 sentiment=5, Samsung S24 sentiment=2
- Set is_primary=true for the entity that is the main subject/focus.
  Set is_primary=false for entities mentioned only as comparisons or context.
- Include mention_context: a verbatim excerpt (~10 chars) showing how the entity was mentioned.
- ALWAYS write entity names in English or romanized Latin script. If a name appears in a
  non-Latin script (Korean, Chinese, Japanese, Arabic, etc.), romanize it to its standard
  English transliteration (e.g., 기사식당 → "Kisa Sikdang", 돈부리 → "Donburi"). This is
  mandatory for consistent deduplication -- never output raw non-Latin characters in "name".
- The type field must be the specific entity category (e.g. "golf_course", "phone", "restaurant"),
  NEVER a geographic type like "location", "city", "region", "area", or "country".

SENTIMENT GROUNDING RULES:
- sentiment_score must reflect the author's direct experience with the entity's CORE quality.
  Golf course: conditions, layout, scenery, value, pace of play, facilities.
  Restaurant: food quality, service, atmosphere, price. Product: performance, build, features.
- Do NOT score based on peripheral details: logos, aesthetics, travel context, or any text
  that does not describe hands-on experience with the entity.
- If a comment only names an entity without reviewing it (e.g. lists it, mentions its logo,
  or references it geographically only), set confidence=0.4.

LOCATION / TIME FILTERING:
- For location-specific queries (e.g. "Bay Area golf courses"):
  * Only extract SPECIFIC NAMED VENUES confirmed to be IN that location.
  * If a venue is in a DIFFERENT location, exclude it entirely (entities=[]).
  * Geographic references used as comparisons ("better than LA courses", "beats NY") must NOT
    become entities -- skip them entirely, do not extract "LA" or "NY" as entity names.
- For year-specific queries, only extract entities from that period.
  If the comment is out of scope, set overall_score=1 and entities=[].

WRONG (for query "{query_str}"): {{"name":"[city abbreviation]","type":"location"}} or {{"name":"[region] spots","type":"region"}}
RIGHT (for query "{query_str}"): {{"name":"[specific named {entity_or_venue}]","type":"{entity_or_venue}","confidence":0.9}}

For each comment return:
1. overall_score: 1-5 (general comment sentiment about the main topic)
2. aspect_scores: per-aspect scores (1-5) for the main topic
3. primary_entity: name of the main sentiment-focus entity (string or null)
4. entities: all named entities with per-entity scores
5. solution_key: cluster label for SOLUTION queries (null otherwise)

Return a JSON array -- one object per comment, in input order:
[
    {{
        "overall_score": 4,
        "aspect_scores": {{"quality": 4, "value": 3, "performance": 5}},
        "primary_entity": "iPhone 15 Pro",
        "entities": [
            {{
                "name": "iPhone 15 Pro",
                "type": "phone",
                "confidence": 0.95,
                "sentiment_score": 4.5,
                "aspect_scores": {{"quality": 5, "value": 3}},
                "is_primary": true,
                "mention_context": "iPhone 15 Pro camera is outstanding"
            }},
            {{
                "name": "Samsung Galaxy S24",
                "type": "phone",
                "confidence": 0.85,
                "sentiment_score": 2.0,
                "aspect_scores": {{"quality": 3, "value": 2}},
                "is_primary": false,
                "mention_context": "compared to my old Samsung which felt cheap"
            }}
        ],
        "solution_key": null
    }}
]"""

                response = self.chat(
                    system=system_msg,
                    user=prompt,
                    temperature=0.2,
                    max_tokens=2500,
                )

                batch_annotations = _safe_json_loads(response)
                anno_by_pos = {j: d for j, d in enumerate(batch_annotations) if isinstance(d, dict)}

                batch_results = []
                for j, comment in enumerate(batch):
                    comment_id = comment.get("id", "") if isinstance(comment, dict) else getattr(comment, "id", "")
                    annotation_data = anno_by_pos.get(j, {})
                    entities = []
                    for ed in annotation_data.get("entities", []):
                        entities.append(EntityRef(
                            name=ed.get("name", ""),
                            entity_type=ed.get("type", entity_type or "unknown"),
                            confidence=float(ed.get("confidence", 0.0)),
                            sentiment_score=float(ed.get("sentiment_score", annotation_data.get("overall_score", 3.0))),
                            aspect_scores={k: float(v) for k, v in (ed.get("aspect_scores") or {}).items()},
                            is_primary=bool(ed.get("is_primary", True)),
                            mention_context=str(ed.get("mention_context", ""))[:100],
                        ))
                    batch_results.append(GPTCommentAnno(
                        comment_id=comment_id,
                        overall_score=annotation_data.get("overall_score", 3),
                        aspect_scores=annotation_data.get("aspect_scores", {}),
                        entities=entities,
                        primary_entity=annotation_data.get("primary_entity") or None,
                        solution_key=annotation_data.get("solution_key") or "",
                    ))
                return batch_idx, batch_results

            except Exception as e:
                logger.error(f"GPT annotation batch {batch_idx} failed: {e}")
                fallbacks = []
                for comment in batch:
                    comment_id = comment.get("id", "") if isinstance(comment, dict) else getattr(comment, "id", "")
                    fallbacks.append(GPTCommentAnno(
                        comment_id=comment_id,
                        overall_score=3,
                        aspect_scores={a: 3 for a in aspects},
                        entities=[],
                        primary_entity=None,
                        solution_key="",
                    ))
                return batch_idx, fallbacks

        # Run all batches concurrently (diskcache is thread-safe).
        # max_workers=8 keeps well within OpenAI rate limits for any tier.
        max_workers = min(8, len(batches))
        results_by_idx: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_batch, idx, batch): idx
                       for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                batch_idx, batch_annos = future.result()
                results_by_idx[batch_idx] = batch_annos
                logger.debug(f"Annotation batch {batch_idx + 1}/{len(batches)} complete")

        # Reassemble in original comment order.
        annotations = []
        for idx in range(len(batches)):
            annotations.extend(results_by_idx.get(idx, []))
        return annotations

    def generate_dynamic_aspects(self, query: str, sample_comments: List[Dict] = None) -> List[str]:
        """Generate dynamic aspects based on query and sample comments."""
        try:
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
                max_tokens=300,
            )

            aspects = _safe_json_loads(response)
            if isinstance(aspects, list) and all(isinstance(a, str) for a in aspects):
                return aspects[:6]

        except Exception as e:
            logger.error(f"Dynamic aspect generation failed: {e}")

        return get_domain_aspects(query)


class FallbackLLMService:
    """Fallback LLM service using simple rules."""
    
    def __init__(self):
        logger.info("Using fallback LLM service")
    
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Fallback chat method - returns empty string."""
        logger.warning("Fallback LLM service chat called - no actual LLM available")
        return ""

    def filter_entities_by_type(self, names: list, entity_type: str) -> list:
        """Fallback: return all names unchanged."""
        return names

    def filter_relevant_comments(self, comments: list, query: str, threshold: float = 0.35) -> list:
        """Fallback: return all comments unchanged (no LLM available)."""
        return comments

    def validate_entity_locations(self, entities: list, query: str) -> list:
        """Fallback: return all entities unchanged (no LLM available)."""
        return entities

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
            "comment_must_patterns": []  # GPT sentiment analysis handles quality
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
            entity_type=None
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

    def summarize_ranking_with_gpt(self, query: str, ranking_items: List[Dict]) -> str:
        if not ranking_items:
            return f"No ranked entities found for {query}."
        top = ranking_items[:5]
        lines = [f"{i+1}. {item['name']} -- {item['overall_stars']:.1f}/5 ({item['mentions']} mentions)" for i, item in enumerate(top)]
        return f"Top results for '{query}':\n" + "\n".join(lines)

    def summarize_generic_with_gpt(self, query: str, aspects: Dict[str, float], overall: float, quotes: List[str]) -> str:
        verdict = "positive" if overall >= 3.5 else "mixed" if overall >= 2.5 else "negative"
        return f"Analysis of '{query}': overall {overall:.1f}/5 ({verdict} sentiment). " + \
               "No LLM available for detailed summary."

    def summarize_solutions_with_gpt(self, query: str, clusters: List[Dict]) -> str:
        if not clusters:
            return f"No solution clusters found for '{query}'."
        return f"Found {len(clusters)} solution approach(es) for '{query}': " + \
               ", ".join(c.get("title", "Untitled") for c in clusters[:3]) + "."

    # FallbackLLMService ends here. OpenAI pipeline methods are only in OpenAIService.
