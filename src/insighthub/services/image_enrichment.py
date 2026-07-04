"""Entity image enrichment for InsightHub.

Resolves a photo URL for a ranked entity through a provider chain chosen by the
caller (the query category comes from GPT, so no keyword rules live here):

  places / businesses   → Google Places → Yelp Fusion
  everything else       → Wikipedia lead image (free, keyless, CC-licensed)

Results — including misses — are cached on disk so repeat runs and re-renders
never re-hit the APIs. Returns None when nothing is found or the chain's keys
aren't set, which the frontend treats as "enrichment genuinely failed" (show a
placeholder). New providers (TMDB, OpenLibrary, Custom Search, …) slot in by
adding a `_from_<name>` method and listing the name in `_available`.
"""

import logging
from typing import Optional

import requests
from diskcache import Cache

from ..core.config import settings
from ..core.constants import FileConstants

logger = logging.getLogger(__name__)

# Business photos are stable; cache aggressively (also caches negative results).
_CACHE_EXPIRE = 60 * 60 * 24 * 30  # 30 days
_MISS = object()  # sentinel: distinguishes "cached as None" from "not cached"
_TIMEOUT = 8


PLACE_PROVIDERS = ("google", "yelp")
GENERIC_PROVIDERS = ("wikipedia",)

# Wikimedia asks API clients to identify themselves.
_WIKI_UA = {"User-Agent": "InsightHub/1.0 (entity image enrichment)"}


class ImageEnrichmentService:
    """Resolve and cache entity photo URLs through a caller-chosen provider chain."""

    def __init__(self):
        self.google_key = getattr(settings, "google_places_api_key", "") or ""
        self.yelp_key = getattr(settings, "yelp_api_key", "") or ""
        self._yelp_headers = {"Authorization": f"Bearer {self.yelp_key}"} if self.yelp_key else {}
        self._cache = Cache(FileConstants.IMAGE_CACHE_DIR)

    @property
    def enabled(self) -> bool:
        """True when at least one keyed place provider has credentials."""
        return bool(self.google_key or self.yelp_key)

    def _available(self, provider: str) -> bool:
        """Whether a provider can be called at all (Wikipedia needs no key)."""
        return {"google": bool(self.google_key),
                "yelp": bool(self.yelp_key),
                "wikipedia": True}.get(provider, False)

    def get_image_url(self, name: str, location: str = "",
                      providers: tuple = PLACE_PROVIDERS) -> Optional[str]:
        """Return a photo URL for `name` via the given provider chain, or None.

        Cached by (chain, name, location). A cached None is a real negative result
        and is returned as-is so we don't retry known-misses on every render.
        """
        if not name or not name.strip():
            return None
        chain = [p for p in providers if self._available(p)]
        if not chain:
            return None

        key = ("entity_image_v3", "+".join(chain),
               name.lower().strip(), (location or "").lower().strip())
        cached = self._cache.get(key, default=_MISS)
        if cached is not _MISS:
            return cached

        url = None
        for p in chain:
            # late-bound dispatch: providers are `_from_<name>` methods, and
            # `_available` whitelists the names, so this can't hit arbitrary attrs
            url = getattr(self, f"_from_{p}")(name, location)
            if url:
                break
        self._cache.set(key, url, expire=_CACHE_EXPIRE)
        return url

    def _from_google(self, name: str, location: str) -> Optional[str]:
        """Places API (New): searchText → first result's photo → photo media URI.

        Uses the v1 endpoints (places.googleapis.com). The photo-media call with
        skipHttpRedirect=true returns the key-free googleusercontent CDN URI as
        JSON, so the API key never leaks into the rendered HTML.
        """
        if not self.google_key:
            return None
        try:
            term = f"{name} {location}".strip()
            resp = requests.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={
                    "X-Goog-Api-Key": self.google_key,
                    "X-Goog-FieldMask": "places.id,places.displayName,places.photos",
                    "Content-Type": "application/json",
                },
                json={"textQuery": term, "maxResultCount": 5},
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"Google Places(New) search failed for '{term}': {resp.status_code} {resp.text[:200]}")
                return None
            for place in resp.json().get("places", []):
                photos = place.get("photos") or []
                if not photos:
                    continue
                photo_name = photos[0].get("name")  # e.g. "places/<id>/photos/<ref>"
                if not photo_name:
                    continue
                uri = self._resolve_google_photo(photo_name)
                if uri:
                    return uri
            return None
        except Exception as e:  # noqa: BLE001 — enrichment must never break ranking
            logger.warning(f"Google Places image lookup error for '{name}': {e}")
            return None

    def _resolve_google_photo(self, photo_name: str) -> Optional[str]:
        """Resolve a Places(New) photo resource name to its key-free CDN URI."""
        try:
            r = requests.get(
                f"https://places.googleapis.com/v1/{photo_name}/media",
                params={"maxWidthPx": 800, "skipHttpRedirect": "true", "key": self.google_key},
                timeout=_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json().get("photoUri")
            logger.warning(f"Google Places(New) photo media failed: {r.status_code}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not resolve Google photo media: {e}")
        return None

    def _from_wikipedia(self, name: str, location: str = "") -> Optional[str]:
        """Wikipedia lead image: search → best non-disambiguation page's PageImage.

        Free and keyless; thumbnails are served from the stable Wikimedia CDN
        (upload.wikimedia.org). `pilicense=any` includes fair-use lead images
        (comic covers, game box art) — without it most fictional characters and
        games have no PageImage at all. `location` here is free-text search
        context (the caller passes the GPT-derived entity type): it disambiguates
        single-word names — "Storm" alone finds the weather article, "Storm
        marvel hero" finds the character. The title guard below still matches on
        the entity name only.
        """
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{name} {location}".strip(),
                    "gsrlimit": 5,
                    "prop": "pageimages|pageprops",
                    "piprop": "thumbnail",
                    "pithumbsize": 800,
                    "pilicense": "any",
                    "redirects": 1,
                    "format": "json",
                },
                headers=_WIKI_UA,
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"Wikipedia image search failed for '{name}': {resp.status_code}")
                return None
            pages = (resp.json().get("query") or {}).get("pages") or {}
            # Title guard: full-text search can rank an unrelated article first
            # ("Breville Barista Express" → "Espresso machine"; "Night Crawler" →
            # "Dungeon Crawler Carl" via the shared word). Accept a page only if
            # its title contains ALL of the name's real tokens, or contains the
            # whole name ignoring spacing ("Night Crawler" ⊆ "Nightcrawler
            # (character)") — a wrong photo is worse than a placeholder.
            import re as _re2
            toks = {w for w in _re2.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3}
            spaceless = "".join(_re2.findall(r"[a-z0-9]+", name.lower()))

            def _match_rank(page):
                """0 = title IS the name (ignoring '(…)' qualifier), 1 = superset
                title ("She-Hulk" for "Hulk"), None = unrelated → rejected."""
                title = str(page.get("title", "")).lower()
                base = title.split(" (")[0]
                if "".join(_re2.findall(r"[a-z0-9]+", base)) == spaceless:
                    return 0
                title_toks = set(_re2.findall(r"[a-z0-9]+", title))
                title_spaceless = "".join(_re2.findall(r"[a-z0-9]+", title))
                if toks and not (toks <= title_toks
                                 or (spaceless and spaceless in title_spaceless)):
                    return None
                return 1

            # generator results are unordered; "index" is the search rank.
            # Exact-name titles beat superset titles regardless of search rank.
            candidates = []
            for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
                if "disambiguation" in (page.get("pageprops") or {}):
                    continue
                rank = _match_rank(page)
                thumb = (page.get("thumbnail") or {}).get("source")
                if rank is not None and thumb:
                    candidates.append((rank, page.get("index", 99), thumb))
            if candidates:
                return min(candidates)[2]
            return None
        except Exception as e:  # noqa: BLE001 — enrichment must never break ranking
            logger.warning(f"Wikipedia image lookup error for '{name}': {e}")
            return None

    def _from_yelp(self, name: str, location: str) -> Optional[str]:
        """Yelp Fusion: business search → first match's image_url."""
        if not self.yelp_key:
            return None
        try:
            params = {"term": name, "limit": 1}
            if location:
                params["location"] = location
            resp = requests.get(
                "https://api.yelp.com/v3/businesses/search",
                headers=self._yelp_headers,
                params=params,
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"Yelp image search failed for '{name}': {resp.status_code}")
                return None
            businesses = resp.json().get("businesses", [])
            if businesses:
                return businesses[0].get("image_url") or None
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Yelp image lookup error for '{name}': {e}")
            return None


# Module-level singleton so the disk cache is shared across calls.
_service: Optional[ImageEnrichmentService] = None


def get_image_service() -> ImageEnrichmentService:
    global _service
    if _service is None:
        _service = ImageEnrichmentService()
    return _service
