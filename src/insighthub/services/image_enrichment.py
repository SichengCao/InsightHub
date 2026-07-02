"""Entity image enrichment for InsightHub.

Resolves a photo URL for a ranked entity (e.g. a restaurant) using the Google
Places Photo API, falling back to a Yelp Fusion business image. Results — including
misses — are cached on disk so repeat runs and re-renders never re-hit the APIs.

This is domain-agnostic: it takes an entity name plus free-text location context
(usually the raw query) and asks the providers to resolve it. No keyword lists,
no per-category branching. Returns None when nothing is found or no keys are set,
which the frontend treats as "enrichment genuinely failed" (show a placeholder).
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


class ImageEnrichmentService:
    """Resolve and cache entity photo URLs (Google Places → Yelp Fusion)."""

    def __init__(self):
        self.google_key = getattr(settings, "google_places_api_key", "") or ""
        self.yelp_key = getattr(settings, "yelp_api_key", "") or ""
        self._yelp_headers = {"Authorization": f"Bearer {self.yelp_key}"} if self.yelp_key else {}
        self._cache = Cache(FileConstants.IMAGE_CACHE_DIR)

    @property
    def enabled(self) -> bool:
        """True when at least one image provider has credentials."""
        return bool(self.google_key or self.yelp_key)

    def get_image_url(self, name: str, location: str = "") -> Optional[str]:
        """Return a photo URL for `name` (with optional location context), or None.

        Cached by (name, location). A cached None is a real negative result and is
        returned as-is so we don't retry known-misses on every render.
        """
        if not name or not name.strip():
            return None
        if not self.enabled:
            return None

        key = ("entity_image_v1", name.lower().strip(), (location or "").lower().strip())
        cached = self._cache.get(key, default=_MISS)
        if cached is not _MISS:
            return cached

        url = self._from_google(name, location) or self._from_yelp(name, location)
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
