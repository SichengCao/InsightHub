"""ImageEnrichmentService provider-chain tests — offline, no API keys required.

Covers:
  1. Wikipedia provider parses the generator=search response (rank order,
     disambiguation skip, thumbnail extraction)
  2. Chain routing: generic chain never calls place providers and needs no keys
  3. Negative caching: a miss is cached and not re-fetched
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insighthub.core.constants import FileConstants
from insighthub.services import image_enrichment as ie


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """Service with an isolated disk cache and no place-provider keys.

    Keys are blanked explicitly — a developer .env may hold real credentials and
    these tests must never hit live APIs."""
    monkeypatch.setattr(FileConstants, "IMAGE_CACHE_DIR", str(tmp_path / "imgcache"))
    s = ie.ImageEnrichmentService()
    s.google_key = ""
    s.yelp_key = ""
    s._yelp_headers = {}
    return s


def _wiki_response(pages):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"query": {"pages": pages}}
    return resp


WIKI_PAGES = {
    "1": {"index": 2, "title": "Iron Man (2008 film)",
          "thumbnail": {"source": "https://upload.wikimedia.org/film.jpg"}},
    "2": {"index": 1, "title": "Iron Man",
          "thumbnail": {"source": "https://upload.wikimedia.org/hero.jpg"}},
}


class TestWikipediaProvider:
    def test_picks_best_ranked_page(self, svc):
        with patch.object(ie.requests, "get", return_value=_wiki_response(WIKI_PAGES)):
            assert svc._from_wikipedia("Iron Man") == "https://upload.wikimedia.org/hero.jpg"

    def test_skips_disambiguation_pages(self, svc):
        pages = {
            "1": {"index": 1, "title": "Mercury",
                  "pageprops": {"disambiguation": ""},
                  "thumbnail": {"source": "https://upload.wikimedia.org/disambig.jpg"}},
            "2": {"index": 2, "title": "Mercury (planet)",
                  "thumbnail": {"source": "https://upload.wikimedia.org/planet.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Mercury") == "https://upload.wikimedia.org/planet.jpg"

    def test_skips_pages_with_unrelated_titles(self, svc):
        """A generic article outranking the entity must not supply its photo."""
        pages = {
            "1": {"index": 1, "title": "Espresso machine",
                  "thumbnail": {"source": "https://upload.wikimedia.org/other-brand.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Breville Barista Express") is None

    def test_title_guard_matches_on_all_tokens(self, svc):
        pages = {
            "1": {"index": 1, "title": "Sony α7 IV",
                  "thumbnail": {"source": "https://upload.wikimedia.org/a7.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Sony A7 IV") == "https://upload.wikimedia.org/a7.jpg"

    def test_title_guard_matches_spaceless_containment(self, svc):
        """'Night Crawler' must match 'Nightcrawler (character)'…"""
        pages = {
            "1": {"index": 1, "title": "Nightcrawler (character)",
                  "thumbnail": {"source": "https://upload.wikimedia.org/nc.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Night Crawler") == "https://upload.wikimedia.org/nc.jpg"

    def test_title_guard_rejects_single_shared_word(self, svc):
        """…but NOT 'Dungeon Crawler Carl', which only shares one word."""
        pages = {
            "1": {"index": 1, "title": "Dungeon Crawler Carl",
                  "thumbnail": {"source": "https://upload.wikimedia.org/carl.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Night Crawler") is None

    def test_exact_title_beats_higher_ranked_superset(self, svc):
        """'She-Hulk' outranking 'Hulk' in search must not win for name 'Hulk'."""
        pages = {
            "1": {"index": 1, "title": "She-Hulk",
                  "thumbnail": {"source": "https://upload.wikimedia.org/she-hulk.jpg"}},
            "2": {"index": 2, "title": "Hulk",
                  "thumbnail": {"source": "https://upload.wikimedia.org/hulk.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Hulk") == "https://upload.wikimedia.org/hulk.jpg"

    def test_parenthetical_qualifier_counts_as_exact(self, svc):
        """'Storm (Ororo Munroe)' is an exact title for 'Storm'."""
        pages = {
            "1": {"index": 1, "title": "Storm chasing",
                  "thumbnail": {"source": "https://upload.wikimedia.org/chasing.jpg"}},
            "2": {"index": 2, "title": "Storm (Ororo Munroe)",
                  "thumbnail": {"source": "https://upload.wikimedia.org/ororo.jpg"}},
        }
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Storm") == "https://upload.wikimedia.org/ororo.jpg"

    def test_no_thumbnail_returns_none(self, svc):
        pages = {"1": {"index": 1, "title": "Obscure Thing"}}
        with patch.object(ie.requests, "get", return_value=_wiki_response(pages)):
            assert svc._from_wikipedia("Obscure Thing") is None

    def test_http_error_returns_none(self, svc):
        resp = MagicMock(status_code=503)
        with patch.object(ie.requests, "get", return_value=resp):
            assert svc._from_wikipedia("Anything") is None


class TestChainRouting:
    def test_generic_chain_works_without_keys(self, svc):
        """Wikipedia needs no credentials; place chain without keys is a no-op."""
        assert svc._available("wikipedia") is True
        assert svc.get_image_url("Iron Man", providers=ie.PLACE_PROVIDERS) is None

    def test_generic_chain_never_calls_place_providers(self, svc):
        with patch.object(svc, "_from_google") as g, patch.object(svc, "_from_yelp") as y, \
             patch.object(svc, "_from_wikipedia", return_value="https://upload.wikimedia.org/x.jpg"):
            url = svc.get_image_url("Sony A7 IV", providers=ie.GENERIC_PROVIDERS)
        assert url == "https://upload.wikimedia.org/x.jpg"
        g.assert_not_called()
        y.assert_not_called()

    def test_place_and_generic_cache_keys_do_not_collide(self, svc):
        """Same name through different chains must not share a cached result."""
        with patch.object(svc, "_from_wikipedia", return_value="https://upload.wikimedia.org/x.jpg"):
            assert svc.get_image_url("Mercury", providers=ie.GENERIC_PROVIDERS) \
                == "https://upload.wikimedia.org/x.jpg"
        # keyless place chain resolves to an empty chain → None, not the wiki hit
        assert svc.get_image_url("Mercury", providers=ie.PLACE_PROVIDERS) is None


class TestNegativeCaching:
    def test_miss_is_cached_and_not_refetched(self, svc):
        with patch.object(svc, "_from_wikipedia", return_value=None) as w:
            assert svc.get_image_url("Nonexistent", providers=ie.GENERIC_PROVIDERS) is None
            assert svc.get_image_url("Nonexistent", providers=ie.GENERIC_PROVIDERS) is None
        assert w.call_count == 1

    def test_hit_is_cached(self, svc):
        with patch.object(svc, "_from_wikipedia",
                          return_value="https://upload.wikimedia.org/x.jpg") as w:
            for _ in range(3):
                assert svc.get_image_url("Iron Man", providers=ie.GENERIC_PROVIDERS) \
                    == "https://upload.wikimedia.org/x.jpg"
        assert w.call_count == 1
