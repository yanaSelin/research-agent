"""TavilyBackend — live web search for production mode."""
import logging
from urllib.parse import urlparse

from tavily import TavilyClient

from backends.base import SearchHit  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class TavilyBackend:
    """Wraps the Tavily search API, mapping results to SearchHit."""

    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key)

    def search(self, query: str, k: int = 6) -> list[SearchHit]:
        """Query Tavily and map results to SearchHit. Retryless (Tavily retries)."""
        raw = self._client.search(query, max_results=k, search_depth="basic")
        return [self._map_hit(r) for r in raw.get("results", [])]

    @staticmethod
    def _map_hit(r: dict) -> SearchHit:
        """Map one Tavily result dict to a SearchHit (pure, network-free)."""
        url = r.get("url") or ""
        domain = urlparse(url).netloc.removeprefix("www.")
        content = (r.get("content") or "")[:500]
        return SearchHit(
            url=url,
            domain=domain,
            published=r.get("published_date"),
            title=r.get("title") or "",
            content=content,
        )
