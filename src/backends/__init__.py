"""Backend selection: local corpus for eval/tests, Tavily for production."""
import logging
import os

from backends.base import SearchBackend, SearchHit  # noqa: F401
from backends.internal import InternalKBBackend  # noqa: F401
from backends.local import LocalCorpusBackend
from backends.tavily import TavilyBackend

logger = logging.getLogger(__name__)


def pick_web_backend(mode: str) -> SearchBackend:
    """Return the web-collection backend for the given mode.

    'web' mode uses Tavily when TAVILY_API_KEY is set, otherwise falls back to the
    local corpus with a logged warning. Any other mode uses the local corpus.

    Args:
        mode: 'web' or 'local'.

    Returns:
        A SearchBackend for the 'web' collection.
    """
    if mode == "web":
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.warning(
                "AI_ARCHITECT_SEARCH_MODE=web but TAVILY_API_KEY missing; "
                "falling back to local corpus"
            )
            return LocalCorpusBackend()
        return TavilyBackend(api_key)
    return LocalCorpusBackend()
