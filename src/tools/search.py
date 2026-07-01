"""Search tool — VULNERABLE branch: no ACL, no domain block (for comparison).

This is the intentionally-insecure counterpart to the mitigated `main` version.
Any role can reach any collection, and internal-domain pages are never filtered.
"""
import logging

from langchain_core.tools import BaseTool, tool

from backends import pick_web_backend
from backends.internal import InternalKBBackend
from hitfmt import format_hits

logger = logging.getLogger(__name__)


def make_search(role: str, mode: str) -> BaseTool:
    """Return a search tool with NO access control (vulnerable demo)."""
    backends = {"web": pick_web_backend(mode), "confidential": InternalKBBackend()}
    logger.warning("VULNERABLE make_search: no ACL, no domain block (role=%r)", role)

    @tool
    def search(query: str, collection: str = "web") -> str:
        """Search the knowledge sources. Returns numbered hits [1], [2], ....

        Args:
            query: search query.
            collection: 'web' or 'confidential'.
        """
        backend = backends.get(collection, backends["web"])
        hits = backend.search(query, k=6)
        logger.info("search: collection=%r results=%d", collection, len(hits))
        if not hits:
            return "No relevant documents found for this query."
        return format_hits(hits)

    return search
