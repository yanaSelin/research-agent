"""Search tool: code-enforced collection ACL + domain blocking in a closure.

make_search(role, mode) bakes the role's permitted collections and blocked domain
patterns into the returned tool. The LLM cannot widen either — the checks are code.
"""
import fnmatch
import logging

from langchain_core.tools import BaseTool, tool

from backends import pick_web_backend  # type: ignore[import-not-found]
from backends.internal import InternalKBBackend  # type: ignore[import-not-found]
from config import load_policy  # type: ignore[import-not-found]
from hitfmt import format_hits  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _domain_blocked(domain: str, patterns: list[str]) -> bool:
    """True if domain matches any fnmatch blocked pattern."""
    return any(fnmatch.fnmatch(domain, p) for p in patterns)


def make_search(role: str, mode: str) -> BaseTool:
    """Build a search tool restricted to `role`'s collections and domain blocks.

    Args:
        role: Authenticated role ('basic' or 'admin').
        mode: 'local' or 'web' — selects the web-collection backend.

    Returns:
        A LangChain tool enforcing ACL and domain blocks in code.
    """
    policy = load_policy()["roles"][role]
    allowed_collections = set(policy["collections"])
    blocked = policy["blocked_domain_patterns"]
    backends = {
        "web": pick_web_backend(mode),
        "confidential": InternalKBBackend(),
    }
    logger.info("make_search: role=%r mode=%r collections=%s", role, mode, allowed_collections)

    @tool
    def search(query: str, collection: str = "web") -> str:
        """Search the knowledge sources. Returns numbered hits [1], [2], ....

        Args:
            query: search query.
            collection: 'web' (public sources) or 'confidential' (admin only).
        """
        if collection not in allowed_collections:
            logger.warning("ACL denied: role=%r collection=%r", role, collection)
            return f"Access denied: {collection!r} not permitted for role {role!r}."
        if collection not in backends:
            logger.error("Backend missing: collection=%r not in backends", collection)
            return f"Backend for {collection!r} is not configured."
        hits = backends[collection].search(query, k=6)
        hits = [h for h in hits if not _domain_blocked(h.domain, blocked)]
        if not hits:
            return "No relevant documents found for this query."
        logger.info("search: collection=%r results=%d", collection, len(hits))
        return format_hits(hits)

    return search
