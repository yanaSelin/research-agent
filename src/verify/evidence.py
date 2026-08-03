"""Step 1: walk the LangGraph message state and build the global evidence set."""
import logging

from langchain_core.messages import ToolMessage

from config import source_class
from hitfmt import parse_hits
from verify.types import Evidence

logger = logging.getLogger(__name__)


def collect_evidence(state: dict) -> list[Evidence]:
    """Collect deduplicated evidence from every search ToolMessage in the state.

    Args:
        state: The dict returned by agent.invoke (contains 'messages').

    Returns:
        Evidence with fresh global ids [1..N]; source_class computed from domain.
        Deduped by url (or by title when url is empty). Access-denied messages skipped.
    """
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for msg in state.get("messages", []):
        if not isinstance(msg, ToolMessage) or msg.name != "search":
            continue
        content = str(msg.content)
        if content.startswith("Access denied:"):
            continue
        for hit in parse_hits(content):
            key = hit.url or f"title::{hit.title}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                Evidence(
                    id=len(evidence) + 1,
                    url=hit.url,
                    domain=hit.domain,
                    source_class=source_class(hit.domain),
                    published=hit.published,
                    title=hit.title,
                    content=hit.content,
                )
            )
    logger.info("collect_evidence: %d evidence items", len(evidence))
    return evidence
