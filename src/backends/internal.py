"""InternalKBBackend — searches the committed internal KB in both modes."""
import json
import logging
from pathlib import Path

from backends.base import SearchHit  # type: ignore[import-not-found]
from backends.scoring import top_k  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent.parent.parent / "data" / "internal_kb.json"


class InternalKBBackend:
    """Reads data/internal_kb.json; hits carry url='' and domain='internal.corp'."""

    def __init__(self, kb_path: Path | None = None) -> None:
        path = kb_path or _KB_PATH
        self._docs = json.loads(path.read_text(encoding="utf-8"))["documents"]
        logger.info("InternalKBBackend loaded %d docs", len(self._docs))

    def search(self, query: str, k: int = 6) -> list[SearchHit]:
        """Return up to k keyword-scored internal hits (url always empty)."""
        hits = top_k(self._docs, query, k, lambda d: f"{d['title']} {d['content']}")
        return [
            SearchHit(
                url="",
                domain=d.get("domain", "internal.corp"),
                published=d.get("published"),
                title=d["title"],
                content=d["content"],
            )
            for d in hits
        ]
