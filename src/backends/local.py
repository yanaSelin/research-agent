"""LocalCorpusBackend — deterministic in-memory search over the simulated web corpus."""
import json
import logging
from pathlib import Path

from backends.base import SearchHit  # type: ignore[import-not-found]
from backends.scoring import top_k  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_DEFAULT_CORPUS = Path(__file__).parent.parent.parent / "data" / "web_corpus.json"


class LocalCorpusBackend:
    """Reads the committed web corpus once and serves keyword-scored hits."""

    def __init__(self, corpus_path: Path | None = None) -> None:
        path = corpus_path or _DEFAULT_CORPUS
        self._docs = json.loads(path.read_text(encoding="utf-8"))["documents"]
        logger.info("LocalCorpusBackend loaded %d docs from %s", len(self._docs), path)

    def search(self, query: str, k: int = 6) -> list[SearchHit]:
        """Return up to k keyword-scored hits for the query."""
        hits = top_k(self._docs, query, k, lambda d: f"{d['title']} {d['content']}")
        return [
            SearchHit(
                url=d["url"],
                domain=d["domain"],
                published=d.get("published"),
                title=d["title"],
                content=d["content"],
            )
            for d in hits
        ]
