"""Shared keyword scoring for the local corpus and internal KB backends."""
from collections.abc import Callable

_STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
               "do", "does", "did", "how", "why", "when", "where", "who", "which"}


def keyword_score(text: str, query: str) -> int:
    """Count distinct non-stopword query terms appearing in text (case-insensitive)."""
    words = {w.lower().strip("?.,!") for w in query.split()} - _STOP_WORDS
    lowered = text.lower()
    return sum(1 for w in words if w in lowered)


def top_k(docs: list[dict], query: str, k: int, text_of: Callable[[dict], str]) -> list[dict]:
    """Return up to k docs with score > 0, highest score first."""
    scored = [(keyword_score(text_of(d), query), d) for d in docs]
    scored = [(s, d) for s, d in scored if s > 0]
    scored.sort(key=lambda sd: sd[0], reverse=True)
    return [d for _, d in scored[:k]]
