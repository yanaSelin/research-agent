"""SearchBackend contract and the retrieval hit type.

A backend knows nothing about roles or collections — it is pure content retrieval.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchHit:
    """One retrieved document snippet. `url` is '' for internal KB hits."""

    url: str
    domain: str
    published: str | None
    title: str
    content: str


class SearchBackend(Protocol):
    """Content retrieval contract shared by every backend."""

    def search(self, query: str, k: int = 6) -> list[SearchHit]: ...
