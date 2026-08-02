"""Backend selection: always local corpus (deterministic, no external deps)."""
from backends.base import SearchBackend, SearchHit  # noqa: F401
from backends.internal import InternalKBBackend  # noqa: F401
from backends.local import LocalCorpusBackend


def pick_web_backend(mode: str = "local") -> SearchBackend:  # noqa: ARG001
    """Return the web-collection backend — always LocalCorpusBackend."""
    return LocalCorpusBackend()
