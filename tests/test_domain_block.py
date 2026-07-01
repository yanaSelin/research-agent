from tools.search import _domain_blocked  # type: ignore[import-not-found]

_BASIC_PATTERNS = ["*.internal.example", "*.corp.example"]


def test_blocks_matching_internal_domains():
    assert _domain_blocked("docs.internal.example", _BASIC_PATTERNS) is True
    assert _domain_blocked("wiki.corp.example", _BASIC_PATTERNS) is True


def test_allows_public_domains():
    assert _domain_blocked("encyclopedia.example", _BASIC_PATTERNS) is False
    assert _domain_blocked("news.reuters.example", _BASIC_PATTERNS) is False


def test_empty_patterns_block_nothing():
    assert _domain_blocked("docs.internal.example", []) is False
