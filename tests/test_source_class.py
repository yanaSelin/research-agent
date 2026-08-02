from config import source_class, trust_weight  # type: ignore[import-not-found]


def test_known_domain_maps_to_its_class():
    assert source_class("encyclopedia.example") == "encyclopedia"
    assert source_class("forum.example") == "forum"
    assert source_class("internal.corp") == "internal"


def test_unknown_domain_maps_to_unknown():
    assert source_class("never-seen-before.example") == "unknown"


def test_decoy_domains_map_to_blog():
    # both internal-like decoy domains classify consistently (not left as 'unknown')
    assert source_class("docs.internal.example") == "blog"
    assert source_class("wiki.corp.example") == "blog"


def test_trust_weight_known_and_fallback():
    assert trust_weight("encyclopedia") == 1.0
    assert trust_weight("forum") == 0.3
    assert trust_weight("no-such-class") == 0.2
