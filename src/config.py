"""Configuration loaders: access policy, source classes/trust, verifier thresholds.

All trust and access data lives in config/*.json so adding a domain or adjusting a
threshold is a data change, not a code change.
"""
import json
from functools import lru_cache
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent / "config"


@lru_cache(maxsize=1)
def load_policy() -> dict:
    """Load the role → collections + blocked_domain_patterns access policy."""
    return json.loads((_CONFIG_DIR / "policy.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_source_classes() -> dict:
    """Load source-class trust weights and the domain → class map."""
    return json.loads((_CONFIG_DIR / "source_classes.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    """Load verifier thresholds (T_support_ok, T_contra_veto, min_domain_class_weight)."""
    return json.loads((_CONFIG_DIR / "verifier.json").read_text(encoding="utf-8"))


def source_class(domain: str) -> str:
    """Map a domain to its source class by exact lookup; unknown domains → 'unknown'.

    Args:
        domain: Bare host, e.g. 'encyclopedia.example'.

    Returns:
        A source-class name from source_classes.json, or 'unknown'.
    """
    return load_source_classes()["domain_class_map"].get(domain, "unknown")


def trust_weight(source_cls: str) -> float:
    """Return the trust weight for a source class; unknown class → the 'unknown' weight.

    Args:
        source_cls: Source-class name (e.g. 'encyclopedia').

    Returns:
        Trust weight in [0, 1].
    """
    classes = load_source_classes()["classes"]
    if source_cls in classes:
        return float(classes[source_cls]["trust_weight"])
    return float(classes["unknown"]["trust_weight"])
