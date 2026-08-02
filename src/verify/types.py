"""Data types for the verification pipeline."""
from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass(frozen=True)
class Evidence:
    """One globally-numbered piece of evidence collected from the conversation."""

    id: int
    url: str
    domain: str
    source_class: str
    published: str | None
    title: str
    content: str


@dataclass(frozen=True)
class Claim:
    """An atomic factual claim extracted from the draft, with its citations."""

    text: str
    cited_evidence_ids: tuple[int, ...]


class Stance(str, Enum):
    """How one evidence item relates to one claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class ClaimVerdict:
    """The code-computed verdict for a claim."""

    claim: Claim
    stance_by_evidence: dict[int, Stance]
    support_weight: float
    contra_weight: float
    verdict: Literal["verified", "contested", "unsupported"]
