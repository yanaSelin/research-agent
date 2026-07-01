"""Step 2: extract atomic claims (and their citations) from the draft. One LLM call."""
import logging

from verify import llm
from verify.types import Claim, Evidence

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract atomic factual claims from an assistant answer draft. The draft cites "
    "evidence with markers like [1], [3]. Preserve those citations per claim. Return JSON: "
    '{"claims": [{"text": "...", "cited_evidence_ids": [1, 3]}]}'
)


def _as_ids(values: list) -> list[int]:
    ids: list[int] = []
    for v in values:
        s = str(v).lstrip("-")
        if s.isdigit():
            ids.append(int(s))  # CR-006: was int(v), which stored negative IDs unchanged
    return ids


def extract_claims(draft: str, evidence: list[Evidence]) -> list[Claim]:
    """Extract claims with preserved citations from the draft."""
    ids = [e.id for e in evidence]
    user = f"Draft:\n{draft}\n\nAvailable evidence IDs: {ids}"
    data = llm.chat_json(_SYSTEM, user)
    claims_raw = data.get("claims") or []  # CR-007: .get("claims", []) returns None on {"claims": null}
    if not isinstance(claims_raw, list):
        claims_raw = []
    claims = [
        Claim(text=str(c.get("text", "")).strip(),
              cited_evidence_ids=_as_ids(c.get("cited_evidence_ids", [])))
        for c in claims_raw
        if isinstance(c, dict)
    ]
    logger.info("extract_claims: %d claims", len(claims))
    return claims
