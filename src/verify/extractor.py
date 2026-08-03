"""Step 2: extract atomic claims (and their citations) from the draft. One LLM call."""
import logging

from verify import llm
from verify.types import Claim, Evidence

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract atomic factual claims from an assistant answer draft. The draft cites "
    "evidence with markers like [1], [3]. Preserve those citations per claim.\n\n"
    "IMPORTANT: Extract the underlying factual ASSERTION — not epistemic hedges. "
    "Strip phrases like 'some sources say', 'may be linked to', 'according to [n]', "
    "'studies suggest'. Write the bare claim: 'X causes Y', not 'some say X may cause Y'. "
    "If the draft presents two conflicting facts, extract each as a separate claim.\n\n"
    'Return JSON: {"claims": [{"text": "...", "cited_evidence_ids": [1, 3]}]}'
)


def _as_ids(values: list) -> tuple[int, ...]:
    ids: list[int] = []
    for v in values:
        s = str(v).lstrip("-")
        if s.isdigit():
            ids.append(int(v))
    return tuple(ids)


def extract_claims(draft: str, evidence: list[Evidence]) -> list[Claim]:
    """Extract claims with preserved citations from the draft."""
    ids = [e.id for e in evidence]
    user = f"Draft:\n{draft}\n\nAvailable evidence IDs: {ids}"
    data = llm.chat_json(_SYSTEM, user)
    claims = [
        Claim(text=str(c.get("text", "")).strip(),
              cited_evidence_ids=_as_ids(c.get("cited_evidence_ids", [])))
        for c in data.get("claims", [])
    ]
    logger.info("extract_claims: %d claims", len(claims))
    return claims
