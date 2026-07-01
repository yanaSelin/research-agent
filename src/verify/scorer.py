"""Step 3: one matrix LLM call scoring every claim against every evidence item."""
import logging

from verify import llm
from verify.types import Claim, Evidence, Stance

logger = logging.getLogger(__name__)

_SYSTEM = (
    "For each claim and each evidence item, decide whether the evidence SUPPORTS, "
    "CONTRADICTS, or is NEUTRAL toward the claim. Return JSON: "
    '{"matrix": {"<claim_index>": {"<evidence_id>": "supports"|"contradicts"|"neutral"}}}'
)


def _stance(value: str) -> Stance:
    try:
        return Stance(value.lower().strip())
    except ValueError:
        return Stance.NEUTRAL


def score_claims(claims: list[Claim], evidence: list[Evidence]) -> dict[int, dict[int, Stance]]:
    """Return {claim_index: {evidence_id: Stance}} for all claim×evidence cells."""
    if not claims or not evidence:
        return {i: {} for i in range(len(claims))}
    claim_lines = "\n".join(f"  {i}: {c.text}" for i, c in enumerate(claims))
    ev_lines = "\n".join(f"  [{e.id}] {e.domain} — {e.content}" for e in evidence)
    user = f"Claims:\n{claim_lines}\n\nEvidence:\n{ev_lines}"
    logger.info("score_claims: %d claims × %d evidence", len(claims), len(evidence))
    data = llm.chat_json(_SYSTEM, user)
    matrix_raw = data.get("matrix") or {}
    result: dict[int, dict[int, Stance]] = {}
    for i in range(len(claims)):
        cell = matrix_raw.get(str(i), {})
        result[i] = {
            int(eid): _stance(str(st))
            for eid, st in cell.items()
            if str(eid).isdigit()  # CR-010: strip "-" guard allowed negative IDs; isdigit() rejects them
        }
    logger.debug(
        "score_claims matrix: %s",
        {f"c{i}": {f"e{eid}": s.value for eid, s in stances.items()} for i, stances in result.items()},
    )
    return result
