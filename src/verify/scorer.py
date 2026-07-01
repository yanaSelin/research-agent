"""Step 3: one matrix LLM call scoring every claim against every evidence item."""
import logging

from verify import llm  # type: ignore[import-not-found]
from verify.types import Claim, Evidence, Stance  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_SYSTEM = (
    "For each (claim, evidence) pair decide: SUPPORTS, CONTRADICTS, or NEUTRAL.\n\n"
    "CONTRADICTS means the evidence asserts something incompatible with the claim — even if "
    "framed differently. This applies in BOTH directions of negation:\n"
    "- claim='X causes Y', evidence='X is NOT linked to Y' → CONTRADICTS\n"
    "- claim='X does NOT cause Y', evidence='X DOES cause Y' → CONTRADICTS\n"
    "- claim='there IS gravity at Z', evidence='there is NO gravity at Z' → CONTRADICTS\n"
    "- claim='event caused mass panic', evidence='no widespread panic occurred' → CONTRADICTS\n"
    "NEUTRAL means the evidence is on a completely different topic and neither confirms "
    "nor denies the claim.\n\n"
    "For every cell write ONE short reason, then the verdict.\n"
    'Return JSON: {"matrix": {"<claim_index>": {"<evidence_id>": '
    '{"r": "<one-sentence reason>", "v": "supports"|"contradicts"|"neutral"}}}}'
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
    data = llm.chat_json(_SYSTEM, user)
    matrix_raw = data.get("matrix", {})
    result: dict[int, dict[int, Stance]] = {}
    for i in range(len(claims)):
        cell = matrix_raw.get(str(i), {})
        result[i] = {
            int(eid): _stance(
                cell_val.get("v", "") if isinstance(cell_val, dict) else str(cell_val)
            )
            for eid, cell_val in cell.items()
            if str(eid).lstrip("-").isdigit()
        }
    return result
