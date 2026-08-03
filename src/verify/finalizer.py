"""Step 5: rewrite the draft per verdicts into the final answer. One LLM call."""
import logging

from verify import llm
from verify.types import ClaimVerdict, Evidence

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You finalize a research answer using per-claim verdicts.\n"
    "- verified claims: keep as-is with their [n] citations.\n"
    "- contested claims: present both sides, citing each side with [n] for its evidence "
    "(e.g. 'Encyclopedia [3] says X, while forums [1][2] say Y'), and state which side "
    "is preferred by trust weight.\n"
    "- unsupported claims: drop them or mark with '⚠ unsupported by evidence'.\n"
    "If nothing is supported, say plainly that you could not find reliable evidence.\n"
    "End with a 'Sources:' list of the evidence you used (id, domain, title). "
    'Return JSON: {"final_answer": "..."}'
)


def finalize(draft: str, verdicts: list[ClaimVerdict], evidence: list[Evidence]) -> str:
    """Produce the finalized answer string from the draft and verdicts."""
    verdict_lines = "\n".join(
        f"  - verdict={v.verdict} support={v.support_weight:.2f} "
        f"contra={v.contra_weight:.2f} :: {v.claim.text}"
        for v in verdicts
    )
    ev_lines = "\n".join(
        f"  [{e.id}] {e.domain} ({e.source_class}) — {e.title}" for e in evidence
    )
    user = f"Draft:\n{draft}\n\nClaim verdicts:\n{verdict_lines}\n\nEvidence:\n{ev_lines}"
    data = llm.chat_json(_SYSTEM, user)
    return str(data.get("final_answer", draft)).strip()
