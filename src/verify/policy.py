"""Step 4: pure-code verdict from the stance matrix and source-class trust weights."""
from config import trust_weight  # type: ignore[import-not-found]
from verify.types import Claim, ClaimVerdict, Evidence, Stance  # type: ignore[import-not-found]


def classify(
    claim: Claim,
    stance_by_id: dict[int, Stance],
    evidence: dict[int, Evidence],
    thresholds: dict,
) -> ClaimVerdict:
    """Compute a claim verdict from weighted support/contradiction. No LLM.

    Args:
        claim: The claim being judged.
        stance_by_id: evidence_id → Stance for this claim.
        evidence: evidence_id → Evidence (for source-class trust lookup).
        thresholds: verifier thresholds dict.

    Returns:
        ClaimVerdict with support/contra weights and a verdict label.

    Evidence whose source-class trust weight is below ``min_domain_class_weight``
    is ignored on both sides, so very-low-trust sources cannot move a verdict.
    """
    floor = thresholds.get("min_domain_class_weight", 0.0)
    support = sum(
        w
        for i, s in stance_by_id.items()
        if s == Stance.SUPPORTS and i in evidence
        and (w := trust_weight(evidence[i].source_class)) >= floor
    )
    contra = sum(
        w
        for i, s in stance_by_id.items()
        if s == Stance.CONTRADICTS and i in evidence
        and (w := trust_weight(evidence[i].source_class)) >= floor
    )
    if support >= thresholds["T_support_ok"] and contra < thresholds["T_contra_veto"]:
        verdict = "verified"
    elif support > 0 and contra > 0:
        verdict = "contested"
    else:
        verdict = "unsupported"
    return ClaimVerdict(claim, stance_by_id, support, contra, verdict)
