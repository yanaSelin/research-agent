"""Composes the five verifier steps into production and debug entry points."""
import logging

from config import load_thresholds  # type: ignore[import-not-found]
from verify.evidence import collect_evidence  # type: ignore[import-not-found]
from verify.extractor import extract_claims  # type: ignore[import-not-found]
from verify.finalizer import finalize  # type: ignore[import-not-found]
from verify.policy import classify  # type: ignore[import-not-found]
from verify.scorer import score_claims  # type: ignore[import-not-found]
from verify.types import ClaimVerdict  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _run(state: dict, draft: str) -> tuple[str, list[ClaimVerdict]]:
    evidence = collect_evidence(state)
    if not evidence:
        logger.info("verify: no evidence; returning draft unchanged")
        return draft, []
    ev_by_id = {e.id: e for e in evidence}
    claims = extract_claims(draft, evidence)
    if not claims:
        logger.info("verify: no claims extracted; returning draft unchanged")
        return draft, []
    matrix = score_claims(claims, evidence)
    thresholds = load_thresholds()
    verdicts = [
        classify(claim, matrix.get(i, {}), ev_by_id, thresholds)
        for i, claim in enumerate(claims)
    ]
    final = finalize(draft, verdicts, evidence)
    return final, verdicts


def verify_pipeline(state: dict, draft: str) -> str:
    """Production path: return the finalized answer string."""
    return _run(state, draft)[0]


def verify_pipeline_debug(state: dict, draft: str) -> tuple[str, list[ClaimVerdict]]:
    """Eval path: return the finalized answer plus the claim verdicts."""
    return _run(state, draft)
