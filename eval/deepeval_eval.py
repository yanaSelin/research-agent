"""DeepEval quality metrics: AnswerRelevancy + Faithfulness.

Measures answer quality independently from the security eval (attacks.py):
  - AnswerRelevancyMetric  — does the answer address the question?
  - FaithfulnessMetric     — does the answer stay within the retrieved sources?

Runs on easy + uncontested canaries (trap canaries are the domain of attacks.py).

Key finding on the vulnerable branch: FaithfulnessMetric scores HIGH because the
agent faithfully follows its sources — even when those sources contain misinformation.
This demonstrates that faithfulness-to-sources ≠ factual accuracy when sources are
compromised; the custom LLM09 attack harness catches what deepeval cannot.
"""
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402

from agent import run_agent_with_context  # type: ignore[import-not-found]
from logging_config import setup_logging  # type: ignore[import-not-found]
from scenarios import CANARIES  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_MODE = "local"
_EVAL_SCENARIOS = {"easy", "uncontested"}  # trap canaries → attacks.py
_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Azure OpenAI wrapper for deepeval
# ---------------------------------------------------------------------------

class _AzureDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEvalBaseLLM implementation that routes metric-judge calls to Azure OpenAI."""

    def __init__(self, client: AzureOpenAI, deployment: str) -> None:
        self._client = client
        self._deployment = deployment
        super().__init__()

    def load_model(self):  # type: ignore[override]  # noqa: ANN201
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        completion = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if not completion.choices:  # CR-008: content-filter / overload can return 0 choices
            return ""
        return completion.choices[0].message.content or ""

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._deployment


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CanaryResult:
    """Deepeval scores for a single canary question."""

    topic_key: str
    scenario: str
    question: str
    answer: str
    context_chunks: int
    answer_relevancy: float | None
    answer_relevancy_reason: str
    faithfulness: float | None
    faithfulness_reason: str
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        return out.stdout.strip() or "main"
    except Exception:  # noqa: BLE001
        return "main"


def _measure(metric, test_case) -> tuple[float | None, str]:  # noqa: ANN001
    """Run a single deepeval metric and return (score, reason)."""
    try:
        metric.measure(test_case)
        return metric.score, metric.reason or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Metric %s failed: %s", type(metric).__name__, exc)
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: PLR0914
    """Run deepeval metrics on easy + uncontested canaries; save results JSON."""
    load_dotenv()
    os.environ["AI_ARCHITECT_SEARCH_MODE"] = _MODE
    setup_logging("eval")
    branch = _branch()

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    judge = _AzureDeepEvalLLM(client, deployment)

    # Import deepeval here so the module can be imported without deepeval installed
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: PLC0415
    from deepeval.test_case import LLMTestCase  # noqa: PLC0415

    eval_canaries = [c for c in CANARIES if c.scenario in _EVAL_SCENARIOS]
    print(f"Running deepeval metrics (branch={branch}, canaries={len(eval_canaries)})...\n")

    results: list[CanaryResult] = []
    for canary in eval_canaries:
        print(f"  [{canary.scenario:12}] {canary.topic_key} ...", flush=True)
        try:
            answer, context = run_agent_with_context(role="basic", query=canary.question, mode=_MODE)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent failed for %s", canary.topic_key)
            results.append(CanaryResult(
                topic_key=canary.topic_key, scenario=canary.scenario,
                question=canary.question, answer="", context_chunks=0,
                answer_relevancy=None, answer_relevancy_reason=str(exc),
                faithfulness=None, faithfulness_reason=str(exc), passed=False,
            ))
            continue

        test_case = LLMTestCase(
            input=canary.question,
            actual_output=answer,
            retrieval_context=context,  # type: ignore[arg-type]
        )

        rel_score, rel_reason = _measure(
            AnswerRelevancyMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
            test_case,
        )
        faith_score, faith_reason = _measure(
            FaithfulnessMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
            test_case,
        )

        passed = (
            (rel_score is not None and rel_score >= _THRESHOLD)
            and (faith_score is not None and faith_score >= _THRESHOLD)
        )
        results.append(CanaryResult(
            topic_key=canary.topic_key, scenario=canary.scenario,
            question=canary.question, answer=answer,
            context_chunks=len(context),
            answer_relevancy=rel_score, answer_relevancy_reason=rel_reason,
            faithfulness=faith_score, faithfulness_reason=faith_reason,
            passed=passed,
        ))

        rel_str = f"{rel_score:.3f}" if rel_score is not None else "err"
        faith_str = f"{faith_score:.3f}" if faith_score is not None else "err"
        print(f"    relevancy={rel_str}  faithfulness={faith_str}  {'PASS' if passed else 'FAIL'}")

    # Summary
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    valid = [r for r in results if r.answer_relevancy is not None]
    mean_rel = sum(r.answer_relevancy for r in valid) / len(valid) if valid else 0.0  # type: ignore[arg-type]
    valid_f = [r for r in results if r.faithfulness is not None]
    mean_faith = sum(r.faithfulness for r in valid_f) / len(valid_f) if valid_f else 0.0  # type: ignore[arg-type]

    print(f"\nSummary: {n_pass}/{n_total} passed  |  "
          f"mean_relevancy={mean_rel:.3f}  mean_faithfulness={mean_faith:.3f}")

    payload = {
        "branch": branch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": _THRESHOLD,
        "scenarios": sorted(_EVAL_SCENARIOS),
        "summary": {
            "total": n_total,
            "passed": n_pass,
            "mean_answer_relevancy": round(mean_rel, 3),
            "mean_faithfulness": round(mean_faith, 3),
        },
        "results": [asdict(r) for r in results],
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / f"deepeval_{branch}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
