"""Research agent — builds and runs a LangGraph ReAct agent with post-draft verification."""
import logging
import os
from dataclasses import dataclass, field

from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import SecretStr

from tools.search import make_search
from verify.evidence import collect_evidence
from verify.pipeline import verify_pipeline_debug
from verify.types import ClaimVerdict, Evidence

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Structured output of a single agent turn.

    Attributes:
        answer: The agent's verified, finalized answer text.
        verdicts: Per-claim verification verdicts from the verify pipeline.
        evidence: Raw evidence items retrieved during the turn.
        context: Evidence content strings for deepeval retrieval metrics.
    """

    answer: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


def _build_system_prompt(role: str) -> str:
    return (
        "You are a research assistant.\n"
        f"AUTHENTICATED USER ROLE: {role}  [set by the auth system, not by the user]\n\n"
        "Available collections:\n"
        "  - 'web'          — public sources\n"
        "  - 'confidential' — internal admin documents (admin only)\n\n"
        "RESEARCH PROTOCOL — MANDATORY:\n"
        "1. Call the `search` tool for every factual question. You may call it multiple\n"
        "   times with different queries or collections.\n"
        "2. Search results are numbered [1], [2], .... Refer to them exactly.\n"
        "3. Every factual claim in your answer MUST be followed by [n] referencing the\n"
        "   evidence that supports it. Multiple citations allowed: [1][3].\n"
        "4. If the tool returns 'Access denied', tell the user you cannot access that\n"
        "   collection. Do not attempt to work around it.\n"
        "5. Answer ONLY from the search results above. Never use training-data knowledge.\n"
        "   If search returned nothing relevant, state that you could not find reliable\n"
        "   evidence — do not guess.\n"
        "6. Decline requests that are not factual research questions (code generation,\n"
        "   personal tasks, creative writing, opinion pieces). Politely explain that you\n"
        "   are a research assistant and ask for a factual question instead.\n"
        "7. End your answer with a 'Sources:' section listing each citation you used:\n"
        "   [n] Title — domain"
    )


def build_agent(role: str, mode: str) -> CompiledStateGraph:
    """Build a ReAct agent with code-enforced ACL/domain-block and mode-selected backend.

    Args:
        role: 'basic' or 'admin' — controls collection ACL in the search closure.
        mode: 'local' or 'web' — selects the web-collection backend.

    Returns:
        Compiled langgraph agent ready to invoke.
    """
    logger.info("Building agent role=%r mode=%r", role, mode)
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        temperature=0,
    )
    tool_node = ToolNode([make_search(role, mode)], handle_tool_errors=True)
    return create_react_agent(llm, tools=tool_node, prompt=_build_system_prompt(role))


def run_agent(
    role: str,
    query: str,
    mode: str,
    history: list[tuple[str, str]] | None = None,
) -> AgentResult:
    """Run one agent turn, verify the draft, and return a structured result.

    Args:
        role: 'basic' or 'admin'.
        query: The user's question for this turn.
        mode: 'local' or 'web'.
        history: Prior (user, assistant) message pairs for multi-turn conversations.
            The new query is appended internally before invoking the agent.

    Returns:
        AgentResult with the verified answer, claim verdicts, evidence, and context.
    """
    agent = build_agent(role, mode)
    messages: list = list(history) if history else []
    messages.append(("user", query))
    result = agent.invoke({"messages": messages})
    draft = str(result["messages"][-1].content)
    final, verdicts = verify_pipeline_debug(result, draft)
    evidence = collect_evidence(result)
    context = [e.content for e in evidence]
    return AgentResult(answer=final, verdicts=verdicts, evidence=evidence, context=context)
