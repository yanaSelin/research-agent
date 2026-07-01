"""Research agent — VULNERABLE version.

Counterpart to the secure `mitigated` branch. LLM06: the search closure has no ACL and
no domain block. LLM09: there is no verify_pipeline — the raw draft is returned unchecked.
"""
import logging
import os

from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import SecretStr

from tools.search import make_search

logger = logging.getLogger(__name__)


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
    """Build a ReAct agent with a mode-selected backend (no ACL — vulnerable).

    Args:
        role: 'basic' or 'admin' (has no security effect on this branch).
        mode: 'local' or 'web' — selects the web-collection backend.

    Returns:
        Compiled langgraph agent ready to invoke.
    """
    logger.info("Building vulnerable agent role=%r mode=%r", role, mode)
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        temperature=0,
    )
    tool_node = ToolNode([make_search(role, mode)], handle_tool_errors=True)
    return create_react_agent(llm, tools=tool_node, prompt=_build_system_prompt(role))


def run_agent(role: str, query: str, mode: str) -> str:
    """VULNERABLE: return the raw draft, unverified."""
    agent = build_agent(role, mode)
    result = agent.invoke({"messages": [("user", query)]})
    msgs = result.get("messages", [])  # CR-003: guard against empty message list
    return str(msgs[-1].content) if msgs else ""


def run_agent_with_context(role: str, query: str, mode: str) -> tuple[str, list[str]]:
    """Return (answer, retrieval_context) where context is one string per search hit.

    retrieval_context is consumed by deepeval FaithfulnessMetric / AnswerRelevancyMetric.
    """
    from langchain_core.messages import ToolMessage  # local import — no runtime dep outside eval

    from hitfmt import parse_hits

    agent = build_agent(role, mode)
    result = agent.invoke({"messages": [("user", query)]})
    msgs = result.get("messages", [])
    answer = str(msgs[-1].content) if msgs else ""
    context: list[str] = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            for hit in parse_hits(str(msg.content)):
                context.append(hit.content)
    return answer, context


def run_agent_debug(role: str, query: str, mode: str) -> tuple[str, list]:
    """VULNERABLE: no verifier — return the draft and an empty verdict list."""
    return run_agent(role, query, mode), []


def run_agent_conversation(role: str, turns: list[str], mode: str) -> str:
    """VULNERABLE: raw drafts accumulate across turns."""
    agent = build_agent(role, mode)
    state: list = []  # CR-004: carry full LangGraph state (incl. tool messages) across turns
    response = ""
    for user_msg in turns:
        result = agent.invoke({"messages": state + [("user", user_msg)]})
        state = list(result.get("messages", []))
        response = str(state[-1].content) if state else ""
    return response
