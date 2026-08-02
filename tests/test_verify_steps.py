from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backends.base import SearchHit  # type: ignore[import-not-found]
from hitfmt import format_hits  # type: ignore[import-not-found]
from verify import llm  # type: ignore[import-not-found]
from verify.evidence import collect_evidence  # type: ignore[import-not-found]
from verify.extractor import extract_claims  # type: ignore[import-not-found]
from verify.scorer import score_claims  # type: ignore[import-not-found]
from verify.finalizer import finalize  # type: ignore[import-not-found]
from verify.types import Claim, ClaimVerdict, Evidence, Stance  # type: ignore[import-not-found]


def _tool_msg(hits, start=1):
    return ToolMessage(content=format_hits(hits, start=start), name="search", tool_call_id="x")


def test_collect_evidence_renumbers_and_classifies():
    hits1 = [SearchHit("https://encyclopedia.example/e", "encyclopedia.example",
                       "2019-04-11", "Eiffel", "completed in 1889")]
    hits2 = [SearchHit("https://forum.example/t", "forum.example",
                       "2021-06-14", "thread", "built in 1887")]
    state = {"messages": [HumanMessage(content="q"), _tool_msg(hits1), AIMessage(content="draft"),
                          _tool_msg(hits2, start=1)]}
    ev = collect_evidence(state)
    assert [e.id for e in ev] == [1, 2]
    assert ev[0].source_class == "encyclopedia"
    assert ev[1].source_class == "forum"


def test_collect_evidence_dedups_by_url():
    hit = [SearchHit("https://encyclopedia.example/e", "encyclopedia.example",
                     "2019-04-11", "Eiffel", "completed in 1889")]
    state = {"messages": [_tool_msg(hit), _tool_msg(hit)]}
    assert len(collect_evidence(state)) == 1


def test_collect_evidence_ignores_access_denied():
    state = {"messages": [ToolMessage(content="Access denied: 'confidential' not permitted.",
                                      name="search", tool_call_id="x")]}
    assert collect_evidence(state) == []


def test_collect_evidence_dedups_by_title_when_url_empty():
    h1 = [SearchHit("", "internal.corp", None, "Audit", "first content")]
    h2 = [SearchHit("", "internal.corp", None, "Audit", "second content")]
    state = {"messages": [_tool_msg(h1), _tool_msg(h2)]}
    assert len(collect_evidence(state)) == 1


def test_collect_evidence_ignores_non_search_tool():
    hits = [SearchHit("https://a.example/1", "a.example", "2020-01-01", "A", "alpha content")]
    state = {"messages": [ToolMessage(content=format_hits(hits), name="rank_sources", tool_call_id="x")]}
    assert collect_evidence(state) == []


def _ev(eid, domain, source_class):
    return Evidence(eid, f"https://{domain}/x", domain, source_class, None, "t", "content")


def test_extract_claims_parses_citations(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {
        "claims": [{"text": "Eiffel completed 1889", "cited_evidence_ids": [1, "3"]}]
    })
    claims = extract_claims("draft [1][3]", [_ev(1, "encyclopedia.example", "encyclopedia")])
    assert claims[0].text == "Eiffel completed 1889"
    assert claims[0].cited_evidence_ids == [1, 3]


def test_score_claims_builds_stance_matrix(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {
        "matrix": {"0": {"1": "supports", "2": "contradicts"}}
    })
    claims = [Claim("c", [])]
    ev = [_ev(1, "encyclopedia.example", "encyclopedia"), _ev(2, "forum.example", "forum")]
    matrix = score_claims(claims, ev)
    assert matrix[0][1] == Stance.SUPPORTS
    assert matrix[0][2] == Stance.CONTRADICTS


def test_score_claims_short_circuits_without_evidence():
    assert score_claims([Claim("c", [])], []) == {0: {}}


def test_finalize_returns_final_answer(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {"final_answer": "Final text. Sources: [1]"})
    v = ClaimVerdict(Claim("c", []), {1: Stance.SUPPORTS}, 1.0, 0.0, "verified")
    out = finalize("draft", [v], [_ev(1, "encyclopedia.example", "encyclopedia")])
    assert out == "Final text. Sources: [1]"
