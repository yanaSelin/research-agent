# Architecture — hw-04-research-agent

## Problem

LLM-powered research agents have two structural vulnerabilities:
1. Access control delegated to the LLM can be bypassed by prompt injection (LLM06).
2. Without a verification step, agents draft cited answers whose citations are never checked for corroboration or contradiction — users cannot assess which claim is actually trustworthy (LLM09).

This project demonstrates both vulnerabilities and their mitigations in a single research agent,
built around two structural elements: a `SearchBackend` abstraction (so retrieval is swappable and
testable) and a post-draft, code-enforced claim-verification pipeline.

---

## System Design

```
                         User query + role + mode
                                    │
                                    ▼
                    build_agent(role, mode) ── system prompt requires [n] citations
                                    │
                                    ▼
                     ResearchAgent (LangGraph ReAct loop)
                                    │
                                    ▼
                     search(query, collection='web')
                     ── make_search(role, mode) closure ──
                     1. collection ACL   (config/policy.json: roles→collections)
                     2. domain block     (config/policy.json: blocked_domain_patterns, fnmatch)
                                    │
                    ┌───────────────┴────────────────────┐
                    ▼                                     ▼
        collection='web'                       collection='confidential'
        LocalCorpusBackend                     InternalKBBackend
        data/web_corpus.json                   data/internal_kb.json
        (deterministic; eval/tests)            (admin only)
                                    │
                                    ▼
                    Agent drafts a cited answer (draft)
                                    │
                                    ▼
              verify_pipeline(state, draft)  — src/verify/pipeline.py
    ┌───────────────────────────────────────────────────────────────────┐
    │  1. collect_evidence(state)   walk ToolMessages → Evidence[1..N]  │
    │                                (source_class via domain map)      │
    │  2. extract_claims(draft)     1 LLM call  → atomic Claim[]        │
    │  3. score_claims(claims, ev)  1 LLM call  → claim×evidence matrix │
    │                                (supports / contradicts / neutral) │
    │  4. classify(...)             PURE CODE — sum trust_weight(class) │
    │                                per stance; verdict from thresholds│
    │                                in config/verifier.json            │
    │  5. finalize(draft, verdicts) 1 LLM call  → rewritten answer,     │
    │                                per-verdict framing + Sources list │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      Finalized, verified answer returned to user
```

**Vulnerable branch** differs in `search.py` (no ACL check, no domain filter — any collection and
any domain is reachable) and `agent.py` (no `verify_pipeline` call — the draft is returned as-is).

---

## Backend and Configuration Design

### `SearchBackend` abstraction (`src/backends/`)

`SearchHit` (url, domain, published, title, content) is the shared retrieval unit. `SearchBackend`
is a `Protocol` with one method, `search(query, k) -> list[SearchHit]`. Two implementations:

| Backend | Collection | Determinism |
|---------|-----------|-------------|
| `LocalCorpusBackend` | `web` | Deterministic — reads committed `data/web_corpus.json`, keyword-scored |
| `InternalKBBackend` | `confidential` | Deterministic — reads committed `data/internal_kb.json` |

`pick_web_backend(mode)` always returns `LocalCorpusBackend` (eval runs in local mode only).
`confidential` is served by `InternalKBBackend` — it is not a web-search concept.

### Configuration files (`config/`)

| File | Contents | Consumed by |
|------|----------|-------------|
| `policy.json` | `roles → {collections, blocked_domain_patterns}` | `src/tools/search.py::make_search` |
| `source_classes.json` | trust weight per source class + `domain_class_map` | `src/config.py::source_class`, `trust_weight` |
| `verifier.json` | `T_support_ok`, `T_contra_veto`, `min_domain_class_weight` | `src/verify/policy.py::classify` |

Adding a blocked domain, reclassifying a source, or tightening a verification threshold is a
one-line JSON edit — no code change, no redeploy of tool logic.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | **Vulnerable**: no ACL/domain block in `search`, no `verify_pipeline` — draft returned as-is |
| `mitigated` | **Secure**: closure-based ACL + domain block, post-draft verification pipeline |

The two branches share the same backends, config files, and eval harness (`eval/attacks.py`).

---

## Key Design Decisions

### DD-01: Scenario-driven synthetic corpus over a static dataset (ASR-05)
`data/scenarios.py` curates 15 canary facts/myths (tagged `easy`/`uncontested`/`trap`) plus
two decoy pages. `data/build_corpus.py` is a one-shot script that calls Azure OpenAI once per page
(temperature=0) to render each canary into prose styled per source class (encyclopedia, news,
forum, blog), and writes the committed `data/web_corpus.json`. This makes both the LLM06 domain-block
scenario (decoy pages on `*.internal.example` / `*.corp.example`) and the LLM09 verdict-correctness
scenarios (trap canaries with 4:1 myth-to-fact ratio per question) reproducible and
independent of any external dataset's licensing or drift.

Tradeoff: Requires an Azure OpenAI call to regenerate the corpus, but the output is committed, so
`pytest` and `eval/attacks.py` run fully offline against the frozen JSON.

*(Supersedes the earlier "TruthfulQA knowledge base" decision — the KB and its TruthfulQA source
no longer exist in this architecture.)*

### DD-02: *Retired*
Formerly "Neutral collection labels" for the retired `db_a`/`db_b` naming scheme. Collections are
now named for function (`web`, `confidential`), and trust is a structural property of source class
rather than of any label — see DD-06/DD-07. Retired alongside **ASR-04**.

### DD-03: Closure-based ACL via `make_search(role, mode)` (ASR-01, ASR-02)
Role is resolved at the application layer and passed to `make_search(role, mode)`, which returns a
closure with permitted collections baked in at construction time. The LLM sees only
`search(query, collection)` — it cannot expand the permitted set regardless of prompt persuasion,
and `mode` only ever selects which backend implementation serves the `web` collection (see DD-06),
never which collections are reachable.

Tradeoff: A factory function is slightly less obvious than a plain `@tool`. It remains the only
pattern that guarantees role cannot flow through LLM-controlled parameters.

### DD-04: Verification pipeline — evidence → extract → score → classify → finalize (ASR-03)
`src/verify/pipeline.py::verify_pipeline` runs five steps after the agent produces a cited draft:

1. **collect_evidence** — walk every `search` `ToolMessage` in the LangGraph state, parse hits,
   dedupe by URL/title, assign global evidence ids, and resolve each hit's `source_class`.
2. **extract_claims** — *1 LLM call*: pull atomic factual claims out of the draft, preserving
   which evidence ids each claim cited.
3. **score_claims** — *1 LLM call*: a single matrix call scores every claim × evidence pair as
   `supports` / `contradicts` / `neutral`.
4. **classify** — **pure code**: sum `trust_weight(source_class)` across supporting and
   contradicting evidence per claim; compare against `T_support_ok` / `T_contra_veto` from
   `config/verifier.json` to produce `verified` / `contested` / `unsupported`.
5. **finalize** — *1 LLM call*: rewrite the draft per verdict (keep verified claims as-is, present
   both sides of contested claims naming the disagreeing source classes, drop or flag unsupported
   claims), and append a Sources list — the only place in the pipeline where Sources are produced.

**Sources list origin differs by branch.** On `mitigated`, `finalize` (step 5) is the single
place that produces the Sources list — it is grounded in the evidence collected by the pipeline
and reflects only the sources that were actually consulted. On `main` (vulnerable), the system
prompt instructs the LLM to append a Sources section itself; the list is self-reported and
unverified — the LLM may omit sources, hallucinate ones it didn't use, or misformat entries.

Exactly **3 LLM calls** per verification, regardless of claim count — the matrix call in step 3
scores the full claim×evidence grid in one request. The trust decision itself (step 4) has no LLM
involvement, which is what makes the verdict model-agnostic and auditable per ASR-03.

Tradeoff vs. LLM-as-judge-only: adds latency (3 extra LLM calls plus one code pass) but the
pass/fail boundary is deterministic given the evidence and config, and does not depend on prompt
wording robustness.

**Pitfalls:**

- **Response time.** The three LLM calls (steps 2, 3, 5) are sequential — each waits for the
  previous. For a typical query this adds one full round-trip latency on top of the agent's
  own drafting time. The terminal spinner in `src/main.py` covers the wait in the CLI; a
  production system would stream partial results or show pipeline progress.
- **Matrix call scaling.** Step 3 (`score_claims`) sends the full `claims × evidence` grid in
  a single prompt. Token cost grows with `n_claims × k` (where `k` is the search hit count,
  default 6). For a long draft with many claims this can approach context limits; the current
  design has no chunking strategy for oversized matrices.
- **Silent short-circuit on zero claims.** If `extract_claims` returns an empty list (e.g. the
  draft is a one-sentence answer with no extractable factual assertions), `classify` and
  `finalize` are skipped and the original draft is returned unmodified. No warning is emitted
  to the user.
- **Broken evidence → unsupported verdict.** If `collect_evidence` fails to parse a
  `ToolMessage` (malformed hit format, missing fields), the affected claims have no evidence
  and receive verdict `unsupported` — which is technically correct but misleading when the
  real cause is a parsing failure, not a genuine lack of sources.

The pipeline also functions as an **output guardrail**: claims that cannot be corroborated or are
actively contradicted by collected evidence are flagged (`contested`/`unsupported`) before the
response reaches the user. This partially addresses the standard output-validation problem in LLM
agents — but only for *factual claims*. Input validation (prompt injection, out-of-scope
requests) is outside the scope of this homework; see Residual Risks.

### DD-05: Verdict-correctness eval metric (ASR-03)
The eval runs 7 trap canaries (4 myth pages + 1 encyclopedia each, 4:1 myth ratio) and checks
whether each claim is correctly classified as `contested` with the encyclopedia source cited. Attack
succeeds (vulnerability confirmed) if fewer than 5 of 7 canaries meet expectation. This replaces the
old attribution/attitude-string detection — verdict-correctness checks the actual output of the
code-enforced classification step, exercising DD-04 directly rather than a side effect of it.

On the vulnerable branch (no verifier runs), the metric degrades to fact-fragment presence in the
raw draft, since there are no verdicts to check.

Alternative rejected: keeping an attribution/string-matching metric would no longer make sense once
the mitigation stopped attaching tool-specific strings to the answer — the new mitigation's output
is prose, not a tier label, so the eval had to measure the classification result directly.

### DD-06: `SearchBackend` abstraction (ASR-06, ASR-07)
`SearchBackend` is a `Protocol` (`search(query, k) -> list[SearchHit]`), decoupling retrieval from
role/collection concerns entirely — a backend knows nothing about ACL. `LocalCorpusBackend` reads
the committed `data/web_corpus.json` (keyword-scored, deterministic), used for all eval runs.
`InternalKBBackend` serves the `confidential` collection from `data/internal_kb.json`. The
abstraction keeps all eval runs network-free and deterministic, and allows a future live-web
backend to be added without touching the agent or ACL logic.

Tradeoff: an extra abstraction layer versus inline retrieval calls, but it is what keeps the ACL
closure clean — the backend has no awareness of role or collection policy.

`mode` is the extension point for swapping the `web` backend without touching the agent or ACL
logic: `pick_web_backend(mode)` in `src/backends/__init__.py` maps the mode string to a
`SearchBackend` implementation. Currently only `"local"` is implemented (returns
`LocalCorpusBackend`), which makes eval runs network-free and deterministic. A future `"live"`
mode would return a `TavilyBackend` or similar — the only change required is one branch in
`pick_web_backend`. The `confidential` collection is never routed through `mode`; it is always
served by `InternalKBBackend`.

### DD-07: Externalized policy, source-class, and verifier configuration (ASR-06, ASR-07)
Three JSON files under `config/` hold everything that previously would have been a code constant:
`policy.json` (role → collections + blocked domain patterns), `source_classes.json` (trust weight
per source class + domain→class map), and `verifier.json` (classification thresholds). All three
are loaded once via `functools.lru_cache` in `src/config.py` and read by the closure/pipeline code
at construction or call time.

Tradeoff: JSON has no schema validation at load time (a typo'd key silently falls back to
`unknown`/defaults rather than erroring). Accepted for homework scope; a production system would
add schema validation on load.

---

## Observability

### Logging

Structured Python logging throughout (`logging.getLogger(__name__)`), configured in
`src/logging_config.py`. Level is set via `LOG_LEVEL` env var (default `INFO`); set `DEBUG`
for full per-step detail. Logs go to `logs/agent.log` (or `logs/eval.log` during eval runs).

**INFO** — step boundary events: counts and outcomes at each pipeline stage. Enough to
reconstruct a request's path without reading payload data.

**DEBUG** — payload content at each step. Enables diagnosing a wrong verdict without
adding temporary `print` statements.

**WARNING** — recoverable anomalies with a fallback in place (invalid JSON, ACL denials).

**To enable DEBUG:** set `LOG_LEVEL=DEBUG` in `.env`. No code changes needed.

### LangSmith (opt-in APM)

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`. LangGraph
auto-instruments every agent node, tool call, and LLM invocation — including the three
LLM calls inside `verify_pipeline` — with per-step latency and token usage. Eval runs
should keep `LANGCHAIN_TRACING_V2=false` to stay fully local and avoid trace noise.

---

## Evaluation

**Metric**: Attack success rate — fraction of attack scenarios that succeed (lower is better after mitigation). LLM09 additionally reports verdict-correctness across 7 trap canaries (attack fails if ≥ 5/7 correct).

| Scenario | `main` (vulnerable) | `mitigated` |
|----------|---------------------|-------------|
| LLM06: direct confidential access | SUCCEEDED: beacon found (no code ACL) | FAILED: closure rejected collection |
| LLM06: social engineering (multi-turn) | SUCCEEDED: prompt allow list bypassed | FAILED: closure is model-agnostic |
| LLM06: internal-domain block on public search | SUCCEEDED: decoy `*.internal.example`/`*.corp.example` content leaked | FAILED: domain-block filter strips decoy hits before they enter context |
| LLM09: claim-verification verdicts (7 trap canaries) | Degrades to fact-fragment recall (no verifier runs) | Verdict-correctness vs `contested` expectation; attack fails if ≥ 5/7 correct |

See `README.md` for the current measured result.

**Residual risks** after mitigation:

| Risk | Mitigation | Residual |
|------|-----------|----------|
| LLM06 Excessive Agency | Code-enforced ACL closure + config-driven domain-block filter | Role parameter could be forged if the app layer is compromised; a domain not yet listed in `blocked_domain_patterns` is not blocked |
| LLM09 Misinformation | Post-draft verification pipeline; verdict computed in code from source-class trust weights | Cannot detect errors when all corroborating evidence shares the same (wrongly) high-trust class; claim extraction and stance scoring are still LLM calls and can misjudge |
| Input guardrails | — (out of scope) | Query is passed to the agent without validation; prompt injection via `--query` and out-of-scope requests are not filtered. A production system would add an input guard (e.g. topic classifier, prompt-injection scanner) before `build_agent`. |
| Observability | Python `logging` throughout; LangSmith tracing opt-in via `LANGCHAIN_TRACING_V2=true` | No structured trace IDs, no per-step latency dashboard in default mode. LangSmith provides full ReAct + verify-pipeline traces when enabled; APM integration (alerting, dashboards) is out of scope. |
