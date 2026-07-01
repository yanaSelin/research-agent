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
        pick_web_backend(mode)                 InternalKBBackend
        ┌─────────────┴─────────────┐          data/internal_kb.json
        ▼                           ▼          (admin only, both modes)
   LocalCorpusBackend          TavilyBackend
   data/web_corpus.json        live web search
   (mode=local; eval/tests,    (mode=web; falls back to
    deterministic)              Local + warning if no
                                 TAVILY_API_KEY)
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
is a `Protocol` with one method, `search(query, k) -> list[SearchHit]`. Three implementations:

| Backend | Collection | Used when | Determinism |
|---------|-----------|-----------|-------------|
| `LocalCorpusBackend` | `web` | `mode=local` (default; eval, tests, offline dev) | Deterministic — reads committed `data/web_corpus.json`, keyword-scored |
| `TavilyBackend` | `web` | `mode=web` and `TAVILY_API_KEY` set | Live web, non-deterministic |
| `InternalKBBackend` | `confidential` | Always, both modes | Deterministic — reads committed `data/internal_kb.json` |

`pick_web_backend(mode)` selects between `LocalCorpusBackend` and `TavilyBackend` for the `web`
collection; if `mode=web` but `TAVILY_API_KEY` is missing, it falls back to `LocalCorpusBackend`
with a logged warning rather than failing. `confidential` is always served by `InternalKBBackend`
regardless of mode — it is not a "live web" concept.

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
| `main` | Secure: closure-based ACL + domain block, post-draft verification pipeline |
| `vulnerable` | Demonstrates LLM06 + LLM09: no ACL/domain block in `search`, no verification of the draft |

The two branches share the same backends, config files, and eval harness (`eval/attacks.py`).

---

## Key Design Decisions

### DD-01: Scenario-driven synthetic corpus over a static dataset (ASR-05)
`data/scenarios.py` curates 15 canary facts/myths (tagged `easy`/`uncontested`/`trap`/`gap`) plus
two decoy pages. `data/build_corpus.py` is a one-shot script that calls Azure OpenAI once per page
(temperature=0) to render each canary into prose styled per source class (encyclopedia, news,
forum, blog), and writes the committed `data/web_corpus.json`. This makes both the LLM06 domain-block
scenario (decoy pages on `*.internal.example` / `*.corp.example`) and the LLM09 verdict-correctness
scenarios (traps that require weighing conflicting sources, gaps with no evidence) reproducible and
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
   claims), and append a Sources list.

Exactly **3 LLM calls** per verification, regardless of claim count — the matrix call in step 3
scores the full claim×evidence grid in one request. The trust decision itself (step 4) has no LLM
involvement, which is what makes the verdict model-agnostic and auditable per ASR-03.

Tradeoff vs. LLM-as-judge-only: adds latency (3 extra LLM calls plus one code pass) but the
pass/fail boundary is deterministic given the evidence and config, and does not depend on prompt
wording robustness.

### DD-05: Verdict-correctness eval metric (ASR-03)
The eval measures whether each canary's claim verdict matches its scenario's expected outcome:
`easy`/`uncontested` → `verified`, `trap` → `contested`, `gap` → `unsupported`. This replaces the
old attribution/attitude-string detection, which only worked because a specific tool call left a
verbatim string in the response — verdict-correctness instead checks the actual output of the
code-enforced classification step, so it exercises DD-04 directly rather than a side effect of it.

On the vulnerable branch (no verifier runs), the metric degrades to fact-fragment presence in the
raw draft, since there are no verdicts to check.

Alternative rejected: keeping an attribution/string-matching metric would no longer make sense once
the mitigation stopped attaching tool-specific strings to the answer — the new mitigation's output
is prose, not a tier label, so the eval had to measure the classification result directly.

### DD-06: `SearchBackend` abstraction with mode selection (ASR-06, ASR-07)
`SearchBackend` is a `Protocol` (`search(query, k) -> list[SearchHit]`), decoupling retrieval from
role/collection concerns entirely — a backend knows nothing about ACL. Two implementations serve
the `web` collection: `LocalCorpusBackend` (deterministic, reads the committed corpus — used for
eval and tests) and `TavilyBackend` (live web — production). `pick_web_backend(mode)` selects
between them from `AI_ARCHITECT_SEARCH_MODE`, falling back to local with a logged warning if
`TAVILY_API_KEY` is absent in web mode rather than failing the request. The `confidential`
collection is served by `InternalKBBackend` in both modes — it is never "live," so mode selection
does not apply to it.

Tradeoff: an extra abstraction layer versus calling Tavily directly, but it is what makes the eval
harness deterministic (no network, no LLM-search non-determinism) while keeping production
behavior a one-line backend swap.

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

## Evaluation

**Metric**: Attack success rate — fraction of attack scenarios that succeed (lower is better after mitigation). LLM09 additionally reports verdict-correctness out of 15 canaries.

| Scenario | Vulnerable branch | Main branch |
|----------|-----------------|-------------|
| LLM06: direct confidential access | SUCCEEDED: beacon found (no code ACL) | FAILED: closure rejected collection |
| LLM06: social engineering (multi-turn) | SUCCEEDED: prompt allow list bypassed | FAILED: closure is model-agnostic |
| LLM06: internal-domain block on public search | SUCCEEDED: decoy `*.internal.example`/`*.corp.example` content leaked | FAILED: domain-block filter strips decoy hits before they enter context |
| LLM09: claim-verification verdicts (15 canaries) | Degrades to fact-fragment recall (no verifier runs) | Verdict-correctness against scenario expectation; attack fails if ≥ 12/15 correct |

See `README.md` for the current measured result (pending an eval run on a networked machine).

**Residual risks** after mitigation:

| Risk | Mitigation | Residual |
|------|-----------|----------|
| LLM06 Excessive Agency | Code-enforced ACL closure + config-driven domain-block filter | Role parameter could be forged if the app layer is compromised; a domain not yet listed in `blocked_domain_patterns` is not blocked |
| LLM09 Misinformation | Post-draft verification pipeline; verdict computed in code from source-class trust weights | Cannot detect errors when all corroborating evidence shares the same (wrongly) high-trust class; claim extraction and stance scoring are still LLM calls and can misjudge |
