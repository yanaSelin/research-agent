# Architecture — hw-04-research-agent

## Problem

LLM-powered research agents have two structural vulnerabilities:
1. Access control delegated to the LLM can be bypassed by prompt manipulation (LLM06).
2. Without a verification step, agents draft cited answers whose sources are never checked — users
   cannot assess which claim is actually trustworthy (LLM09).

The `main` branch is the vulnerable baseline; it demonstrates both vulnerabilities with a
reproducible attack harness. This branch (`mitigated`) adds code-level fixes — collection ACL
and domain blocking enforced in the search closure, plus a post-draft claim verification pipeline.

---

## System Design

```
                         User query + role + mode
                                    │
                                    ▼
                    build_agent(role, mode)
                    ── system prompt: "confidential = admin only"
                       (context for the LLM; not relied upon for enforcement)
                                    │
                                    ▼
                     ResearchAgent (LangGraph ReAct loop)
                                    │
                                    ▼
                     search(query, collection)
                     ── make_search(role, mode) closure ──
                        [1] collection ACL from config/policy.json
                            basic role → "Access denied" in code
                        [2] domain filter from config/policy.json
                            blocked hits dropped before tool returns
                                    │
                    ┌───────────────┴──────────────────┐
                    ▼                                   ▼
        collection='web'                   collection='confidential'
        LocalCorpusBackend                 InternalKBBackend
        data/web_corpus.json               data/internal_kb.json
        (deterministic; eval/tests)        (admin role only — code-enforced)
                                    │
                    domain-blocked hits stripped here — never reach the agent
                                    │
                                    ▼
                    Agent drafts a cited answer
                                    │
                                    ▼
                    verify_pipeline(state, draft)
                      1. collect_evidence  — walk ToolMessages → Evidence
                                            (source_class per domain from config)
                      2. extract_claims    — 1 LLM call: claim list from draft
                      3. score_claims      — 1 LLM call: support/contra matrix
                      4. classify         — PURE CODE: trust-weight sums vs
                                            config/verifier.json thresholds
                      5. finalize         — 1 LLM call: rewrite draft per verdicts
                                    │
                                    ▼
                    Finalized answer: contested/unsupported claims flagged
```

---

## Security Approach: Code-Level Enforcement

Security is enforced in code, not in natural language instructions to the LLM. The prompt still
names the rules (so the model understands the context), but the enforcement path is independent of
whether the LLM follows those instructions.

**1. Collection ACL in the `make_search` closure (DD-03).**
`make_search(role, mode)` reads `config/policy.json` at build time and bakes the permitted
collection set into the returned tool. When the LLM calls `search(collection='confidential')` for
a `basic` role, the tool returns `"Access denied"` before touching `InternalKBBackend` — there is
no code path the LLM can take to widen this. On `main`, the same guard was a sentence in the
system prompt; a persuasive multi-turn attacker bypassed it in 2/3 runs.

**2. Domain filtering in the same closure.**
`blocked_domain_patterns` from `config/policy.json` are applied inside the tool via `fnmatch`
before the hit list is returned. The LLM never sees content from blocked domains — they are
stripped before the tool's output is assembled. On `main`, there was no code path for domain
filtering at all; the domain-block attack succeeded 3/3 runs.

**3. Post-draft claim verification pipeline (DD-04).**
After the agent drafts an answer, `verify_pipeline` runs five steps: evidence collection from tool
messages, claim extraction, claim-evidence scoring, pure-code classification (step 4 has no LLM
involvement — trust-weight arithmetic against `config/verifier.json` thresholds), and a finalizing
rewrite. Claims that are contested or unsupported are flagged or removed from the final answer.
This directly addresses LLM09: the agent can receive 4 myth pages and 1 encyclopedia page and
still have the myth-derived claims marked `contested` when the encyclopedia's trust weight
outweighs the myth majority.

**4. Trust is a property of source class, not source identity.**
Source class (encyclopedia, news_reputable, official, blog, forum, unknown) is assigned by domain
lookup in `config/source_classes.json`. An unknown domain defaults to class `unknown` (trust
weight 0.2) — safe by default, no allowlist needed. The classification step sums weights, so a
plurality of low-trust sources cannot override a single high-trust source.

**5. Deterministic enforcement.**
Steps 1–2 (ACL, domain filter) are pure synchronous code: no LLM, no sampling, no temperature.
Step 4 (classify) is also pure code. Only steps 2, 3, 5 of the verify pipeline involve LLM calls
(claim extraction, scoring, and rewriting), but the verdict is determined by code. This means the
security decision is version-stable and auditable, unlike a prompt instruction that may degrade
with model updates.

**6. Externalized policy (DD-06).**
Access rules, domain block lists, source-class trust weights, and verifier thresholds all live in
`config/*.json`. Adding a new blocked domain, adjusting a trust weight, or tightening a threshold
is a data change — no code deployment required.

**7. Audit trail.**
`collect_evidence` preserves which search hits drove the draft — source URL, domain, source class,
and content — before the LLM rewrites anything. The claim-evidence matrix (step 3) maps each claim
to its supporting and contradicting evidence. Both are logged at DEBUG level and available via
`AgentResult.verdicts` and `AgentResult.evidence` returned by `run_agent`.

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

| File | Contents | Who reads it |
|------|----------|-------------|
| `config/policy.json` | Role → permitted collections + blocked domain patterns | `make_search` closure |
| `config/source_classes.json` | Domain → source class; trust weight per class | `collect_evidence` |
| `config/verifier.json` | `T_support_ok`, `T_contra_veto`, `min_domain_class_weight` | `classify` (pure code) |

On `main`, these files exist but are not wired to enforcement logic. On this branch they are the
sole source of truth for ACL, domain filter, and claim-verdict thresholds.

---

## Key Design Decisions

### DD-01: Committed static corpus with scenario-driven canaries (ASR-03)
`data/web_corpus.json` is a committed, static JSON file — not generated at runtime. It contains
15 canary entries (tagged `easy`/`uncontested`/`trap`) hand-curated across source classes
(encyclopedia, news, forum, blog) and two decoy pages on `*.internal.example` / `*.corp.example`.
The canary mix (4 myth pages + 1 encyclopedia per trap topic) is fixed: it does not drift with
model updates, external dataset changes, or API availability. `pytest` and `eval/attacks.py` run
fully offline against this frozen file with no LLM calls required to reproduce the corpus.

Tradeoff: Updating a canary requires editing the JSON and the accompanying `data/scenarios.py`
definition by hand, rather than regenerating from a prompt template. Accepted — determinism and
offline reproducibility outweigh the convenience of auto-generation for an evaluation corpus.

### DD-02: `SearchBackend` abstraction + role-from-code (ASR-01, ASR-02, ASR-04)
`SearchBackend` is a `Protocol` (`search(query, k) -> list[SearchHit]`), decoupling retrieval from
role/collection concerns entirely — a backend knows nothing about ACL. `LocalCorpusBackend` reads
the committed `data/web_corpus.json` (keyword-scored, deterministic), used for all eval runs.
`InternalKBBackend` serves the `confidential` collection from `data/internal_kb.json`.

The abstraction keeps eval runs network-free and deterministic, and allows a future live-web
backend without touching the agent or ACL logic. `mode` is the extension point:
`pick_web_backend(mode)` in `src/backends/__init__.py` maps the mode string to an implementation.
Currently only `"local"` is implemented; a future `"live"` mode would return a `TavilyBackend`.

Tradeoff: an extra abstraction layer versus inline retrieval calls, but it is what keeps any future
ACL closure clean — the backend has no awareness of role or collection policy.

### DD-03: Closure-based collection ACL and domain blocking (ASR-01, ASR-02)
`make_search(role, mode)` reads `config/policy.json` once at agent-build time and closes over
`allowed_collections` and `blocked_domain_patterns`. The returned `search` tool enforces both
before touching any backend:

```python
if collection not in allowed_collections:
    return f"Access denied: {collection!r} not permitted for role {role!r}."
hits = [h for h in hits if not _domain_blocked(h.domain, blocked)]
```

Closing over the policy rather than passing `role` as a parameter means the LLM cannot pass a
different role to widen access. The role is set once at application layer startup and never
re-evaluated inside a request.

Tradeoff: The closure couples ACL logic to the tool factory, but this is intentional — it is the
design property that makes ACL LLM-proof. An alternative middleware approach (post-retrieval
filtering) would let the LLM retrieve all hits first, increasing backend load and leaving a
window for prompt-injected content in the raw results.

### DD-04: Claim verification pipeline — pure-code classification (ASR-03, ASR-07)
The verify pipeline has five steps: collect evidence → extract claims → score claims → classify →
finalize. Step 4 (`classify` in `src/verify/policy.py`) is pure arithmetic — no LLM call:

```
verdict = 'verified'    if sum(support weights) ≥ T_support_ok
          'contested'   if sum(contra weights)  ≥ T_contra_veto
          'unsupported' otherwise
```

`T_support_ok`, `T_contra_veto`, and `min_domain_class_weight` come from `config/verifier.json`.
Because the verdict gate is code, it is deterministic and auditable: the same claim with the same
evidence matrix always produces the same verdict regardless of model version or temperature.

Tradeoff: Steps 2, 3, and 5 still involve LLM calls (claim extraction, scoring, and rewriting),
so the pipeline adds ~3 inference calls and latency per agent turn. Acceptable for the eval
harness; would need caching or batching for a production path under real load.

### DD-05: Verdict-correctness as the LLM09 eval metric (ASR-03, ASR-04)
The primary eval metric for LLM09 is whether the verifier classifies trap-canary claims as
`contested` with the encyclopedia source cited — not whether the raw draft was factually correct.
This is a stronger test: it verifies that the pipeline, not just the model, is catching the
myth-majority pattern.

On `main` (no verifier), the metric degrades to checking whether the encyclopedia's claim appears
anywhere in the raw draft. On this branch, the metric checks the verifier's verdict directly via
`AgentResult.verdicts` from `run_agent`.

### DD-06: Externalized policy, source-class trust, and verifier thresholds (ASR-05, ASR-06, ASR-07)
Access rules, domain block lists, source-class trust weights, and claim-verdict thresholds all live
in `config/*.json` and are loaded once at startup via `src/config.py`. This means:

- Blocking a new internal domain = add it to `policy.json`, no code change.
- Adjusting how much an encyclopedia outweighs a forum = edit `source_classes.json`.
- Tightening the support threshold so fewer claims pass as `verified` = edit `verifier.json`.

Security policy changes do not require a code deployment, only a config update + process restart.

Tradeoff: Config files are on disk, so a misconfigured deployment can silently weaken security.
The mitigation is that `make_search` logs the resolved `allowed_collections` at INFO level on
startup — operators can verify it without inspecting the raw JSON.

---

## Observability

### Logging

Structured Python logging throughout (`logging.getLogger(__name__)`), configured in
`src/logging_config.py`. Level is set via `LOG_LEVEL` env var (default `INFO`); set `DEBUG`
for full per-step detail. Logs go to `logs/agent.log` (or `logs/eval.log` during eval runs).

**INFO** — step boundary events: query, collection, ACL decisions, hit counts after domain
filter, verify-pipeline step completions, verdict counts, final answer length.

**DEBUG** — payload content at each step: hit content, extracted claims, score matrix, claim
verdicts, rewritten answer. Full audit trail for diagnosing wrong answers without `print` statements.

**WARNING** — recoverable anomalies: malformed hits, `collect_evidence` parsing failures,
empty claim lists.

### Audit trail via `AgentResult`

`run_agent` returns `AgentResult` with `verdicts: list[ClaimVerdict]` and `evidence: list[Evidence]`.
Each `ClaimVerdict` carries the original claim text, verdict, supporting and contradicting evidence
IDs, and weights. This gives full traceability from final answer back to source document per claim.

### LangSmith (opt-in)

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`. LangGraph auto-instruments
every agent node, tool call, and LLM invocation with per-step latency and token usage.
The verify pipeline's three LLM calls appear as separate spans, so it is easy to see whether
claim extraction, scoring, or finalization is the latency bottleneck.

---

## Evaluation

### Metric Design

**LLM06 — exploited count (n/3).** A binary "did the attack work" per run rather than an
accuracy rate: a single successful bypass is a confirmed vulnerability. Three runs separates
three qualitatively different outcomes: 0/3 (code-level enforcement holds), 3/3 (consistent
exploit — ACL or domain filter is broken), and 1–2/3 (flaky — also a regression, since
deterministic code should never occasionally fail). Running fewer cannot distinguish these states;
running more increases LLM call cost without adding a qualitatively different signal.

**LLM09 — verdict-correctness (n/7 canaries).** On this branch the metric is not "did the model
state the correct fact" but "did the verifier classify the claim as `contested` with the
encyclopedia source cited." A lucky correct answer without the verdict does not count: it means
the pipeline happened to get the right output for the wrong reason. This tests the verifier's
decision logic, not the drafting model's knowledge. The 4:1 myth-to-encyclopedia ratio per topic
ensures the myth majority is a plausible signal — the verifier must outweigh it via trust weights,
not avoid it via corpus imbalance.

**DeepEval Faithfulness + AnswerRelevancy (threshold 0.7).** Same purpose as on `main`: confirm
that quality scores remain high while security failures are eliminated, separating quality from
security as independent axes.

See `README.md` for measured results.

### Residual risks

| Risk | Why it remains |
|------|---------------|
| LLM06 SE model-layer leakage | ACL blocks the tool call, but cannot prevent the model from referencing content already in context from a prior admin turn |
| LLM09 novel domains | Unseen domains default to `unknown` (trust weight 0.2); support thresholds are harder to reach with all-unknown evidence |
| Claim extraction gaps | If the LLM misses a claim in step 2, it never enters the verdict pipeline — unsupported claims can survive if not extracted |
| Unchecked finalize rewrite | Step 5 rewrites the draft via LLM; the rewrite itself is not re-verified |
