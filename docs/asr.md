# Architecturally Significant Requirements — hw-04-research-agent

## Overview

LangGraph ReAct research agent with a `SearchBackend` abstraction (local corpus / Tavily /
internal KB) and a post-draft claim-verification pipeline. Demonstrates two OWASP LLM Top 10
risks: LLM06 (Excessive Agency) and LLM09 (Misinformation).

---

## ASR Catalog

### ASR-01 — Access control must be enforced in code, not prompt
- **Category**: Security
- **Priority**: High
- **Requirement**: Collection access restrictions must be enforced by the tool implementation, not by LLM instructions.
- **Rationale**: Prompt-based access control is trivially bypassed. A user can override system prompt directives via social engineering; Python closure logic cannot be overridden from within the LLM context regardless of persuasion.
- **Architectural decision**: `make_search(role, mode)` returns a closure with permitted collections baked in at construction time. The LLM sees only `search(query, collection)` — it cannot expand the permitted set. Role is never extracted from LLM output.
- **Risk if violated**: LLM06 — Excessive Agency. Basic-role users can access the confidential collection by asking or social-engineering the model.
- **Maps to**: DD-03.

### ASR-02 — Role must not flow through LLM context
- **Category**: Security
- **Priority**: High
- **Requirement**: User role must be determined by the application layer and injected directly into tool construction, never inferred or extracted by the LLM.
- **Rationale**: If role flows through the prompt or tool parameters, an adversary can claim a different role in the message. The closure pattern eliminates this: role is resolved once by the application and locked into the function object.
- **Architectural decision**: `build_agent(role, mode)` accepts role from the application layer and passes it to `make_search(role, mode)`. The resulting tool's closure is immutable from the LLM's perspective.
- **Risk if violated**: LLM06 — privilege escalation through prompt manipulation or parameter injection.
- **Maps to**: DD-03.

### ASR-03 — Model-agnostic conflict resolution
- **Category**: Quality
- **Priority**: High
- **Requirement**: Source conflict resolution must be model-agnostic and code-enforced — the LLM must not be the arbiter of which claim is trustworthy.
- **Rationale**: LLM synthesis of conflicting sources is non-deterministic and unattributed. Running verification as a post-hoc, code-enforced step (rather than trusting the drafting model's judgment) makes the resolution predictable, auditable, and independent of prompt wording.
- **Architectural decision**: Verification runs **after** the draft, over the evidence collected across the whole conversation. The LLM is used only for perception (claim extraction, evidence↔claim stance judgement); the verdict itself is arithmetic on trust weights performed entirely in code (`src/verify/policy.py::classify`).
- **Risk if violated**: LLM09 — Misinformation. Agent blends conflicting sources without a auditable, code-computed verdict; user cannot determine which claim is authoritative.
- **Maps to**: DD-04, DD-05.

### ASR-04 — *Retired*
- Formerly "Source labels must not carry trust signals" (neutral `db_a`/`db_b` naming for the old `rank_sources` tier dictionary). The old collection-naming antipattern no longer exists — collections are now `web`/`confidential`, and trust is derived structurally from source class rather than from any label. Superseded by **ASR-06**.

### ASR-05 — Attacks must be reproducible and documented
- **Category**: Observability
- **Priority**: Medium
- **Requirement**: Both vulnerability demonstrations and mitigation tests must be deterministic and runnable via a single command.
- **Rationale**: Homework evaluation requires reproducible evidence of both attack success and attack failure after mitigation, without depending on network access or LLM non-determinism for the pass/fail signal itself.
- **Architectural decision**: `eval/attacks.py` runs 4 scenarios in local mode (no network): LLM06 direct request, LLM06 social-engineering (multi-turn), LLM06 domain-block, and LLM09 (15 canary questions from `data/scenarios.py`, seeded/fixed). Results written to `results/{mitigated,vulnerable}_attacks.json`. The LLM09 metric is verdict-correctness against each canary's expected scenario outcome (easy/uncontested → verified, trap → contested, gap → unsupported) — not answer-string matching — so the signal is branch-specific and unaffected by what the LLM already knows from pretraining. The vulnerable branch (no verifier) degrades this metric to fact-fragment presence.
- **Maps to**: DD-01, DD-05.

---

### ASR-06 — Trust derives from source class, not identity
- **Category**: Security / Quality
- **Priority**: High
- **Requirement**: A source's trust weight must be a function of its *class* (encyclopedia, news_reputable, official, blog, forum, internal, unknown), not a hand-maintained list of specific trusted domains. Unseen domains must default to a low-trust class rather than being trusted by omission.
- **Rationale**: A per-domain allowlist is unmaintainable and fails open on unseen domains (a new hostile domain not yet on any blocklist would otherwise be trusted). Classifying by structural properties of the source and defaulting unknowns to the lowest-trust class keeps the system safe by default in web mode, where the corpus is not curated.
- **Architectural decision**: `config/source_classes.json` defines trust weights per class and a `domain_class_map`. `src/config.py::source_class` looks up a domain's class; any domain absent from the map resolves to `unknown` (trust weight 0.2) — the lowest non-zero weight in the table.
- **Risk if violated**: LLM09 — Misinformation. An unclassified but influential source could be silently weighted as if it were authoritative.
- **Maps to**: DD-06, DD-07.

### ASR-07 — Access policy and domain blocks are configuration
- **Category**: Security / Maintainability
- **Priority**: High
- **Requirement**: Role→collection permissions and the blocked-domain patterns applied to web search results must be externalized as configuration, not hardcoded in the tool implementation.
- **Rationale**: Access rules and domain blocklists change more frequently than the code that enforces them (a new blocked domain, a new role, a quarterly policy review). Requiring a code change and redeploy for every policy edit is both slow and a needless source of review risk in security-sensitive logic.
- **Architectural decision**: `config/policy.json` defines `roles → {collections, blocked_domain_patterns}`. `src/tools/search.py::make_search(role, mode)` reads this policy at tool-construction time and applies both the collection ACL and an `fnmatch`-based blocked-domain filter to hits before they enter the LLM's context.
- **Risk if violated**: LLM06 — Excessive Agency. Policy changes would require code changes, increasing the chance that access rules drift out of sync with actual authorization intent.
- **Maps to**: DD-06, DD-07.

---

## Residual Risks After Mitigation

| Risk | Mitigation | Residual |
|------|-----------|----------|
| LLM06 Excessive Agency | Closure-based collection ACL + config-driven blocked-domain filter | Role parameter could be forged if the application layer is compromised; a domain not yet added to `blocked_domain_patterns` is not blocked |
| LLM09 Misinformation | Post-draft verification pipeline; verdict computed in code from source-class trust weights | Cannot detect errors when all corroborating evidence shares the same (wrongly) high-trust class; claim extraction and stance scoring are still LLM calls and can misjudge |
