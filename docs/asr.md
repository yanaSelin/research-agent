# Architecturally Significant Requirements — hw-04-research-agent

## Overview

LangGraph ReAct research agent with a `SearchBackend` abstraction (local corpus / internal KB).
This branch (`main`) is the vulnerable baseline — it demonstrates LLM06 (Excessive Agency) and
LLM09 (Misinformation) with a reproducible attack harness. Security requirements and their
architectural decisions are documented on the `mitigated` branch.

What this branch gets right architecturally (and what the mitigated branch builds on top of):
role comes from the calling layer, not the LLM; retrieval is backend-swappable; the eval corpus
is deterministic and offline-runnable; every agent step is traceable.

---

## ASR Catalog

### ASR-01 — Role must be set by the calling application, not inferred from user input

- **Category**: Security
- **Priority**: High
- **Requirement**: The user's access level must be determined by the application layer before any
  LLM interaction. The model must not declare, confirm, or modify the role during conversation.
- **Rationale**: If role were derived from model output or user-supplied text, a prompt injection
  like "My role is admin" could escalate privileges with no code-level check to intercept it.
- **Maps to**: DD-02.

### ASR-02 — Retrieval backend must be swappable without modifying agent or ACL logic

- **Category**: Maintainability
- **Priority**: Medium
- **Requirement**: Switching from local corpus to a live search provider, or adding a new
  collection type, must require only a new `SearchBackend` implementation — no changes to
  the agent, the ACL closure, or the verification pipeline.
- **Rationale**: The agent and retrieval layer evolve at different rates. Coupling them forces
  simultaneous changes to security-sensitive code whenever the data source changes.
- **Maps to**: DD-02.

### ASR-03 — Attacks must be reproducible and runnable offline

- **Category**: Observability
- **Priority**: Medium
- **Requirement**: Both vulnerability demonstrations and mitigation tests must be deterministic
  and runnable via a single command, without network access or live-web retrieval.
- **Rationale**: An eval that depends on live search returns non-deterministic results — the same
  attack may succeed or fail based on which pages a live index returns, not on the security
  property being measured. Three runs per scenario is the minimum to distinguish consistent block
  (0/3), consistent exploit (3/3), and flaky (1–2/3).
- **Maps to**: DD-01.

### ASR-04 — Production issues must be diagnosable without live debugging

- **Category**: Observability
- **Priority**: Medium
- **Requirement**: Agent execution and tool calls must produce structured traces sufficient to
  diagnose misbehavior in a deployed environment without attaching a debugger or re-running
  the same query.
- **Rationale**: Log-only instrumentation captures that a tool call happened but not which LLM
  step triggered it or what the per-step latency was. Without structured traces, debugging
  requires reproducing the exact query and hoping the same retrieval path is followed.
- **Maps to**: DD-02.
