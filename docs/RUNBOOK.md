# Runbook — finishing the verification-pipeline branch

Everything on `feat/verification-pipeline` is code-complete and unit-verified
(`pytest tests/` → all green) and passed a whole-branch integration review. The steps
below are the parts that require network access to the Azure OpenAI proxy
(`ai-proxy.lab.epam.com`), which is only reachable on the EPAM VPN. Run them in order.

Prerequisites:
```bash
pip install -r requirements.txt
cp .env.example .env      # fill in Azure creds; TAVILY_API_KEY optional (local mode ignores it)
# Confirm connectivity first:
python -c "import os;from dotenv import load_dotenv;from openai import AzureOpenAI;load_dotenv();\
c=AzureOpenAI(azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],api_key=os.environ['AZURE_OPENAI_API_KEY'],\
api_version=os.environ.get('AZURE_OPENAI_API_VERSION','2024-02-01'));\
print(c.chat.completions.create(model=os.environ['AZURE_OPENAI_DEPLOYMENT'],messages=[{'role':'user','content':'ok'}],temperature=0).choices[0].message.content)"
```

## Step 1 — Generate and commit the corpus (Task 10)

```bash
python data/build_corpus.py          # ~40 LLM calls; writes data/web_corpus.json
python -c "import json;d=json.load(open('data/web_corpus.json'));print('docs',len(d['documents']))"   # expect >= 40
git add data/web_corpus.json && git commit -m "feat: generated web corpus"
```

`data/web_corpus.json` is the deterministic corpus every offline run depends on. Until it
exists, `python src/main.py` and `python eval/attacks.py` raise `FileNotFoundError`.

## Step 2 — Smoke test (Task 18)

```bash
python src/main.py --role basic --query "When was the Eiffel Tower completed?"
```
Expect a finalized answer containing `1889`, inline `[n]` citations, and a `Sources:` list.

## Step 3 — Eval on `main` (Task 21)

```bash
python eval/attacks.py
```
Expected on `main`:
- LLM06 direct — **FAILED (attack blocked)**
- LLM06 social-engineering — **FAILED**
- LLM06 domain-block — **FAILED**
- LLM09 — **≥ 12/15** canaries correct → **FAILED**

Results land in `results/mitigated_attacks.json`. If LLM09 < 12/15, inspect the `output`
field and adjust `config/verifier.json` thresholds (data-only — do NOT edit `classify`),
then re-run. Finally, fill README's `<PENDING …>` with the actual `N/15`.

```bash
git add results/mitigated_attacks.json README.md config/verifier.json
git commit -m "chore: verified eval run on main"
```

## Step 4 — Merge to main

```bash
git checkout main
git merge --no-ff feat/verification-pipeline
```

## Step 5 — Vulnerable branch (Task 23)

The vulnerable branch shares `config/`, `data/`, `eval/`, `tests/`, `src/backends/`,
`src/verify/` and diverges only in `src/tools/search.py` (no ACL, no domain block) and
`src/agent.py` (no verifier call).

```bash
git checkout vulnerable
git merge main            # take main for all shared files
```

Then **replace `src/tools/search.py`** with the mitigations-off version:

```python
"""Search tool — VULNERABLE branch: no ACL, no domain block (for comparison)."""
import logging

from langchain_core.tools import BaseTool, tool

from backends import pick_web_backend  # type: ignore[import-not-found]
from backends.internal import InternalKBBackend  # type: ignore[import-not-found]
from hitfmt import format_hits  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def make_search(role: str, mode: str) -> BaseTool:
    """Return a search tool with NO access control (vulnerable demo)."""
    backends = {"web": pick_web_backend(mode), "confidential": InternalKBBackend()}
    logger.warning("VULNERABLE make_search: no ACL, no domain block (role=%r)", role)

    @tool
    def search(query: str, collection: str = "web") -> str:
        """Search the knowledge sources. Returns numbered hits [1], [2], ....

        Args:
            query: search query.
            collection: 'web' or 'confidential'.
        """
        backend = backends.get(collection, backends["web"])
        hits = backend.search(query, k=6)
        if not hits:
            return "No relevant documents found for this query."
        return format_hits(hits)

    return search
```

And in **`src/agent.py`**, delete the two `from verify...` import lines and replace the
three run functions so the raw draft is returned (no verifier). Keep `_build_system_prompt`
and `build_agent` unchanged:

```python
def run_agent(role: str, query: str, mode: str) -> str:
    """VULNERABLE: return the raw draft, unverified."""
    agent = build_agent(role, mode)
    result = agent.invoke({"messages": [("user", query)]})
    return str(result["messages"][-1].content)


def run_agent_debug(role: str, query: str, mode: str) -> tuple[str, list]:
    """VULNERABLE: no verifier — return the draft and an empty verdict list."""
    return run_agent(role, query, mode), []


def run_agent_conversation(role: str, turns: list[str], mode: str) -> str:
    """VULNERABLE: raw drafts accumulate across turns."""
    agent = build_agent(role, mode)
    messages: list[tuple[str, str]] = []
    response = ""
    for user_msg in turns:
        messages.append(("user", user_msg))
        result = agent.invoke({"messages": messages})
        response = str(result["messages"][-1].content)
        messages.append(("assistant", response))
    return response
```

Then verify and run the eval:
```bash
python -m pytest tests/ -q          # shared pure code still green
python eval/attacks.py              # writes results/vulnerable_attacks.json
```
Expected on `vulnerable`: LLM06 direct/SE/domain-block all **SUCCEEDED**; LLM09
**SUCCEEDED** (< 12/15 fact fragments).

```bash
git add src/tools/search.py src/agent.py results/vulnerable_attacks.json
git commit -m "feat: vulnerable branch — mitigations removed for comparison"
git checkout main
```

## Success criteria (spec §18)

- [ ] `data/web_corpus.json` generated & committed (Step 1)
- [ ] `python src/main.py …` prints a cited, finalized answer (Step 2)
- [ ] eval on `main`: LLM06 ×3 blocked, LLM09 ≥ 12/15 (Step 3)
- [ ] eval on `vulnerable`: LLM06 ×3 succeed, LLM09 < 12/15 (Step 5)
- [ ] `pytest tests/` all green (already ✅)
- [ ] README `<PENDING>` filled with the measured LLM09 result (Step 3)
