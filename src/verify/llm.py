"""Single Azure OpenAI JSON-mode helper shared by the verifier perception steps."""
import json
import logging
import os

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    return _client


def chat_json(system: str, user: str) -> dict:
    """Call Azure OpenAI with temperature=0 JSON mode and return the parsed dict.

    Args:
        system: System instruction.
        user: User content.

    Returns:
        Parsed JSON object (empty dict on empty content).
    """
    resp = _get_client().chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    if not resp.choices:
        return {}
    try:
        result = json.loads(resp.choices[0].message.content or "{}")
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        logger.warning("chat_json: LLM returned invalid JSON; returning {}")
        return {}
