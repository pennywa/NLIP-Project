"""OpenAI provider — queries OpenAI's chat completions API for structured recommendations.

Requires OPENAI_API_KEY in the environment.
Model defaults to gpt-4o-mini but can be overridden via OPENAI_MODEL env var.

Uses the same structured JSON prompt as Gemini and Ollama so all three
AI providers return comparable candidate shapes the ranker can score
on the same three axes (price, distance, rating).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_TIMEOUT = 45

_PROMPT_TEMPLATE = """
Return the best recommendations for the user query below.

User query:
{query}

Respond as JSON only with this exact schema:
{{
  "query_summary": "short explanation",
  "candidates": [
    {{
      "name": "place name",
      "price": 12.5,
      "distance_miles": 0.8,
      "rating": 4.4,
      "notes": "why this fits"
    }}
  ]
}}

Rules:
- Return up to {top_k} candidates.
- Use numeric values for price, distance_miles, and rating whenever possible.
- If a value is unknown, use null.
- Do not include markdown, code fences, or extra text.
""".strip()


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        import httpx

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")

        prompt = _PROMPT_TEMPLATE.format(query=user_query, top_k=max_results)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": "You return structured JSON recommendations only."},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        try:
            raw_text = resp.json()["choices"][0]["message"]["content"]
            payload = json.loads(raw_text)
        except Exception as exc:
            raise ProviderError(f"OpenAI response parse error: {exc}") from exc

        return _parse_results(payload, max_results)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else None


def _parse_results(payload: dict[str, Any], max_results: int) -> list[ProviderResult]:
    results = []
    for i, c in enumerate(payload.get("candidates", [])[:max_results]):
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        results.append(ProviderResult(
            title=name,
            snippet=str(c.get("notes", "")).strip(),
            provider="openai",
            rank_in_provider=i,
            price=_parse_float(c.get("price")),
            distance=_parse_float(c.get("distance_miles")),
            rating=_parse_float(c.get("rating")),
            sponsored=False,
        ))
    return results
