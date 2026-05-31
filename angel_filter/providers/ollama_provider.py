"""Ollama provider — queries a local Ollama instance and returns ProviderResult objects.

Requires Ollama running locally (default: http://localhost:11434).
Override the base URL via OLLAMA_URL and the model via OLLAMA_MODEL.

Uses the same structured JSON prompt as the Gemini provider so both
providers return comparable candidate shapes that the ranker can score
on the same three axes (price, distance, rating).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from angel_filter.constraints import QueryConstraints
from angel_filter.prompt import build_prompt
from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult

logger = logging.getLogger(__name__)

_DEFAULT_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
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


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str = _DEFAULT_URL, model: str = _DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def query(
        self,
        user_query: str,
        max_results: int = 10,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult]:
        import httpx

        prompt = build_prompt(user_query, max_results, constraints, strict_format=True)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.2},
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            raw_text = resp.json().get("response", "")
            payload = _extract_json(raw_text)
        except Exception as exc:
            raise ProviderError(f"Ollama response parse error: {exc}") from exc

        return _parse_results(payload, max_results)


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in Ollama response")
    return json.loads(text[start:end + 1])


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
        area = str(c.get("area", "")).strip()
        notes = str(c.get("notes", "")).strip()
        snippet = f"{area} — {notes}" if area else notes
        results.append(ProviderResult(
            title=name,
            snippet=snippet,
            provider="ollama",
            rank_in_provider=i,
            price=_parse_float(c.get("price")),
            distance=None,
            rating=_parse_float(c.get("rating")),
            sponsored=False,
        ))
    return results
