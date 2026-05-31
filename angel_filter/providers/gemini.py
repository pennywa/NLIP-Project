"""Gemini provider — queries Google's Gemini API and returns ProviderResult objects.

Requires GEMINI_API_KEY in the environment. Model defaults to gemini-2.5-flash
but can be overridden via the GEMINI_MODEL env var.

Prompt strategy: ask Gemini to return structured JSON with name, price,
distance_miles, rating, and notes for each candidate. The JSON is parsed
and normalized into ProviderResult objects so the ranker sees a consistent
interface regardless of which provider generated the data.
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

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={api_key}"
)
_TIMEOUT = 45
_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model

    async def query(
        self,
        user_query: str,
        max_results: int = 10,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult]:
        import httpx

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set")

        prompt = build_prompt(user_query, max_results, constraints)
        url = _GEMINI_URL.format(model=self.model, api_key=api_key)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.2,
                        },
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        try:
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            payload = _extract_json(raw_text)
        except Exception as exc:
            raise ProviderError(f"Gemini response parse error: {exc}") from exc

        return _parse_results(payload, max_results)


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in Gemini response")
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
            provider="gemini",
            rank_in_provider=i,
            price=_parse_float(c.get("price")),
            distance=None,  # AI has no location context
            rating=_parse_float(c.get("rating")),
            sponsored=False,
        ))
    return results
