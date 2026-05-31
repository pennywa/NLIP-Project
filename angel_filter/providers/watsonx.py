"""WatsonX provider — queries IBM's WatsonX generative AI API for structured recommendations.

Requires in the environment:
  WATSONX_API_KEY    — IBM Cloud API key
  WATSONX_PROJECT_ID — WatsonX project ID
  WATSONX_REGION     — e.g. us-east, us-south, eu-gb (default: us-south)
  WATSONX_MODEL      — model ID (default: ibm/granite-13b-instruct-v2)

WatsonX uses IAM token auth — the API key is exchanged for a bearer token
before each request. Tokens expire after 1 hour; we refresh lazily.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult

logger = logging.getLogger(__name__)

_IAM_URL      = "https://iam.cloud.ibm.com/identity/token"
_WATSONX_URL  = "https://{region}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
_TIMEOUT      = 45
_DEFAULT_MODEL  = os.getenv("WATSONX_MODEL",  "ibm/granite-13b-instruct-v2")
_DEFAULT_REGION = os.getenv("WATSONX_REGION", "us-south")

_PROMPT_TEMPLATE = """<|system|>
You are a helpful assistant that returns structured JSON recommendations only.
<|user|>
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
<|assistant|>
"""


class WatsonXProvider(BaseProvider):
    name = "watsonx"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        region: str = _DEFAULT_REGION,
    ):
        self.model  = model
        self.region = region
        self._token: str | None = None
        self._token_expiry: float = 0.0

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        import httpx

        api_key    = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")

        if not api_key:
            raise ProviderError("WATSONX_API_KEY is not set")
        if not project_id:
            raise ProviderError("WATSONX_PROJECT_ID is not set")

        token = await self._get_token(api_key)
        prompt = _PROMPT_TEMPLATE.format(query=user_query, top_k=max_results)
        url    = _WATSONX_URL.format(region=self.region)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type":  "application/json",
                        "Accept":        "application/json",
                    },
                    json={
                        "model_id":   self.model,
                        "project_id": project_id,
                        "input":      prompt,
                        "parameters": {
                            "decoding_method": "greedy",
                            "max_new_tokens":  512,
                            "temperature":     0.2,
                            "stop_sequences":  ["<|user|>", "<|system|>"],
                        },
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"WatsonX request failed: {exc}") from exc

        try:
            raw_text = resp.json()["results"][0]["generated_text"]
            payload  = _extract_json(raw_text)
        except Exception as exc:
            raise ProviderError(f"WatsonX response parse error: {exc}") from exc

        return _parse_results(payload, max_results)

    async def _get_token(self, api_key: str) -> str:
        import httpx

        # Reuse cached token if still valid (with 60s buffer)
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    _IAM_URL,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                        "apikey":     api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                self._token        = data["access_token"]
                self._token_expiry = time.time() + int(data.get("expires_in", 3600))
                return self._token
        except Exception as exc:
            raise ProviderError(f"WatsonX IAM token request failed: {exc}") from exc


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in WatsonX response")
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
        results.append(ProviderResult(
            title=name,
            snippet=str(c.get("notes", "")).strip(),
            provider="watsonx",
            rank_in_provider=i,
            price=_parse_float(c.get("price")),
            distance=_parse_float(c.get("distance_miles")),
            rating=_parse_float(c.get("rating")),
            sponsored=False,
        ))
    return results
