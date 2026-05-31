"""Brave Search provider — queries Brave's Web Search API.

Requires BRAVE_API_KEY in the environment. Get a free key at:
https://api.search.brave.com/

Brave is a reliable alternative to DuckDuckGo because it uses an official
API with proper rate limits instead of scraping. Free tier allows 2,000
queries/month which is plenty for demos and development.
"""

from __future__ import annotations

import logging
import os

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT = 15
_DEFAULT_COUNT = 10


class BraveProvider(BaseProvider):
    name = "brave"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("BRAVE_API_KEY")

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        import httpx

        if not self._api_key:
            raise ProviderError("BRAVE_API_KEY is not set")

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _BRAVE_URL,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                    params={
                        "q": user_query,
                        "count": min(max_results, _DEFAULT_COUNT),
                        "safesearch": "moderate",
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"Brave Search request failed: {exc}") from exc

        try:
            data = resp.json()
            web_results = data.get("web", {}).get("results", [])
        except Exception as exc:
            raise ProviderError(f"Brave Search parse error: {exc}") from exc

        results = []
        for i, item in enumerate(web_results[:max_results]):
            results.append(ProviderResult(
                title=item.get("title", "").strip(),
                snippet=item.get("description", "").strip(),
                url=item.get("url"),
                provider="brave",
                rank_in_provider=i,
                sponsored=False,
            ))
        return [r for r in results if r.title]
