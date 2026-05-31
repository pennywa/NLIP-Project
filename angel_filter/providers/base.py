"""Base contract for all provider adapters.

Every provider (Google, Bing/Copilot, DuckDuckGo, Anthropic free-tier, etc.)
must implement the same async `query()` method so the proxy can fan out to
all of them in parallel and normalize their responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from angel_filter.constraints import QueryConstraints


@dataclass
class ProviderResult:
    """A single normalized result from any provider.

    All providers must squash their native response format into this shape so
    the ranker sees a consistent interface. If a field is not available from a
    given provider, leave it empty/None — do not fabricate.
    """

    title: str
    snippet: str
    url: str | None = None
    provider: str = ""                       # "claude", "openai", "gemini", "duckduckgo"...
    rank_in_provider: int = 0                # where this appeared in the provider's own list
    price: float | None = None               # P1 — price in dollars
    distance: float | None = None            # P2 — distance in miles from user
    rating: float | None = None              # P3 — star rating 0-5
    sponsored: bool | None = None            # True if the provider flagged it as an ad
    raw: dict[str, Any] = field(default_factory=dict)  # the untouched original payload


class BaseProvider(ABC):
    """Abstract base class for provider adapters.

    Implementors should:
      - set `name` to a short lowercase identifier ("google", "bing", ...)
      - implement `query()` as a non-blocking coroutine
      - normalize everything into ProviderResult instances
      - raise ProviderError (below) on unrecoverable failures; the proxy will
        log and continue with other providers' results
    """

    name: str = "base"

    @abstractmethod
    async def query(
        self,
        user_query: str,
        max_results: int = 10,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult]:
        """Submit a query and return normalized results.

        Args:
            user_query:   the raw text the user typed.
            max_results:  soft cap on how many results to return.
            constraints:  parsed constraints to pass to AI providers for
                          better-targeted prompts. May be None.

        Returns:
            A list of ProviderResult, ordered as the provider returned them.
        """
        raise NotImplementedError


class ProviderError(Exception):
    """Raised when a provider call fails in a way the proxy should log + skip."""
