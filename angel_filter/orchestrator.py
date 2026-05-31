"""Orchestrator — runs every registered provider in parallel, collects results,
then hands the combined pile to the ranker.

Responsible for three things the ranker doesn't own:
  1. Fan-out — fire all providers simultaneously, isolate failures.
  2. Intent detection — classify the query as price / distance / rating / general.
  3. Constraint extraction — parse explicit values ($15, 5 miles, 4 stars) from
     the query so the ranker can compute real axis gaps instead of neutral proxies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from angel_filter.constraints import QueryConstraints, extract_constraints
from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult
from angel_filter.ranker import QueryIntent, RankedResult, Ranker, detect_intent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    ranked: list[RankedResult]
    providers_used: list[str]
    providers_failed: list[str]
    intent: QueryIntent
    constraints: QueryConstraints
    axis_scores: dict[str, float] | None = None  # P1/P2/P3 of the top result


class Orchestrator:
    def __init__(self, providers: list[BaseProvider], ranker: Ranker | None = None):
        if not providers:
            raise ValueError("Orchestrator needs at least one provider.")
        self.providers = providers
        self.ranker = ranker or Ranker()

    async def handle_query(
        self,
        user_query: str,
        user_preference: str | None = None,
        top_k: int = 5,
    ) -> OrchestratorResponse:
        """Run the full pipeline: extract constraints → detect intent → fan out → rank."""

        # Combine query + preference so signals in either field are captured
        full_text   = f"{user_query} {user_preference or ''}".strip()
        intent      = detect_intent(full_text)
        constraints = extract_constraints(full_text)

        logger.info(
            "Intent: %s | Constraints: budget=%s, distance=%s, rating=%s",
            intent.value,
            constraints.budget,
            constraints.max_distance,
            constraints.min_rating,
        )

        # Fan out to all providers in parallel, passing constraints so AI
        # providers can inject them directly into their prompts
        tasks = [self._safe_query(p, user_query, constraints) for p in self.providers]
        per_provider = await asyncio.gather(*tasks)

        all_results: list[ProviderResult] = []
        used: list[str] = []
        failed: list[str] = []
        for provider, outcome in zip(self.providers, per_provider):
            if outcome is None:
                failed.append(provider.name)
            else:
                used.append(provider.name)
                all_results.extend(outcome)

        if not all_results:
            return OrchestratorResponse(
                ranked=[],
                providers_used=used,
                providers_failed=failed,
                intent=intent,
                constraints=constraints,
            )

        ranked = await self.ranker.rank(
            user_preference or user_query,
            all_results,
            top_k=top_k,
            intent=intent,
            constraints=constraints,
        )

        return OrchestratorResponse(
            ranked=ranked,
            providers_used=used,
            providers_failed=failed,
            intent=intent,
            constraints=constraints,
            axis_scores=ranked[0].axis_scores if ranked else None,
        )

    async def _safe_query(
        self,
        provider: BaseProvider,
        user_query: str,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult] | None:
        try:
            return await provider.query(user_query, constraints=constraints)
        except ProviderError as exc:
            logger.warning("Provider %s failed: %s", provider.name, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Provider %s raised unexpected error: %s", provider.name, exc)
            return None
