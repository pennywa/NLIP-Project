"""Mock provider — returns canned results for deterministic testing.

Canned data uses a lunch-spot scenario with realistic price, distance, and
rating fields so all three P-axes have live data during demos and tests.
One result is sponsored so the penalty is visible in the ranking output.

The `name` parameter lets you instantiate two MockProviders with different
names (e.g. "mock_a", "mock_b") so consensus clustering tests can simulate
two independent providers agreeing on the same venue.

`query()` filters the canned set by keyword overlap so different query
strings return different subsets — making intent/constraint tests realistic.
"""

from angel_filter.providers.base import BaseProvider, ProviderResult


class MockProvider(BaseProvider):

    def __init__(
        self,
        canned_results: list[ProviderResult] | None = None,
        name: str = "mock",
    ):
        self.name = name
        self._canned = canned_results or _default_lunch_results()

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        """Return canned results filtered by keyword overlap with the query.

        Any result whose title or snippet shares at least one non-trivial token
        with the query is included. If nothing matches (e.g. totally unrelated
        query), all results are returned so tests never get an empty set by
        accident. Provider name is stamped onto each result at query time so
        two MockProvider instances produce results with distinct provider tags.
        """
        tokens = {t.lower().strip(".,!?;:'\"") for t in user_query.split() if len(t) > 2}
        if tokens:
            filtered = [
                r for r in self._canned
                if tokens & {t.lower().strip(".,!?;:'\"") for t in (r.title + " " + r.snippet).split()}
            ]
        else:
            filtered = list(self._canned)

        # Fall back to full set when nothing matched
        pool = filtered if filtered else list(self._canned)

        import dataclasses
        return [dataclasses.replace(r, provider=self.name) for r in pool[:max_results]]


def _default_lunch_results() -> list[ProviderResult]:
    """Lunch-spot demo set — price, distance, and rating all populated.

    Designed so:
      - A $15 budget query should surface Joe's Pizza or Terry Yaki
      - A nearest query should surface Chipotle (0.3 mi)
      - A highest-rated query should surface Joe's Pizza (4.8★)
      - SponsorCo Bistro is over budget and sponsored — penalty should push it down
    """
    return [
        ProviderResult(
            title="SponsorCo Bistro",
            snippet="Upscale lunch experience. Featured partner. Reservations recommended.",
            url="https://example.com/sponsorco-bistro",
            provider="mock",
            rank_in_provider=0,
            price=28.00,
            distance=0.8,
            rating=3.9,
            sponsored=True,
        ),
        ProviderResult(
            title="Joe's Pizza",
            snippet="Classic New York slices. Cash only. Best cheese pizza in the neighborhood.",
            url="https://example.com/joespizza",
            provider="mock",
            rank_in_provider=1,
            price=12.00,
            distance=0.5,
            rating=4.8,
            sponsored=False,
        ),
        ProviderResult(
            title="Terry Yaki",
            snippet="Fast Japanese teriyaki bowls. Generous portions, fresh ingredients.",
            url="https://example.com/terryyaki",
            provider="mock",
            rank_in_provider=2,
            price=13.50,
            distance=0.7,
            rating=4.5,
            sponsored=False,
        ),
        ProviderResult(
            title="Chipotle",
            snippet="Build-your-own burritos and bowls. Consistent quality, quick service.",
            url="https://example.com/chipotle",
            provider="mock",
            rank_in_provider=3,
            price=14.00,
            distance=0.3,
            rating=4.2,
            sponsored=False,
        ),
        ProviderResult(
            title="Hunter Halal Cart",
            snippet="Street halal over rice. Huge portions, white sauce, hot sauce.",
            url="https://example.com/hunterhalal",
            provider="mock",
            rank_in_provider=4,
            price=9.00,
            distance=1.1,
            rating=4.6,
            sponsored=False,
        ),
    ]
