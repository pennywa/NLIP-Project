"""Test the orchestrator end-to-end with the mock provider.

These tests do not require network or Ollama — they prove the fan-out and
ranking pipeline works as a standalone unit. Run with:
    poetry run pytest
"""

import pytest

from angel_filter.orchestrator import Orchestrator
from angel_filter.providers import MockProvider
from angel_filter.ranker import detect_intent, QueryIntent


# ---------------------------------------------------------------------------
# Basic pipeline smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_returns_ranked_results():
    orch = Orchestrator(providers=[MockProvider()])
    response = await orch.handle_query(
        user_query="cheap pizza lunch",
        user_preference="low price, casual spot",
        top_k=5,
    )

    assert "mock" in response.providers_used
    assert response.providers_failed == []
    assert len(response.ranked) > 0


@pytest.mark.asyncio
async def test_sponsored_results_are_penalized():
    """The sponsored item must score lower than it would without the penalty.

    We compare SponsorCo Bistro's score against the same result would score
    without the SPONSORED_PENALTY by checking it is not ranked #1 when an
    organic result with comparable keyword overlap exists.
    """
    from angel_filter.ranker import SPONSORED_PENALTY

    orch = Orchestrator(providers=[MockProvider()])
    orch.ranker._ollama_available = False  # deterministic keyword path

    # Use a query that matches ALL canned results equally (one shared token: "lunch")
    # so keyword similarity is the same for everyone — penalty is the only differentiator
    response = await orch.handle_query(
        user_query="lunch pizza halal bowls slices",
        top_k=10,
    )

    sponsored = [r for r in response.ranked if r.result.sponsored]
    organic = [r for r in response.ranked if not r.result.sponsored]

    assert sponsored, "expected at least one sponsored item in the canned data"
    assert organic, "expected at least one organic item in the canned data"

    # The sponsored result's score must be lower than the top organic result's score
    top_organic_score = organic[0].score
    sponsored_score = sponsored[0].score
    assert sponsored_score < top_organic_score, (
        f"sponsored score {sponsored_score} >= top organic score {top_organic_score} "
        f"— penalty of {SPONSORED_PENALTY} is not being applied"
    )


@pytest.mark.asyncio
async def test_orchestrator_tolerates_provider_failures():
    """If one provider blows up, the others still return results."""

    from angel_filter.providers.base import BaseProvider, ProviderError

    class BrokenProvider(BaseProvider):
        name = "broken"

        async def query(self, user_query: str, max_results: int = 10):
            raise ProviderError("simulated outage")

    orch = Orchestrator(providers=[MockProvider(), BrokenProvider()])
    response = await orch.handle_query(user_query="pizza lunch")

    assert "broken" in response.providers_failed
    assert "mock" in response.providers_used
    assert len(response.ranked) > 0


# ---------------------------------------------------------------------------
# Constraint-aware ranking tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_constraint_demotes_over_budget_results():
    """With a $15 budget, SponsorCo Bistro ($28) should rank below cheaper spots."""
    orch = Orchestrator(providers=[MockProvider()])
    orch.ranker._ollama_available = False  # deterministic keyword path

    response = await orch.handle_query(
        user_query="lunch under $15 cheap pizza halal teriyaki",
        top_k=5,
    )

    assert response.constraints.budget == 15.0, (
        f"expected budget=15.0, got {response.constraints.budget}"
    )
    # SponsorCo Bistro costs $28 and is sponsored — it should NOT be #1
    top_title = response.ranked[0].result.title
    assert top_title != "SponsorCo Bistro", (
        f"$28 sponsored bistro landed at #1 despite $15 budget: {top_title}"
    )


@pytest.mark.asyncio
async def test_nearest_query_favors_chipotle():
    """A 'nearest' query should surface Chipotle (0.3 mi, closest in the set)."""
    orch = Orchestrator(providers=[MockProvider()])
    orch.ranker._ollama_available = False

    response = await orch.handle_query(
        user_query="nearest lunch spot nearby chipotle pizza halal",
        top_k=5,
    )

    assert response.intent == QueryIntent.DISTANCE
    titles = [r.result.title for r in response.ranked]
    chipotle_rank = titles.index("Chipotle") if "Chipotle" in titles else len(titles)
    # Chipotle at 0.3 mi should be in the top 2
    assert chipotle_rank <= 1, (
        f"Chipotle (0.3 mi) ranked #{chipotle_rank + 1}, expected top 2. Rankings: {titles}"
    )


@pytest.mark.asyncio
async def test_highest_rated_query_favors_joes_pizza():
    """A 'best rated' query should surface Joe's Pizza (4.8★, highest in set)."""
    orch = Orchestrator(providers=[MockProvider()])
    orch.ranker._ollama_available = False

    response = await orch.handle_query(
        user_query="best rated top reviewed pizza lunch",
        top_k=5,
    )

    assert response.intent == QueryIntent.RATING
    titles = [r.result.title for r in response.ranked]
    assert "Joe's Pizza" in titles, f"Joe's Pizza (4.8★) not in results: {titles}"
    joes_rank = titles.index("Joe's Pizza")
    assert joes_rank <= 1, (
        f"Joe's Pizza (4.8★) ranked #{joes_rank + 1}, expected top 2. Rankings: {titles}"
    )


@pytest.mark.asyncio
async def test_axis_scores_present_on_all_results():
    """Every ranked result must carry P1/P2/P3 axis_scores."""
    orch = Orchestrator(providers=[MockProvider()])
    orch.ranker._ollama_available = False

    response = await orch.handle_query(user_query="pizza lunch", top_k=5)

    for r in response.ranked:
        assert "P1_price" in r.axis_scores, f"missing P1_price on {r.result.title}"
        assert "P2_distance" in r.axis_scores, f"missing P2_distance on {r.result.title}"
        assert "P3_rating" in r.axis_scores, f"missing P3_rating on {r.result.title}"
        for key, val in r.axis_scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] for {r.result.title}"


# ---------------------------------------------------------------------------
# Consensus bonus tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consensus_bonus_applied_when_two_providers_agree():
    """Results appearing in both mock_a and mock_b get a consensus_count > 1."""
    # Two MockProvider instances with different names simulate two independent
    # providers that happen to return the same venues.
    mock_a = MockProvider(name="mock_a")
    mock_b = MockProvider(name="mock_b")

    orch = Orchestrator(providers=[mock_a, mock_b])
    orch.ranker._ollama_available = False  # token consensus path

    response = await orch.handle_query(
        user_query="pizza lunch halal teriyaki chipotle",
        top_k=5,
    )

    # With two providers, every matched result should have consensus_count >= 2
    # (token consensus counts provider-distinct matches by normalised title)
    multi_provider = [r for r in response.ranked if r.consensus_count > 1]
    assert multi_provider, (
        "expected at least one result with consensus_count > 1 when two providers agree"
    )


# ---------------------------------------------------------------------------
# Intent detection unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_intent", [
    ("cheap affordable pizza under $10",         QueryIntent.PRICE),
    ("best deal budget lunch save money",        QueryIntent.PRICE),
    ("nearest restaurant close to me",           QueryIntent.DISTANCE),
    ("walking distance nearby local spot",       QueryIntent.DISTANCE),
    ("best rated top reviewed highest quality",  QueryIntent.RATING),
    ("trusted popular recommended restaurant",   QueryIntent.RATING),
    ("I want lunch",                             QueryIntent.GENERAL),
    ("where should I eat today",                 QueryIntent.GENERAL),
])
def test_detect_intent(query, expected_intent):
    result = detect_intent(query)
    assert result == expected_intent, (
        f"detect_intent({query!r}) → {result.value}, expected {expected_intent.value}"
    )


# ---------------------------------------------------------------------------
# Constraint extraction unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,budget,max_dist,min_rating", [
    ("lunch under $15",            15.0,  None, None),
    ("cheap pizza $12 budget",     12.0,  None, None),
    ("within 0.5 miles",           None,  0.5,  None),
    ("nearby lunch",               None,  1.0,  None),
    ("rated at least 4 stars",     None,  None, 4.0),
    ("$15 lunch nearby 4 stars",   15.0,  1.0,  4.0),
    ("just food please",           None,  None, None),
])
def test_extract_constraints(query, budget, max_dist, min_rating):
    from angel_filter.constraints import extract_constraints
    c = extract_constraints(query)
    assert c.budget == budget,       f"budget: got {c.budget}, expected {budget} for {query!r}"
    assert c.max_distance == max_dist, f"max_dist: got {c.max_distance}, expected {max_dist} for {query!r}"
    assert c.min_rating == min_rating, f"min_rating: got {c.min_rating}, expected {min_rating} for {query!r}"
