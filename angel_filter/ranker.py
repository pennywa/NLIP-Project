"""Ranker — the brain of the Angel Filter.

Scoring has four layers, applied in order:

  1. Semantic similarity — cosine similarity between the user preference
     embedding and the result text (via Ollama). Falls back to keyword
     overlap when Ollama is offline.

  2. Real-gap axis scoring — compute actual deltas between the candidate's
     structured fields and the user's extracted constraints:
       P1 price_gap    = candidate.price    - budget       (negative = under budget)
       P2 distance_gap = candidate.distance - max_distance (negative = closer)
       P3 rating_gap   = min_rating         - candidate.rating (negative = meets threshold)
     Gaps are normalised to 0-1 and weighted by the detected intent axis.

  3. Fuzzy consensus — candidates mentioned by multiple providers are boosted.
     Matching uses token overlap AND (when Ollama is available) embedding
     distance, so "Joe's Pizza" and "Joe Pizza" cluster together.

  4. Sponsored penalty — explicit deduction for any ad-flagged result.

Final score:
    score = semantic_similarity
            + (AXIS_WEIGHT  * axis_score)
            + (CONSENSUS_BONUS * extra_providers)
            - (SPONSORED_PENALTY if sponsored)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult

logger = logging.getLogger(__name__)

# --- Tunable weights ----------------------------------------------------------
# Weights sum to 1.0 across the three scoring components so no single
# signal can dominate. Adjust these to shift ranking behaviour.
W_SIMILARITY: float  = 0.50   # semantic similarity contribution
W_AXIS: float        = 0.35   # axis score contribution (split across all 3 axes)
W_CONSENSUS: float   = 0.15   # consensus contribution (capped)

SPONSORED_PENALTY: float = 0.20   # raised — ads should be clearly demoted
CONSENSUS_BONUS: float   = 0.075  # per extra provider; capped at 2 providers max
FUZZY_THRESHOLD: float   = 0.75   # embedding similarity above which two titles cluster
DEFAULT_EMBED_MODEL: str = "nomic-embed-text"

# Keep for backwards-compat in tests that reference AXIS_WEIGHT
AXIS_WEIGHT: float = W_AXIS


# --- Query intent -------------------------------------------------------------

class QueryIntent(str, Enum):
    PRICE    = "price"
    DISTANCE = "distance"
    RATING   = "rating"
    GENERAL  = "general"


_PRICE_KEYWORDS    = {"price", "cheap", "cheapest", "cost", "budget", "affordable",
                      "inexpensive", "low", "deal", "discount", "free", "save"}
_DISTANCE_KEYWORDS = {"near", "nearest", "close", "closest", "nearby", "distance",
                      "walking", "local", "around", "location", "convenient"}
_RATING_KEYWORDS   = {"best", "top", "rated", "rating", "review", "reviews",
                      "trusted", "quality", "popular", "recommended", "highest"}


def detect_intent(query: str) -> QueryIntent:
    tokens        = {t.lower().strip(".,!?;:'\"") for t in query.split()}
    price_hits    = len(tokens & _PRICE_KEYWORDS)
    distance_hits = len(tokens & _DISTANCE_KEYWORDS)
    rating_hits   = len(tokens & _RATING_KEYWORDS)
    best = max(price_hits, distance_hits, rating_hits)
    if best == 0:
        return QueryIntent.GENERAL
    if price_hits == best:
        return QueryIntent.PRICE
    if distance_hits == best:
        return QueryIntent.DISTANCE
    return QueryIntent.RATING


# --- Result dataclass ---------------------------------------------------------

@dataclass
class RankedResult:
    result: ProviderResult
    score: float
    rationale: str
    axis_scores: dict[str, float] = field(default_factory=dict)
    consensus_count: int = 0


# --- Ranker -------------------------------------------------------------------

class Ranker:
    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL):
        self.embed_model = embed_model
        self._ollama_available: bool | None = None

    async def rank(
        self,
        user_preference: str,
        results: list[ProviderResult],
        top_k: int = 5,
        intent: QueryIntent = QueryIntent.GENERAL,
        constraints: QueryConstraints | None = None,
    ) -> list[RankedResult]:
        if not results:
            return []

        constraints = constraints or QueryConstraints()

        # Hard constraint filtering — remove results that clearly violate budget
        # or minimum rating before scoring so they can't sneak into the top 5
        results = _apply_hard_constraints(results, constraints)
        if not results:
            return []

        if await self._has_ollama():
            # Build fuzzy consensus map using embeddings when available
            embeddings = await self._embed_all(results)
            consensus  = _build_fuzzy_consensus(results, embeddings, FUZZY_THRESHOLD)
            scored     = await self._score_with_embeddings(
                user_preference, results, intent, constraints, consensus, embeddings
            )
        else:
            logger.warning("Ollama unavailable; using keyword-overlap fallback.")
            consensus = _build_token_consensus(results)
            scored    = _score_with_keywords(
                user_preference, results, intent, constraints, consensus
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # -- private ---------------------------------------------------------------

    async def _has_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import ollama
            ollama.embeddings(model=self.embed_model, prompt="ping")
            self._ollama_available = True
        except Exception as exc:
            logger.info("Ollama probe failed: %s", exc)
            self._ollama_available = False
        return self._ollama_available

    async def _embed_all(
        self, results: list[ProviderResult]
    ) -> dict[int, list[float]]:
        """Return {result_index: embedding_vector} for every result."""
        import ollama
        vecs: dict[int, list[float]] = {}
        for i, r in enumerate(results):
            text = f"{r.title}. {r.snippet}"
            vecs[i] = ollama.embeddings(model=self.embed_model, prompt=text)["embedding"]
        return vecs

    async def _score_with_embeddings(
        self,
        user_preference: str,
        results: list[ProviderResult],
        intent: QueryIntent,
        constraints: QueryConstraints,
        consensus: dict[str, int],
        embeddings: dict[int, list[float]],
    ) -> list[RankedResult]:
        import ollama

        pref_vec = ollama.embeddings(
            model=self.embed_model, prompt=user_preference
        )["embedding"]

        scored: list[RankedResult] = []
        for i, r in enumerate(results):
            similarity  = _cosine(pref_vec, embeddings[i])
            axis_scores = _compute_gap_scores(r, constraints)
            axis_bonus  = _axis_bonus(axis_scores, intent)
            c_count     = consensus.get(_normalise(r.title), 1)
            # Cap consensus at 2 extra providers to prevent gang-up effect
            c_bonus     = CONSENSUS_BONUS * min(c_count - 1, 2)
            penalty     = SPONSORED_PENALTY if r.sponsored else 0.0

            # Balanced formula: each component has a defined weight
            final_score = (
                W_SIMILARITY * similarity
                + W_AXIS * axis_bonus
                + W_CONSENSUS * c_bonus
                - penalty
            )

            scored.append(RankedResult(
                result=r,
                score=round(final_score, 4),
                rationale=_explain(similarity, axis_scores, intent, c_count, constraints, r.sponsored),
                axis_scores=axis_scores,
                consensus_count=c_count,
            ))
        return scored


# --- Keyword fallback ---------------------------------------------------------

def _score_with_keywords(
    user_preference: str,
    results: list[ProviderResult],
    intent: QueryIntent,
    constraints: QueryConstraints,
    consensus: dict[str, int],
) -> list[RankedResult]:
    pref_tokens = _tokens(user_preference)
    scored: list[RankedResult] = []
    for r in results:
        haystack    = _tokens(f"{r.title} {r.snippet}")
        overlap     = len(pref_tokens & haystack)
        similarity  = overlap / max(len(pref_tokens), 1)
        axis_scores = _compute_gap_scores(r, constraints)
        axis_bonus  = _axis_bonus(axis_scores, intent)
        c_count     = consensus.get(_normalise(r.title), 1)
        c_bonus     = CONSENSUS_BONUS * min(c_count - 1, 2)
        penalty     = SPONSORED_PENALTY if r.sponsored else 0.0

        final_score = (
            W_SIMILARITY * similarity
            + W_AXIS * axis_bonus
            + W_CONSENSUS * c_bonus
            - penalty
        )

        rationale = (
            f"[keyword fallback] {overlap} terms matched"
            + (f", {intent.value} axis" if intent != QueryIntent.GENERAL else "")
            + (f", {c_count} providers agreed" if c_count > 1 else "")
            + (" — sponsored penalty applied" if r.sponsored else "")
        )
        scored.append(RankedResult(
            result=r,
            score=round(final_score, 4),
            rationale=rationale,
            axis_scores=axis_scores,
            consensus_count=c_count,
        ))
    return scored


# --- Hard constraint filtering ------------------------------------------------

def _apply_hard_constraints(
    results: list[ProviderResult],
    c: QueryConstraints,
) -> list[ProviderResult]:
    """Remove results that clearly violate hard constraints.

    Only filters when the result has data for that axis AND the violation
    is significant (>25% over budget, below min rating). Results with no
    data for an axis pass through — we don't penalize missing data.
    """
    filtered = []
    for r in results:
        # Budget: reject if price is more than 25% over budget
        if c.budget is not None and r.price is not None:
            if r.price > c.budget * 1.25:
                logger.debug("Hard filter: %s ($%.2f) exceeds budget $%.2f", r.title, r.price, c.budget)
                continue
        # Rating: reject if rating is more than 0.5 stars below minimum
        if c.min_rating is not None and r.rating is not None:
            if r.rating < c.min_rating - 0.5:
                logger.debug("Hard filter: %s (%.1f★) below min rating %.1f★", r.title, r.rating, c.min_rating)
                continue
        filtered.append(r)

    # Never return empty — if everything got filtered, return all results
    # (better to show something than nothing)
    return filtered if filtered else results


# --- Real-gap axis scoring ----------------------------------------------------

def _compute_gap_scores(r: ProviderResult, c: QueryConstraints) -> dict[str, float]:
    """Compute normalised 0-1 scores for each P axis using real constraint gaps.

    Gap convention: negative gap = candidate meets or beats the constraint.
    We map gap → score so that meeting the constraint gives 1.0 and badly
    missing it gives 0.0.

    When no constraint is set for an axis, score is neutral 0.5.
    When the candidate has no data for an axis, score is neutral 0.5.
    """

    # P1 — Price: lower is better
    if c.budget is not None and r.price is not None:
        gap = r.price - c.budget          # negative = under budget
        # Map: gap=-budget (free) → 1.0, gap=0 → 0.75, gap=budget → 0.0
        p1 = max(0.0, min(1.0, 0.75 - (gap / max(c.budget, 1.0)) * 0.75))
    elif r.price is not None:
        # No budget set — score by absolute price (lower = better, $100 ceiling)
        p1 = max(0.0, 1.0 - (r.price / 100.0))
    else:
        p1 = 0.5

    # P2 — Distance: closer is better
    if c.max_distance is not None and r.distance is not None:
        gap = r.distance - c.max_distance  # negative = within range
        p2 = max(0.0, min(1.0, 0.75 - (gap / max(c.max_distance, 0.1)) * 0.75))
    elif r.distance is not None:
        # No distance constraint — score by absolute distance (5 mile ceiling)
        p2 = max(0.0, 1.0 - (r.distance / 5.0))
    else:
        p2 = 0.5

    # P3 — Rating: higher is better
    if c.min_rating is not None and r.rating is not None:
        gap = c.min_rating - r.rating      # negative = meets or exceeds threshold
        p3 = max(0.0, min(1.0, 0.75 + (gap / -5.0) * 0.25)) if gap <= 0 else max(0.0, 0.75 - gap * 0.25)
    elif r.rating is not None:
        p3 = r.rating / 5.0
    else:
        p3 = 0.5

    return {
        "P1_price":    round(p1, 3),
        "P2_distance": round(p2, 3),
        "P3_rating":   round(p3, 3),
    }


def _axis_bonus(axis_scores: dict[str, float], intent: QueryIntent) -> float:
    """Compute weighted axis score — all three axes always contribute.

    Intent shifts the weights so the dominant axis gets more influence,
    but the other two axes still count. This handles "cheap AND nearby"
    queries correctly instead of winner-take-all on a single axis.

    Weights per intent (dominant / secondary / tertiary):
      PRICE:    60% / 20% / 20%
      DISTANCE: 60% / 20% / 20%
      RATING:   60% / 20% / 20%
      GENERAL:  33% / 33% / 33%
    """
    p1 = axis_scores["P1_price"]
    p2 = axis_scores["P2_distance"]
    p3 = axis_scores["P3_rating"]

    if intent == QueryIntent.PRICE:
        return 0.60 * p1 + 0.20 * p2 + 0.20 * p3
    if intent == QueryIntent.DISTANCE:
        return 0.20 * p1 + 0.60 * p2 + 0.20 * p3
    if intent == QueryIntent.RATING:
        return 0.20 * p1 + 0.20 * p2 + 0.60 * p3
    return (p1 + p2 + p3) / 3


# --- Fuzzy consensus clustering -----------------------------------------------

def _build_fuzzy_consensus(
    results: list[ProviderResult],
    embeddings: dict[int, list[float]],
    threshold: float,
) -> dict[str, int]:
    """Cluster results by embedding similarity, then count providers per cluster.

    Two results are in the same cluster if their embeddings are above
    `threshold` similar AND they come from different providers. The cluster
    representative is the normalised title of the first member seen.
    """
    n = len(results)
    # Union-find cluster assignment
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if results[i].provider == results[j].provider:
                continue  # don't cluster results from the same provider
            if _cosine(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    # Count distinct providers per cluster
    cluster_providers: dict[int, set[str]] = {}
    for i, r in enumerate(results):
        root = find(i)
        cluster_providers.setdefault(root, set()).add(r.provider)

    # Map each normalised title to the provider count of its cluster
    counts: dict[str, int] = {}
    for i, r in enumerate(results):
        root = find(i)
        counts[_normalise(r.title)] = len(cluster_providers[root])
    return counts


def _build_token_consensus(results: list[ProviderResult]) -> dict[str, int]:
    """Fallback consensus: simple normalised-title exact match across providers."""
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for r in results:
        key = (_normalise(r.title), r.provider)
        if key not in seen:
            seen.add(key)
            counts[_normalise(r.title)] += 1
    return dict(counts)


# --- Maths & helpers ----------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {t.lower().strip(".,!?;:") for t in text.split() if len(t) > 2}


def _normalise(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _explain(
    similarity: float,
    axis_scores: dict[str, float],
    intent: QueryIntent,
    consensus_count: int,
    constraints: QueryConstraints,
    sponsored: bool | None,
) -> str:
    tag  = "strong match" if similarity > 0.7 else "partial match" if similarity > 0.4 else "weak match"
    base = f"{tag} (similarity {similarity:.2f})"

    if intent == QueryIntent.PRICE and constraints.budget is not None:
        base += f", P1 price score {axis_scores['P1_price']:.2f} (budget ${constraints.budget})"
    elif intent == QueryIntent.DISTANCE and constraints.max_distance is not None:
        base += f", P2 distance score {axis_scores['P2_distance']:.2f} (within {constraints.max_distance} mi)"
    elif intent == QueryIntent.RATING and constraints.min_rating is not None:
        base += f", P3 rating score {axis_scores['P3_rating']:.2f} (min {constraints.min_rating}★)"
    elif intent != QueryIntent.GENERAL:
        axis_key = {"price": "P1_price", "distance": "P2_distance", "rating": "P3_rating"}[intent.value]
        base += f", {intent.value} axis {axis_scores[axis_key]:.2f}"

    if consensus_count > 1:
        base += f", {consensus_count} providers agreed"
    if sponsored:
        base += f" — sponsored, penalty {SPONSORED_PENALTY} applied"

    return base
