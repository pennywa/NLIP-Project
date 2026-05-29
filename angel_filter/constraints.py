"""Constraint extraction — parse explicit user constraints from a query string.

Turns natural language like "lunch under $15, within 5 miles, at least 4 stars"
into structured values the ranker can use to compute real axis gaps instead of
neutral proxies.

Extracted values feed directly into the P1/P2/P3 gap calculation:
    price_gap    = candidate.price    - budget          (negative = under budget = good)
    distance_gap = candidate.distance - max_distance    (negative = closer = good)
    rating_gap   = min_rating         - candidate.rating (negative = meets threshold = good)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryConstraints:
    """Structured constraints parsed from the user's query.

    None means the user didn't specify that constraint — the ranker will
    use neutral scoring for that axis rather than a real gap.
    """
    budget: float | None = None        # P1 — maximum price the user will pay
    max_distance: float | None = None  # P2 — maximum distance in miles
    min_rating: float | None = None    # P3 — minimum acceptable star rating


# --- Regex patterns -----------------------------------------------------------

# Price: "$15", "under $20", "budget $10", "15 dollars", "less than $30"
_PRICE_RE = re.compile(
    r"""
    (?:under|below|budget|max|less\s+than|within|around|about|up\s+to)?\s*
    \$\s*(\d+(?:\.\d+)?)
    |
    (\d+(?:\.\d+)?)\s*(?:dollars?|bucks?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Distance: "within 5 miles", "5 mile radius", "under 2 km", "nearby" (→ 1 mi default)
_DISTANCE_RE = re.compile(
    r"""
    (?:within|under|less\s+than|in\s+a|in)?\s*
    (\d+(?:\.\d+)?)\s*(?:mile(?:s)?|mi|km|kilometer(?:s)?)
    |
    \b(nearby|walking\s+distance|close\s+by)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Rating: "4 stars", "at least 4.5 stars", "rated above 3", "4+ stars"
# The number MUST be followed by a star keyword OR preceded by an explicit
# rating prefix — bare numbers are not captured to avoid false positives from
# price/distance figures (e.g. "0.5 miles" must not become a rating).
_RATING_RE = re.compile(
    r"""
    (?:at\s+least|above|over|minimum|min|rated\s+(?:at\s+least|above|over)?)
    \s*(\d+(?:\.\d+)?)
    |
    (\d+(?:\.\d+)?)\s*\+?\s*stars?
    |
    (\d+(?:\.\d+)?)\s+star\s+rating
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_constraints(query: str) -> QueryConstraints:
    """Parse a raw user query and return whatever constraints can be found.

    Unrecognised constraints are left as None — the ranker handles None
    gracefully by falling back to neutral axis scoring.
    """
    budget       = _extract_price(query)
    max_distance = _extract_distance(query)
    min_rating   = _extract_rating(query)
    return QueryConstraints(
        budget=budget,
        max_distance=max_distance,
        min_rating=min_rating,
    )


def _extract_price(text: str) -> float | None:
    m = _PRICE_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_distance(text: str) -> float | None:
    m = _DISTANCE_RE.search(text)
    if not m:
        return None
    # "nearby" / "walking distance" → treat as 1 mile
    if m.group(2):
        return 1.0
    raw = m.group(1)
    try:
        val = float(raw)
        # Rough km → miles conversion if the unit was km
        unit = m.group(0).lower()
        if "km" in unit or "kilo" in unit:
            val = val * 0.621371
        return round(val, 2)
    except (TypeError, ValueError):
        return None


def _extract_rating(text: str) -> float | None:
    m = _RATING_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2) or m.group(3)
    try:
        val = float(raw)
        if 0.0 <= val <= 5.0:
            return val
        return None
    except (TypeError, ValueError):
        return None
