"""Shared prompt builder for AI providers.

Injects parsed constraints directly into the prompt so models don't have
to re-parse the user query. Removes distance_miles from the schema since
AI models have no location context and fabricate it — distance scoring
only applies when a real search provider returns actual distance data.
"""

from __future__ import annotations

from angel_filter.constraints import QueryConstraints


def build_prompt(
    user_query: str,
    top_k: int,
    constraints: QueryConstraints | None = None,
    strict_format: bool = False,
) -> str:
    """Build a structured recommendation prompt with explicit constraints.

    Args:
        user_query:    Raw query from the user.
        top_k:         How many candidates to return.
        constraints:   Parsed constraints to inject explicitly.
        strict_format: If True, add extra formatting reminders for models
                       that struggle with JSON (e.g. Ollama/llama3.2).
    """
    c = constraints or QueryConstraints()

    constraint_lines = []
    if c.budget is not None:
        constraint_lines.append(f"- Budget: under ${c.budget:.2f} per person")
    if c.min_rating is not None:
        constraint_lines.append(f"- Minimum rating: {c.min_rating} stars or above")
    if c.max_distance is not None:
        constraint_lines.append(f"- Maximum distance: within {c.max_distance} miles")

    constraints_block = (
        "Explicit constraints to satisfy:\n" + "\n".join(constraint_lines)
        if constraint_lines
        else "No explicit constraints — return the best overall matches."
    )

    format_reminder = (
        "\nCRITICAL: Output raw JSON only. No markdown. No ```json fences. "
        "No explanation before or after. Start your response with { and end with }."
        if strict_format
        else ""
    )

    return f"""You are a helpful local recommendations assistant.

User query: {user_query}

{constraints_block}

Return exactly {top_k} recommendations as JSON matching this schema exactly:
{{
  "query_summary": "one sentence explaining what you searched for",
  "candidates": [
    {{
      "name": "place or result name",
      "price": 12.50,
      "rating": 4.4,
      "area": "neighborhood or general location",
      "notes": "1-2 sentences on why this fits the query and constraints"
    }}
  ]
}}

Rules:
- Return exactly {top_k} candidates, ranked best-first.
- price must be a number (average spend per person in USD). Use null if unknown.
- rating must be a number 0-5. Use null if unknown.
- area is a neighborhood, district, or city area — not a full address.
- notes must explain how this result satisfies the constraints.
- Do NOT include distance_miles — you do not know the user's location.
- Do not include markdown, code fences, or any text outside the JSON.{format_reminder}
""".strip()
