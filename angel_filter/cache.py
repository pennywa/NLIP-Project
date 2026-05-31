"""Query result cache — in-memory, TTL-based.

Keyed by (query, preference) normalized to lowercase stripped strings.
Entries expire after TTL_SECONDS. No external dependencies.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_HISTORY = 10


@dataclass
class CacheEntry:
    payload: dict[str, Any]
    stored_at: float = field(default_factory=time.time)

    def is_fresh(self) -> bool:
        return (time.time() - self.stored_at) < TTL_SECONDS

    def age_minutes(self) -> int:
        return int((time.time() - self.stored_at) / 60)


class QueryCache:
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        # Ordered list of (key, query_text) for history — most recent first
        self._history: list[tuple[str, str]] = []

    def _key(self, query: str, preference: str | None) -> str:
        combined = f"{query.strip().lower()}|{(preference or '').strip().lower()}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, query: str, preference: str | None) -> dict[str, Any] | None:
        entry = self._store.get(self._key(query, preference))
        if entry and entry.is_fresh():
            return entry.payload
        return None

    def set(self, query: str, preference: str | None, payload: dict[str, Any]) -> None:
        key = self._key(query, preference)
        self._store[key] = CacheEntry(payload=payload)

        # Update history — remove duplicates, prepend, cap at MAX_HISTORY
        self._history = [(k, q) for k, q in self._history if k != key]
        self._history.insert(0, (key, query.strip()))
        self._history = self._history[:MAX_HISTORY]

    def history(self) -> list[str]:
        """Return the MAX_HISTORY most recent query strings, freshest first."""
        fresh = []
        for key, query_text in self._history:
            entry = self._store.get(key)
            if entry and entry.is_fresh():
                fresh.append(query_text)
        return fresh

    def stats(self) -> dict[str, int]:
        total = len(self._store)
        fresh = sum(1 for e in self._store.values() if e.is_fresh())
        return {"total": total, "fresh": fresh, "history": len(self.history())}


# Module-level singleton shared across the app
CACHE = QueryCache()
