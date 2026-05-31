"""NLIP server for the Angel Filter.

Follows the pattern documented in nlip_server's README: subclass NLIPApplication
and NLIPSession, then pass them to the server startup helper.

Reference: https://github.com/nlip-project/nlip_server — see echo.py for the
minimum viable example this is modeled on.

Run locally with:
    poetry run fastapi dev angel_filter/server.py

If the nlip_server imports below fail, it means the NLIP packages are not
installed yet (poetry install has not been run, or the repos weren't
accessible). The fallback FastAPI app at the bottom of this file lets you
still run the proxy for local testing — see docs/DEV_FALLBACK.md.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from angel_filter.cache import CACHE
from angel_filter.orchestrator import Orchestrator
from angel_filter.providers import DuckDuckGoProvider, MockProvider

logger = logging.getLogger(__name__)

# --- Prometheus metrics -------------------------------------------------------
QUERY_COUNT = Counter(
    "angel_filter_queries_total",
    "Total number of /query requests",
    ["status"],  # "success" or "error"
)
QUERY_LATENCY = Histogram(
    "angel_filter_query_duration_seconds",
    "Time spent processing a /query request",
)
SPONSORED_PENALTY_COUNT = Counter(
    "angel_filter_sponsored_penalties_total",
    "Number of results that had the sponsored penalty applied",
)
UPTIME_GAUGE = Gauge(
    "angel_filter_start_timestamp_seconds",
    "Unix timestamp when the server process started",
)
_START_TIME = time.time()
UPTIME_GAUGE.set(_START_TIME)


# --- Build the orchestrator once at import time ---
# The set of providers the proxy fans out to. MockProvider is kept in the
# default list so the demo always returns something even if the network is
# unreliable. Once Google/Bing adapters land, append them here.
def _build_orchestrator() -> Orchestrator:
    providers = [
        DuckDuckGoProvider(),
        MockProvider(),
    ]
    return Orchestrator(providers=providers)


ORCHESTRATOR = _build_orchestrator()


# --- Shared health helper -----------------------------------------------------

def _health_response(mode: str, nlip_available: bool) -> dict:
    return {
        "ok": True,
        "mode": mode,
        "nlip_available": nlip_available,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "providers": [p.name for p in ORCHESTRATOR.providers],
    }


# --- NLIP integration ---------------------------------------------------------
# Per nlip_server's README, we subclass NLIPApplication and NLIPSession, and
# start the server via its helper. The exact import paths below mirror their
# echo.py example; if an upstream refactor changes them, fix here in one place.

try:
    from nlip_server.server import NLIP_Application, NLIP_Session, setup_server
    from nlip_sdk.nlip import NLIP_Factory, NLIP_Message

    _NLIP_AVAILABLE = True
except ImportError as exc:
    logger.warning("NLIP libraries not importable (%s); server.py will expose "
                   "a plain FastAPI fallback instead. Run `poetry install` once "
                   "dependencies resolve.", exc)
    _NLIP_AVAILABLE = False


if _NLIP_AVAILABLE:

    class AngelFilterSession(NLIP_Session):
        """One session = one user's ongoing conversation with the proxy."""

        async def start(self) -> None:
            logger.info("AngelFilterSession started.")

        async def stop(self) -> None:
            logger.info("AngelFilterSession stopped.")

        async def execute(self, msg: NLIP_Message) -> NLIP_Message:
            user_query = str(msg.content) if msg.content else ""
            logger.info("Incoming query: %r", user_query)
            response = await ORCHESTRATOR.handle_query(user_query=user_query)
            return NLIP_Factory.create_text(_format_reply(response))


    class AngelFilterApplication(NLIP_Application):
        """The NLIP application — spawns one session per client connection."""

        async def startup(self) -> None:
            logger.info("AngelFilterApplication starting up.")

        async def shutdown(self) -> None:
            logger.info("AngelFilterApplication shutting down.")

        def create_session(self) -> AngelFilterSession:
            return AngelFilterSession()


    app = setup_server(AngelFilterApplication())

    # Mount the demo UI routes onto the NLIP app so /query and / work for the
    # frontend regardless of whether NLIP is the active transport.
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path
    from pydantic import BaseModel

    _STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

    class QueryIn(BaseModel):
        query: str
        preference: str | None = None

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"msg": "Angel Filter — POST to /query or /nlip/"}

    @app.get("/health")
    async def health():
        return _health_response(mode="nlip", nlip_available=True)

    @app.post("/query")
    async def query(body: QueryIn):
        with QUERY_LATENCY.time():
            try:
                response = await ORCHESTRATOR.handle_query(
                    user_query=body.query,
                    user_preference=body.preference,
                )
                QUERY_COUNT.labels(status="success").inc()
                for r in response.ranked:
                    if r.result.sponsored:
                        SPONSORED_PENALTY_COUNT.inc()
            except Exception:
                QUERY_COUNT.labels(status="error").inc()
                raise
        return {
            "providers_used": response.providers_used,
            "providers_failed": response.providers_failed,
            "intent": response.intent.value,
            "constraints": {
                "budget": response.constraints.budget,
                "max_distance": response.constraints.max_distance,
                "min_rating": response.constraints.min_rating,
            },
            "results": [
                {
                    "title": r.result.title,
                    "snippet": r.result.snippet,
                    "url": r.result.url,
                    "provider": r.result.provider,
                    "score": round(r.score, 3),
                    "rationale": r.rationale,
                    "sponsored": r.result.sponsored,
                    "consensus_count": r.consensus_count,
                    "axis_scores": r.axis_scores,
                }
                for r in response.ranked
            ],
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


else:
    # --- Fallback: plain FastAPI so the demo still runs without NLIP installed -
    # This path exists so Friday's demo is not held hostage by a dependency
    # install problem. It exposes a single POST /query endpoint that does the
    # same thing the NLIP session would do. Remove once NLIP is reliably
    # installable on every teammate's machine.
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pathlib import Path
    from pydantic import BaseModel

    app = FastAPI(
        title="Angel Filter",
        description="A local proxy that re-ranks multi-provider AI results and penalizes sponsored content.",
        version="0.1.0",
    )

    _STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

    class QueryIn(BaseModel):
        query: str
        preference: str | None = None

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"msg": "Angel Filter is running (fallback mode). POST to /query."}

    @app.get("/health")
    async def health():
        return _health_response(mode="fallback", nlip_available=False)

    @app.post("/query")
    async def query(body: QueryIn):
        # Return cached result if fresh
        cached = CACHE.get(body.query, body.preference)
        if cached:
            logger.info("Cache hit for query: %r", body.query)
            return cached

        with QUERY_LATENCY.time():
            try:
                response = await ORCHESTRATOR.handle_query(
                    user_query=body.query,
                    user_preference=body.preference,
                )
                QUERY_COUNT.labels(status="success").inc()
                for r in response.ranked:
                    if r.result.sponsored:
                        SPONSORED_PENALTY_COUNT.inc()
            except Exception:
                QUERY_COUNT.labels(status="error").inc()
                raise

        payload = {
            "providers_used": response.providers_used,
            "providers_failed": response.providers_failed,
            "intent": response.intent.value,
            "constraints": {
                "budget": response.constraints.budget,
                "max_distance": response.constraints.max_distance,
                "min_rating": response.constraints.min_rating,
            },
            "cached": False,
            "results": [
                {
                    "title": r.result.title,
                    "snippet": r.result.snippet,
                    "url": r.result.url,
                    "provider": r.result.provider,
                    "score": round(r.score, 3),
                    "rationale": r.rationale,
                    "sponsored": r.result.sponsored,
                    "consensus_count": r.consensus_count,
                    "axis_scores": r.axis_scores,
                }
                for r in response.ranked
            ],
        }
        CACHE.set(body.query, body.preference, {**payload, "cached": True})
        return payload

    @app.get("/history")
    async def history():
        return {"queries": CACHE.history(), "cache_stats": CACHE.stats()}

    @app.post("/cache/clear")
    async def cache_clear():
        CACHE._store.clear()
        CACHE._history.clear()
        return {"ok": True, "message": "Cache cleared."}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Helpers ------------------------------------------------------------------

def _format_reply(response) -> str:
    if not response.ranked:
        return "No results from any provider. Providers tried: " + ", ".join(
            response.providers_used + response.providers_failed
        )
    lines = [f"Ranked {len(response.ranked)} results "
             f"(providers used: {', '.join(response.providers_used)}):"]
    for i, r in enumerate(response.ranked, start=1):
        src = r.result.provider
        tag = " [SPONSORED]" if r.result.sponsored else ""
        lines.append(f"{i}. {r.result.title}{tag} — {src} — {r.rationale}")
        if r.result.url:
            lines.append(f"   {r.result.url}")
    if response.providers_failed:
        lines.append("(failed: " + ", ".join(response.providers_failed) + ")")
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
