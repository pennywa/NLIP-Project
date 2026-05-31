# Angel Filter

A multi-provider AI proxy that fans queries out to multiple AI and search
providers simultaneously, ranks results using semantic embeddings and a
three-axis scoring system (price · distance · rating), and penalizes
sponsored content — putting the user's interests ahead of advertiser dollars.

CUNY capstone project — final demo May 15, 2026.

---

## Contributing

All changes go through pull requests — no direct commits to `main`, including from project owners.

1. Create a branch from `main`:
   ```bash
   git checkout main && git pull origin main
   git checkout -b your-name/short-description
   ```
2. Make your changes, commit, and push:
   ```bash
   git push -u origin your-name/short-description
   ```
3. Open a pull request on GitHub targeting `main`. Add a brief description of what changed and why.
4. Get at least one teammate review before merging.

---

## Status (as of May 31, 2026)

| Component | State |
|---|---|
| FastAPI server (fallback mode) | **Working** |
| NLIP server (`NLIPApplication` / `NLIPSession`) | Pending — NLIP libraries not yet installable |
| Provider: OpenAI (`gpt-4o-mini`) | **Working** — needs `OPENAI_API_KEY` |
| Provider: Gemini (`gemini-2.5-flash`) | **Working** — needs `GEMINI_API_KEY` |
| Provider: Ollama (`llama3.2`) | **Working** — runs locally, no key needed |
| Provider: WatsonX (`granite-13b-instruct-v2`) | **Working** — needs `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` |
| Provider: Brave Search | **Ready** — needs `BRAVE_API_KEY` |
| Provider: Mock (canned lunch data for tests) | **Working** (tests only, not in server build) |
| Orchestrator (parallel fan-out, failure isolation) | **Working** |
| Constraint extraction (`$15`, `within 1 mile`, `4 stars`) | **Working** |
| Intent detection (price / distance / rating / general) | **Working** |
| Ranker — semantic similarity (Ollama embeddings) | **Working** |
| Ranker — three-axis gap scoring (P1/P2/P3) | **Working** |
| Ranker — multi-intent axis weighting | **Working** |
| Ranker — hard constraint filtering | **Working** |
| Ranker — fuzzy consensus clustering | **Working** |
| Ranker — sponsored content penalty | **Working** |
| Query result cache (3-hour TTL, 10 query history) | **Working** |
| `GET /health` | **Working** |
| `GET /metrics` (Prometheus) | **Working** |
| `GET /history` (recent queries) | **Working** |
| `POST /cache/clear` | **Working** |
| Demo UI — ranked results with score bars | **Working** |
| Demo UI — 3D scoring space (Plotly) | **Working** |
| Demo UI — radar chart (top 3 comparison) | **Working** |
| Demo UI — provider breakdown panel | **Working** |
| Demo UI — query history dropdown | **Working** |
| Tests | **23 passing** |

---

## Architecture

```
    user (browser)
          │
          │  POST /query
          ▼
    ┌─────────────────────────────┐
    │     FastAPI server          │   angel_filter/server.py
    │     + Query cache (3hr TTL) │   angel_filter/cache.py
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │       Orchestrator          │   angel_filter/orchestrator.py
    │  1. extract_constraints()   │   angel_filter/constraints.py
    │  2. detect_intent()         │
    │  3. fan-out in parallel     │
    └──┬──────┬──────┬──────┬────┘
       │      │      │      │
   OpenAI  Gemini Ollama WatsonX     angel_filter/providers/*.py
       │      │      │      │        (Brave also available)
       └──────┴──┬───┴──────┘
                 │  normalized ProviderResult list
                 ▼
    ┌─────────────────────────────┐
    │          Ranker             │   angel_filter/ranker.py
    │  1. hard constraint filter  │
    │  2. Ollama embeddings       │
    │     → semantic similarity   │
    │  3. P1/P2/P3 axis scoring   │
    │  4. fuzzy consensus cluster │
    │  5. sponsored penalty       │
    └─────────────┬───────────────┘
                  │  RankedResult list
                  ▼
    ┌─────────────────────────────┐
    │       Demo UI               │   static/index.html
    │  - ranked result cards      │
    │  - score bars               │
    │  - 3D scoring space         │
    │  - radar chart              │
    │  - provider breakdown       │
    │  - query history            │
    └─────────────────────────────┘
```

---

## How scoring works

Every result is scored across four layers:

### 1. Semantic similarity (weight: 50%)
The user's query and each result's title + snippet are embedded using Ollama
(`nomic-embed-text`). Cosine similarity between the query vector and each
result vector produces a 0–1 score. Falls back to keyword overlap when Ollama
is offline.

### 2. Three-axis gap scoring (weight: 35%)

Explicit constraints are extracted from the query and injected into provider
prompts and the ranker:

| Axis | Constraint example | Gap math |
|---|---|---|
| P1 Price | `under $15` | `candidate.price - budget` (negative = under budget) |
| P2 Distance | `within 1 mile` | `candidate.distance - max_distance` (negative = closer) |
| P3 Rating | `rated 4 stars` | `min_rating - candidate.rating` (negative = meets threshold) |

Each gap maps to a 0–1 score. Intent detection (price / distance / rating /
general) shifts the axis weights — a price query gives P1 60% of the axis
score, with P2 and P3 splitting the remaining 40%. All three axes always
contribute — no winner-take-all.

Hard constraint filtering removes results that are more than 25% over budget
or more than 0.5★ below the minimum rating before scoring begins.

### 3. Fuzzy consensus bonus (weight: 15%, capped at 2 providers)
Results mentioned by multiple providers are boosted. Matching uses embedding
cosine similarity ≥ 0.75 so "Joe's Pizza" and "Joe Pizza" cluster together.
Capped at a maximum of 2 extra providers to prevent a mediocre result from
winning just because every provider mentioned it.

### 4. Sponsored penalty (flat −0.20)
Any result flagged as sponsored receives a flat score deduction regardless of
how well it matches the query. This is the thesis of the project.

**Final score formula:**
```
score = 0.50 × similarity
      + 0.35 × axis_score
      + 0.15 × consensus_bonus
      - 0.20 (if sponsored)
```

---

## Setup

**Prerequisites:** Python 3.12 (via pyenv), Ollama running locally.

> Note: Poetry's venv is broken on this machine due to a Homebrew Python 3.14
> update. Use `python3.12` directly until the venv is rebuilt.

```bash
# 1. Clone the repo
git clone https://github.com/adonisja/NLIP-Project
cd NLIP-Project

# 2. Install dependencies
pip3.12 install fastapi uvicorn httpx prometheus-client ollama

# 3. Pull embedding model for semantic ranking
ollama pull nomic-embed-text

# 4. Pull a generation model for the Ollama provider
ollama pull llama3.2

# 5. Copy and fill in your API keys
cp .env.example .env   # then edit .env
```

### Environment variables (`.env`)

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=178e05b2-3352-4f21-8388-572e6b13d65d
WATSONX_REGION=us-east
WATSONX_MODEL=ibm/granite-13b-instruct-v2
OLLAMA_MODEL=llama3.2:latest
BRAVE_API_KEY=...        # optional — get free key at api.search.brave.com
```

Providers are enabled automatically when their key is present. The server
will not start if no providers are configured.

### Starting the server

```bash
./start.sh
```

Then open **http://localhost:8005** in a browser.

---

## Running the tests

```bash
python3.12 -m pytest tests/ -v
```

23 tests covering:
- End-to-end pipeline with all providers
- Sponsored penalty applied and visible in scores
- Provider failure isolation
- Budget constraint filtering (`$15` pushes `$28` bistro out)
- Distance intent favors nearest result
- Rating intent favors highest-rated result
- Axis scores present and in 0–1 range on all results
- Consensus bonus applied when two providers agree
- Intent detection for all four intent types (8 parametrized cases)
- Constraint extraction from natural language (7 parametrized cases)

No tests require network or Ollama — they use the mock provider and
keyword-fallback ranker, making them fast and deterministic.

---

## Demo queries

| Query | What it demonstrates |
|---|---|
| `lunch under $15` | Budget constraint + price intent |
| `best rated lunch spots near me` | Rating + distance intent together |
| `Find me the top 3 lunch spots under $15, within 1 mile, rated at least 4 stars` | All three axes, hard filter, constraint injection |
| Run any query twice | Cache hit — instant response, "from cache" badge |

---

## Project layout

```
angel_filter/
  server.py             # FastAPI server + provider wiring
  orchestrator.py       # parallel fan-out + ranker call
  ranker.py             # scoring: similarity + axis + consensus + penalty
  constraints.py        # natural language constraint extraction
  prompt.py             # shared prompt builder for AI providers
  cache.py              # in-memory query cache (3-hour TTL)
  providers/
    base.py             # BaseProvider, ProviderResult, ProviderError
    openai_provider.py  # OpenAI gpt-4o-mini
    gemini.py           # Google Gemini
    ollama_provider.py  # Local Ollama (llama3.2)
    watsonx.py          # IBM WatsonX
    brave.py            # Brave Search API
    mock.py             # canned lunch data (tests only)
static/
  index.html            # demo UI (results + 3D plot + radar chart)
  plotly.min.js         # Plotly served locally (gitignored, download once)
tests/
  test_orchestrator.py  # 23 tests
start.sh                # starts server on port 8005, loads .env
pyproject.toml
README.md
```

---

## License

Apache-2.0 (matches the upstream NLIP projects).
