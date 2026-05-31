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

Choose your operating system below. You need at least one API key to run the server — Gemini has a free tier and is the easiest to get started with.

---

### Mac

#### Requirements
- macOS 11 or later
- [Homebrew](https://brew.sh) (package manager)
- Python 3.12
- [Ollama](https://ollama.com) (local AI — free, no key needed)
- Git
- At least one API key (see [API Keys](#api-keys) below)

#### Step-by-step

**1. Install Homebrew** (skip if already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install Python 3.12 and Git**
```bash
brew install python@3.12 git
```

Verify:
```bash
python3.12 --version   # should print Python 3.12.x
git --version
```

**3. Install Ollama**

Download from **https://ollama.com/download** and run the installer.

Then pull the two models Angel Filter needs:
```bash
ollama pull nomic-embed-text   # embedding model — used for ranking
ollama pull llama3.2           # generation model — used as a provider
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```
You should see a JSON list of installed models.

**4. Clone the repo**
```bash
git clone https://github.com/adonisja/NLIP-Project
cd NLIP-Project
```

**5. Install Python dependencies**
```bash
pip3.12 install fastapi "uvicorn[standard]" httpx prometheus-client ollama python-dotenv
```

**6. Download Plotly** (required for the 3D visualization)
```bash
curl -o static/plotly.min.js https://cdn.plot.ly/plotly-2.32.0.min.js
```

**7. Set up your API keys**

Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Then open `.env` in any text editor and fill in your keys (see [API Keys](#api-keys) below).

**8. Start the server**
```bash
./start.sh
```

Open **http://localhost:8005** in your browser.

---

### Windows

#### Requirements
- Windows 10 or 11
- [Python 3.12](https://www.python.org/downloads/) (check "Add to PATH" during install)
- [Ollama for Windows](https://ollama.com/download)
- [Git for Windows](https://git-scm.com/download/win)
- At least one API key (see [API Keys](#api-keys) below)

#### Step-by-step

**1. Install Python 3.12**

Download from **https://www.python.org/downloads/release/python-3120/**

During installation, check **"Add python.exe to PATH"** — this is important.

Verify in a new terminal (Command Prompt or PowerShell):
```
python --version   # should print Python 3.12.x
pip --version
```

**2. Install Git**

Download from **https://git-scm.com/download/win** and run the installer with default settings.

**3. Install Ollama**

Download from **https://ollama.com/download** and run the installer.

Open a new terminal and pull the two models:
```
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify Ollama is running:
```
curl http://localhost:11434/api/tags
```

**4. Clone the repo**
```
git clone https://github.com/adonisja/NLIP-Project
cd NLIP-Project
```

**5. Install Python dependencies**
```
pip install fastapi "uvicorn[standard]" httpx prometheus-client ollama python-dotenv
```

**6. Download Plotly**

In PowerShell:
```powershell
Invoke-WebRequest -Uri "https://cdn.plot.ly/plotly-2.32.0.min.js" -OutFile "static\plotly.min.js"
```

**7. Set up your API keys**

Copy the example env file:
```
copy .env.example .env
```
Open `.env` in Notepad or VS Code and fill in your keys.

**8. Start the server**

On Windows, `start.sh` won't work directly. Run this instead:
```
python -m uvicorn angel_filter.server:app --reload --port 8005
```

Or if you have Git Bash installed:
```bash
./start.sh
```

Open **http://localhost:8005** in your browser.

---

### API Keys

You need **at least one** of the following. The server auto-detects which keys are present and enables those providers.

| Provider | Key name | Where to get it | Cost |
|---|---|---|---|
| Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free tier available |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Free trial credits |
| WatsonX | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` | [cloud.ibm.com](https://cloud.ibm.com) | Free tier available |
| Brave Search | `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) | 2,000 free queries/month |
| Ollama | *(no key needed)* | Runs locally after install | Free |

Create a `.env` file in the project root with your keys:

```
# Required — at least one AI provider
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# WatsonX (needs both values)
WATSONX_API_KEY=your-key-here
WATSONX_PROJECT_ID=your-project-id-here
WATSONX_REGION=us-east
WATSONX_MODEL=ibm/granite-13b-instruct-v2

# Ollama (no key — just set the model name)
OLLAMA_MODEL=llama3.2:latest

# Optional
BRAVE_API_KEY=your-key-here
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.
> Each contributor creates their own `.env` locally.

---

### Verifying your setup

After starting the server, check that providers loaded correctly:

```bash
curl http://localhost:8005/health
```

You should see something like:
```json
{
  "ok": true,
  "mode": "fallback",
  "providers": ["openai", "gemini", "ollama"],
  "uptime_seconds": 5.1
}
```

If `providers` is empty, check your `.env` file and make sure the keys are set correctly.

Run the test suite (no network or API keys needed):
```bash
# Mac
python3.12 -m pytest tests/ -v

# Windows
python -m pytest tests/ -v
```

All 23 tests should pass.

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
