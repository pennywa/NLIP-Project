# Query Pipeline Technical Report

## Purpose

This document explains the current NLIP Angel Filter pipeline from query input to final response output. It describes the actual implementation in this workspace, including the backend scoring path, provider orchestration, JSON normalization, ranking logic, and frontend visualization.

The current system compares two models:

- Gemini as the external reference model
- Ollama as the local reference model

The pipeline enforces a minimum of two successful model responses before it will produce a final filtered result.

## System Components

The system is split into two main runtime layers:

- `angel_filter.py`
  - Core pipeline logic
  - Environment loading
  - Provider discovery
  - Query parsing
  - Parallel model calls
  - Candidate normalization
  - Embedding and consensus scoring
  - Final JSON output construction
- `app.py`
  - Flask web server
  - Serves the frontend
  - Exposes `/api/filter` and `/api/health`
- `templates/index.html`
  - Browser UI shell
- `static/app.js`
  - Frontend orchestration
  - Sends the query to the Flask API
  - Renders winner card, ranked candidates, raw JSON, and 3D plot

## High-Level Flow

The query pipeline is:

1. A user submits a query through either the CLI or the web form.
2. The query is passed into `run_filter(...)` in `angel_filter.py`.
3. A structured prompt is built that forces both models to return the same JSON schema.
4. Gemini and Ollama are called in parallel.
5. Each model response is parsed into a normalized list of candidates.
6. Query constraints are extracted from the user input.
7. Each candidate is projected into a 3D explicit feature space:
   - X axis: price gap
   - Y axis: distance gap
   - Z axis: rating gap
8. Query and candidate texts are embedded using a local sentence-transformer.
9. Candidate names are grouped into consensus clusters using token overlap and embedding distance.
10. Candidates are scored using a weighted combination of:
    - semantic distance to the query
    - distance to the consensus cluster center
    - Euclidean distance from the origin in the explicit 3D `P` space
11. The best candidate is selected as the Angel Filter result.
12. A final JSON object is returned.
13. The frontend renders cards, raw JSON, and a 3D visualization.

## Entry Paths

### CLI path

The CLI path runs:

```bash
python angel_filter.py "Find three top places for lunch and the budget is $15"
```

This path:

- parses command-line arguments
- calls `run_filter(...)`
- prints the full JSON result to stdout

### Web path

The web path runs:

```bash
python app.py
```

The browser workflow is:

1. The user opens the Flask app.
2. The form in `index.html` captures query parameters.
3. `static/app.js` sends the payload to `POST /api/filter`.
4. `app.py` validates inputs and calls `run_filter(...)`.
5. The resulting JSON is returned to the browser.
6. The frontend renders the structured result.

## Environment and Runtime Configuration

### `.env` loading

The backend does not currently depend on `python-dotenv`. Instead, `angel_filter.py` defines `load_env_file(...)`, which:

- opens `.env` if present
- ignores blank lines and comments
- parses `KEY=VALUE` pairs
- adds them to `os.environ` only if they are not already set

This allows shell-defined environment variables to override `.env` values.

### Provider defaults

The system resolves provider defaults at import time.

#### Gemini

`discover_gemini_model()`:

- uses `GEMINI_API_KEY`
- requests the Gemini model catalog from the Google API
- filters to models supporting `generateContent`
- prefers:
  - `gemini-2.5-flash`
  - `gemini-2.0-flash`
  - `gemini-flash-latest`
- falls back to the first compatible text model
- finally falls back to `gemini-2.5-flash` if discovery fails

`normalize_gemini_model_name(...)` removes the `models/` prefix if present.

#### Ollama

`discover_ollama_model(...)`:

- requests `GET /api/tags` from the local Ollama server
- selects the first installed model if available
- falls back to `llama3.1:8b` if discovery fails

The default Ollama URL is:

```text
http://localhost:11434
```

## Input Stage

### User query

The raw user query is the only semantic input that drives the system. Example:

```text
Find three top places for lunch and the budget is $15
```

### Top-K parameter

`top_k` controls how many candidates each provider is asked to return.

The Flask API constrains it to:

- minimum: `1`
- maximum: `10`

### Structured prompt construction

The function `build_structured_prompt(user_query, top_k)` builds a provider-agnostic instruction block. The prompt requires:

- JSON only
- a consistent schema
- numeric values for `price`, `distance_miles`, and `rating` where possible
- `null` for unknown values
- no markdown or surrounding explanation

This is critical because the downstream logic assumes both providers produce a normalized candidate array.

## Provider Execution Stage

### Parallelization model

The backend uses `asyncio.gather(...)` with `asyncio.to_thread(...)` to execute blocking `requests` calls concurrently.

This means:

- Gemini and Ollama are queried at the same time
- the backend waits for both to complete
- exceptions are captured per provider rather than crashing immediately

### Gemini call

`fetch_gemini(...)` performs:

- `POST` to the Google `generateContent` endpoint
- `responseMimeType = application/json`
- low temperature (`0.2`) for more deterministic structure

The response path is:

- `payload["candidates"][0]["content"]["parts"][0]["text"]`

That text is then passed to `extract_json_object(...)`.

### Ollama call

`fetch_ollama(...)` performs:

- `POST` to `http://localhost:11434/api/generate`
- `stream = false`
- `format = json`
- low temperature (`0.2`)

The response text is read from:

- `payload["response"]`

That text is then passed to `extract_json_object(...)`.

### Minimum provider success rule

`fetch_model_outputs(...)` tracks:

- `raw_outputs`: successful provider responses
- `errors`: provider-specific failures

If fewer than two providers succeed, the backend raises:

```text
Angel Filter requires at least two successful model responses.
```

This rule preserves the core design principle that filtering requires intersection across at least two models.

## Response Normalization Stage

### JSON extraction

`extract_json_object(raw_text)`:

- finds the first `{`
- finds the last `}`
- slices that substring
- parses it using `json.loads(...)`

This is a defensive step against model output that may wrap JSON with stray text.

### Candidate parsing

`parse_candidates(model_name, payload)` transforms raw provider output into `Candidate` objects.

Each candidate contains:

- `name`
- `price`
- `distance_miles`
- `rating`
- `notes`
- `source_model`

### Numeric normalization

`parse_float(...)` accepts:

- integers
- floats
- strings containing numeric tokens

Examples:

- `"12.50" -> 12.5`
- `"about 0.8 miles" -> 0.8`
- `null -> None`

This softens provider inconsistencies while keeping the internal representation numeric.

## Query Constraint Extraction Stage

The system extracts explicit user constraints from the natural-language query using regular expressions.

`parse_query_constraints(...)` produces:

- `budget`
- `max_distance_miles`
- `min_rating`

### Budget extraction

Supported patterns include words like:

- `budget`
- `under`
- `below`
- `less than`
- `at most`

Example:

- `budget is $15` -> `budget = 15.0`

### Distance extraction

The parser supports:

- miles
- mi
- minutes
- min

If the user gives minutes, the current implementation approximates:

```text
20 minutes = 1 mile
```

This is a heuristic, not a geospatial model.

### Rating extraction

If no explicit rating is found, the current default is:

```text
4.0
```

That means the system implicitly treats four stars as a baseline trust threshold unless the query says otherwise.

## Explicit 3D `P` Space Construction

The pipeline maps each candidate into a three-dimensional explicit feature space:

```text
P = [price_gap, distance_gap, rating_gap]
```

These values are generated by `build_point_3d(...)`.

### Axis semantics

- `price_gap`
  - how much the candidate exceeds the target budget
- `distance_gap`
  - how much the candidate exceeds the target distance
- `rating_gap`
  - how much the candidate falls below the target rating

All three axes are non-negative. A perfect explicit match is:

```text
[0, 0, 0]
```

### Origin interpretation

The system treats the user query as the origin of this space. A candidate closer to the origin is closer to the user’s explicit requested constraints.

### Normalization rules

The explicit gaps are normalized so the dimensions are comparable:

- price gap divided by target budget
- distance gap divided by target distance
- rating gap divided by `5.0`

This prevents one raw unit scale from dominating the others.

### Missing value behavior

If a candidate is missing a field:

- missing price falls back to the maximum known price in the candidate population
- missing distance falls back to the maximum known distance in the candidate population
- missing rating falls back to `0.0`

This penalizes uncertainty rather than rewarding incomplete data.

## Semantic Embedding Stage

The explicit 3D `P` vector is only part of the ranking. The system also computes semantic similarity.

### Embedding model

The local embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is loaded through `get_embedding_model()` and cached using `lru_cache(maxsize=1)` so the model is only loaded once per process.

### Embedded texts

The system embeds three text surfaces:

- the raw user query
- each candidate converted into a descriptive sentence via `candidate_to_text(...)`
- each candidate name by itself for cluster grouping

### Purpose of each embedding set

- query embedding
  - measures semantic closeness between candidate descriptions and the user query
- candidate description embeddings
  - support semantic ranking
- candidate name embeddings
  - support consensus grouping between near-duplicate place names

## Consensus Construction Stage

The Angel Filter is not just semantic ranking. It also tries to find an intersection across providers.

### Candidate grouping

`build_consensus_groups(...)` groups candidates if either of these tests passes:

1. Jaccard similarity on normalized tokens is at least `0.82`
2. embedding distance between names is at most `0.9`

This is intended to catch cases like:

- `Joe's Pizza`
- `Joes Pizza`
- `Joe Pizza`

even if the provider spellings differ slightly.

### Name normalization

Before Jaccard comparison:

- punctuation is stripped
- text is lowercased
- common filler tokens are removed, including:
  - `the`
  - `and`
  - `restaurant`
  - `grill`
  - `cafe`

## Ranking Stage

### Consensus-first scoring

If a group contains at least two items, it is considered a consensus cluster.

For each candidate in such a cluster, the backend computes:

- `query_distance`
  - Euclidean distance between query embedding and candidate embedding
- `cluster_distance`
  - Euclidean distance between candidate embedding and cluster centroid
- `origin_distance`
  - Euclidean norm of the explicit 3D `P` vector

The main score is:

$$
\text{score} = 0.55 \cdot \text{query\_distance} + 0.25 \cdot \text{cluster\_distance} + 0.20 \cdot \text{origin\_distance}
$$

Interpretation:

- semantic match to the query has the highest weight
- agreement with the cluster is second
- explicit constraint compliance is third

### Fallback scoring

If no consensus cluster of size 2 or greater is found, the system still returns a ranking instead of failing.

Fallback score:

$$
\text{score} = 0.70 \cdot \text{query\_distance} + 0.15 \cdot \text{cluster\_distance} + 0.15 \cdot \text{origin\_distance}
$$

In fallback mode:

- each candidate is effectively a singleton cluster
- semantic similarity becomes even more dominant

### Winner selection

The final Angel Filter result is simply the scored candidate with the lowest `score` after sorting ascending.

## Output Construction Stage

The final JSON is built by `build_output(...)`.

The output contains:

- `query`
- `query_constraints`
- `reference_models`
- `provider_errors`
- `raw_model_outputs`
- `angel_filtered`
- `ranked_candidates`

### `angel_filtered`

This is the top-ranked candidate and includes:

- normalized 3D point
- origin distance
- semantic query distance
- cluster distance
- consensus size
- consensus member names

### `ranked_candidates`

This includes all scored candidates in sorted order.

This is important because the frontend graph does not just show the winner. It shows the entire ranked field.

## Flask API Stage

The Flask service in `app.py` adds a thin HTTP wrapper around the scoring engine.

### `GET /api/health`

Returns:

- status
- resolved Gemini model
- resolved Ollama model
- Ollama base URL

This is a runtime configuration sanity endpoint.

### `POST /api/filter`

Input fields:

- `query`
- `top_k`
- `gemini_model`
- `ollama_model`
- `ollama_url`

Validation behavior:

- missing query returns HTTP `400`
- invalid `top_k` returns HTTP `400`
- backend provider or scoring failures return HTTP `502`

On success, the endpoint returns the full pipeline JSON.

## Frontend Rendering Stage

The browser UI is a visualization layer over the pipeline JSON.

### Form submission

`static/app.js`:

- reads the query form
- serializes the values into JSON
- posts to `/api/filter`
- receives the structured response

### Rendered outputs

The frontend renders:

- winner card
- ranked candidate cards
- raw JSON block
- 3D Plotly graph

### Plot semantics

The graph shows:

- the user query origin
- one point for each ranked candidate
- one line from the origin to each candidate point
- color coding by provider
- a separate highlight for the winning point

### Display-space expansion

The current frontend intentionally expands near-origin points in display space when they are too tightly clustered. This is a visualization technique only.

Important distinction:

- actual scoring uses the true `point_3d` values from the backend
- displayed plot positions may be spread for readability
- hover values show the actual underlying gap values

This avoids a common plotting problem where near-zero candidates overlap and become unreadable.

## Failure Modes and Operational Constraints

### Two-provider dependency

The current system requires both Gemini and Ollama to succeed. If one provider fails, no final result is produced.

### Provider schema dependence

The backend assumes the structured prompt will produce a parseable JSON object. If either provider emits malformed JSON, extraction fails.

### Regex constraint extraction limits

The query parser is intentionally lightweight. It handles common patterns but does not perform deep semantic parsing.

### Distance heuristic limits

The conversion from minutes to miles is a heuristic and not location-aware.

### Candidate truthfulness

The pipeline currently trusts provider-supplied price, distance, and rating values. It does not independently verify them against an external place database.

### Consensus limitations

Consensus is based on candidate overlap between two models, not on formal truth verification. Agreement between providers reduces some noise, but it is not proof of correctness.

## Technical Summary

The current Angel Filter implementation combines two ranking layers:

1. Explicit constraint matching in a 3D `P` space derived from price, distance, and rating
2. Semantic agreement in embedding space across both the user query and cross-model candidate clusters

The system chooses the candidate that is both:

- semantically close to the user’s request
- sufficiently close to the cross-model consensus cluster
- close to the explicit origin defined by the user’s requested constraints

This produces a filtered answer that is not simply the first response from one model, but the lowest-scoring candidate under a combined consensus-and-origin objective.

## Practical Query-to-Response Narrative

For a query like:

```text
Find three top places for lunch and the budget is $15
```

the live path is:

1. The query enters the Flask API or CLI.
2. The backend asks Gemini and Ollama for up to three structured lunch candidates.
3. Each provider returns a JSON array of places with price, distance, rating, and notes.
4. Those outputs are parsed into internal `Candidate` objects.
5. The backend infers a budget target of `$15`.
6. Each candidate is assigned a 3D gap vector relative to that target.
7. Candidate descriptions are embedded and compared to the original query.
8. Candidate names are grouped into cross-model consensus clusters.
9. Each candidate in a consensus cluster is scored.
10. The lowest score becomes the final Angel Filter result.
11. The backend returns full JSON.
12. The frontend renders the winner, ranking table, and graph.

## Recommended Next Technical Extensions

The current implementation is functional, but the next logical improvements would be:

1. Add a verified place-data layer so provider claims about price, rating, and distance can be checked.
2. Add geospatial computation instead of the current minute-to-mile heuristic.
3. Add persistent query history and scoring traces for auditability.
4. Add configurable weighting for semantic distance versus origin distance.
5. Add support for three or more providers so consensus can be modeled as 2-of-3 or 3-of-3 agreement.