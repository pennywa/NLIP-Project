import argparse
import asyncio
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONSENSUS_NAME_SIMILARITY = 0.82
CONSENSUS_EMBEDDING_DISTANCE = 0.9
REQUEST_TIMEOUT_SECONDS = 45
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def load_env_file(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def discover_ollama_model(base_url: str) -> str:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        if models:
            return str(models[0].get("name", "")).strip() or "llama3.1:8b"
    except requests.RequestException:
        pass
    return "llama3.1:8b"


def normalize_gemini_model_name(model_name: str) -> str:
    return model_name.removeprefix("models/")


def discover_gemini_model() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "gemini-2.5-flash"

    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        response.raise_for_status()
        models = response.json().get("models", [])
    except requests.RequestException:
        return "gemini-2.5-flash"

    preferred_models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
    ]
    available_names = {
        model.get("name", "")
        for model in models
        if "generateContent" in model.get("supportedGenerationMethods", [])
    }
    for preferred_model in preferred_models:
        if preferred_model in available_names:
            return normalize_gemini_model_name(preferred_model)

    for model in models:
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        name = str(model.get("name", "")).strip()
        if "tts" in name or "image" in name:
            continue
        return normalize_gemini_model_name(name)

    return "gemini-2.5-flash"


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL") or discover_gemini_model()
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or discover_ollama_model(DEFAULT_OLLAMA_URL)


@dataclass
class QueryConstraints:
    budget: Optional[float]
    max_distance_miles: Optional[float]
    min_rating: Optional[float]


@dataclass
class Candidate:
    name: str
    price: Optional[float]
    distance_miles: Optional[float]
    rating: Optional[float]
    notes: str
    source_model: str


@dataclass
class ScoredCandidate:
    candidate: Candidate
    point_3d: List[float]
    origin_distance: float
    query_distance: float
    cluster_distance: float
    consensus_size: int
    consensus_members: List[str]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLIP Angel Filter consensus scorer")
    parser.add_argument("query", help="User query to send to upstream models")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many candidates each upstream model should return",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the Ollama server",
    )
    return parser.parse_args()


def build_structured_prompt(user_query: str, top_k: int) -> str:
    return f"""
Return the best lunch recommendations for the user query below.

User query:
{user_query}

Respond as JSON only with this exact schema:
{{
  "query_summary": "short explanation",
  "candidates": [
    {{
      "name": "place name",
      "price": 12.5,
      "distance_miles": 0.8,
      "rating": 4.4,
      "notes": "why this fits"
    }}
  ]
}}

Rules:
- Return up to {top_k} candidates.
- Use numeric values for price, distance_miles, and rating whenever possible.
- If a value is unknown, use null.
- Focus on price, distance, and rating.
- Do not include markdown, code fences, or extra text.
""".strip()


def normalize_name(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    tokens = [
        token
        for token in compact.split()
        if token not in {"the", "and", "restaurant", "grill", "cafe"}
    ]
    return " ".join(tokens)


def tokenize_name(name: str) -> set[str]:
    return set(normalize_name(name).split())


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = tokenize_name(left)
    right_tokens = tokenize_name(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(raw_text[start : end + 1])


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_query_constraints(query: str) -> QueryConstraints:
    budget_match = re.search(r"(?:budget|under|below|less than|at most)\s*(?:is\s*)?\$?(\d+(?:\.\d+)?)", query, re.I)
    distance_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(mile|miles|mi|minute|minutes|min)\b",
        query,
        re.I,
    )
    rating_match = re.search(r"(?:rating|rated|stars?|above|at least)\s*(\d(?:\.\d+)?)", query, re.I)

    max_distance_miles: Optional[float] = None
    if distance_match:
        distance_value = float(distance_match.group(1))
        unit = distance_match.group(2).lower()
        max_distance_miles = (
            distance_value if unit.startswith("mile") or unit == "mi" else distance_value / 20.0
        )

    min_rating = float(rating_match.group(1)) if rating_match else 4.0
    budget = float(budget_match.group(1)) if budget_match else None
    return QueryConstraints(budget=budget, max_distance_miles=max_distance_miles, min_rating=min_rating)


def candidate_to_text(candidate: Candidate) -> str:
    return (
        f"{candidate.name}. Price ${candidate.price if candidate.price is not None else 'unknown'}. "
        f"Distance {candidate.distance_miles if candidate.distance_miles is not None else 'unknown'} miles. "
        f"Rating {candidate.rating if candidate.rating is not None else 'unknown'}. "
        f"Notes: {candidate.notes}"
    )


def build_point_3d(candidate: Candidate, constraints: QueryConstraints, population: Sequence[Candidate]) -> List[float]:
    max_price = max((item.price for item in population if item.price is not None), default=25.0)
    max_distance = max((item.distance_miles for item in population if item.distance_miles is not None), default=5.0)

    target_budget = constraints.budget if constraints.budget is not None else max_price
    target_distance = constraints.max_distance_miles if constraints.max_distance_miles is not None else max_distance
    target_rating = constraints.min_rating if constraints.min_rating is not None else 4.0

    price = candidate.price if candidate.price is not None else max_price
    distance = candidate.distance_miles if candidate.distance_miles is not None else max_distance
    rating = candidate.rating if candidate.rating is not None else 0.0

    price_gap = max(0.0, price - target_budget) / max(target_budget, 1.0)
    distance_gap = max(0.0, distance - target_distance) / max(target_distance, 0.25)
    rating_gap = max(0.0, target_rating - rating) / 5.0
    return [round(price_gap, 4), round(distance_gap, 4), round(rating_gap, 4)]


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def fetch_gemini(prompt: str, model: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = GEMINI_URL_TEMPLATE.format(model=normalize_gemini_model_name(model), api_key=api_key)
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json_object(content)


def fetch_ollama(prompt: str, model: str, base_url: str) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("response", "")
    return extract_json_object(content)


async def fetch_model_outputs(
    query: str,
    gemini_model: str,
    ollama_model: str,
    ollama_url: str,
    top_k: int,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    prompt = build_structured_prompt(query, top_k)
    task_map = {
        "gemini": asyncio.to_thread(fetch_gemini, prompt, gemini_model),
        "ollama": asyncio.to_thread(fetch_ollama, prompt, ollama_model, ollama_url),
    }
    results = await asyncio.gather(*task_map.values(), return_exceptions=True)

    raw_outputs: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for provider_name, result in zip(task_map.keys(), results):
        if isinstance(result, Exception):
            errors[provider_name] = str(result)
            continue
        raw_outputs[provider_name] = result

    if len(raw_outputs) < 2:
        raise RuntimeError(
            "Angel Filter requires at least two successful model responses. "
            f"Succeeded: {list(raw_outputs.keys())}. Errors: {errors}"
        )
    return raw_outputs, errors


def parse_candidates(model_name: str, payload: Dict[str, Any]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for raw_candidate in payload.get("candidates", []):
        candidates.append(
            Candidate(
                name=str(raw_candidate.get("name", "")).strip(),
                price=parse_float(raw_candidate.get("price")),
                distance_miles=parse_float(raw_candidate.get("distance_miles")),
                rating=parse_float(raw_candidate.get("rating")),
                notes=str(raw_candidate.get("notes", "")).strip(),
                source_model=model_name,
            )
        )
    return [candidate for candidate in candidates if candidate.name]


def build_consensus_groups(candidates: Sequence[Candidate], name_embeddings: np.ndarray) -> List[List[int]]:
    groups: List[List[int]] = []
    for index, candidate in enumerate(candidates):
        matched_group: Optional[List[int]] = None
        for group in groups:
            representative = candidates[group[0]]
            similarity = jaccard_similarity(candidate.name, representative.name)
            embedding_distance = euclidean_distance(name_embeddings[index], name_embeddings[group[0]])
            if similarity >= CONSENSUS_NAME_SIMILARITY or embedding_distance <= CONSENSUS_EMBEDDING_DISTANCE:
                matched_group = group
                break
        if matched_group is None:
            groups.append([index])
        else:
            matched_group.append(index)
    return groups


def score_candidates(query: str, candidates: Sequence[Candidate], model: SentenceTransformer) -> List[ScoredCandidate]:
    constraints = parse_query_constraints(query)
    query_embedding = model.encode([query], normalize_embeddings=False)[0]
    candidate_texts = [candidate_to_text(candidate) for candidate in candidates]
    candidate_embeddings = np.asarray(model.encode(candidate_texts, normalize_embeddings=False))
    name_embeddings = np.asarray(model.encode([candidate.name for candidate in candidates], normalize_embeddings=False))
    groups = build_consensus_groups(candidates, name_embeddings)

    scored: List[ScoredCandidate] = []
    for group in groups:
        if len(group) < 2:
            continue
        cluster_vectors = candidate_embeddings[group]
        cluster_center = np.mean(cluster_vectors, axis=0)
        members = [candidates[index] for index in group]
        for index in group:
            point_3d = build_point_3d(candidates[index], constraints, candidates)
            query_distance = euclidean_distance(query_embedding, candidate_embeddings[index])
            cluster_distance = euclidean_distance(cluster_center, candidate_embeddings[index])
            origin_distance = math.sqrt(sum(axis * axis for axis in point_3d))
            score = (0.55 * query_distance) + (0.25 * cluster_distance) + (0.20 * origin_distance)
            scored.append(
                ScoredCandidate(
                    candidate=candidates[index],
                    point_3d=point_3d,
                    origin_distance=round(origin_distance, 6),
                    query_distance=round(query_distance, 6),
                    cluster_distance=round(cluster_distance, 6),
                    consensus_size=len(group),
                    consensus_members=[member.name for member in members],
                    score=round(score, 6),
                )
            )

    if scored:
        return sorted(scored, key=lambda item: item.score)

    fallback_scored: List[ScoredCandidate] = []
    cluster_center = np.mean(candidate_embeddings, axis=0)
    for index, candidate in enumerate(candidates):
        point_3d = build_point_3d(candidate, constraints, candidates)
        query_distance = euclidean_distance(query_embedding, candidate_embeddings[index])
        cluster_distance = euclidean_distance(cluster_center, candidate_embeddings[index])
        origin_distance = math.sqrt(sum(axis * axis for axis in point_3d))
        score = (0.7 * query_distance) + (0.15 * cluster_distance) + (0.15 * origin_distance)
        fallback_scored.append(
            ScoredCandidate(
                candidate=candidate,
                point_3d=point_3d,
                origin_distance=round(origin_distance, 6),
                query_distance=round(query_distance, 6),
                cluster_distance=round(cluster_distance, 6),
                consensus_size=1,
                consensus_members=[candidate.name],
                score=round(score, 6),
            )
        )
    return sorted(fallback_scored, key=lambda item: item.score)


def build_output(
    query: str,
    constraints: QueryConstraints,
    raw_outputs: Dict[str, Dict[str, Any]],
    provider_errors: Dict[str, str],
    scored: Sequence[ScoredCandidate],
) -> Dict[str, Any]:
    best = scored[0]
    return {
        "query": query,
        "query_constraints": asdict(constraints),
        "reference_models": list(raw_outputs.keys()),
        "provider_errors": provider_errors,
        "raw_model_outputs": raw_outputs,
        "angel_filtered": {
            "name": best.candidate.name,
            "price": best.candidate.price,
            "distance_miles": best.candidate.distance_miles,
            "rating": best.candidate.rating,
            "notes": best.candidate.notes,
            "source_model": best.candidate.source_model,
            "point_3d": best.point_3d,
            "origin_distance": best.origin_distance,
            "score": best.score,
            "query_distance": best.query_distance,
            "cluster_distance": best.cluster_distance,
            "consensus_size": best.consensus_size,
            "consensus_members": best.consensus_members,
        },
        "ranked_candidates": [
            {
                "name": item.candidate.name,
                "price": item.candidate.price,
                "distance_miles": item.candidate.distance_miles,
                "rating": item.candidate.rating,
                "notes": item.candidate.notes,
                "source_model": item.candidate.source_model,
                "point_3d": item.point_3d,
                "origin_distance": item.origin_distance,
                "score": item.score,
                "query_distance": item.query_distance,
                "cluster_distance": item.cluster_distance,
                "consensus_size": item.consensus_size,
                "consensus_members": item.consensus_members,
            }
            for item in scored
        ],
    }


def run_filter(
    query: str,
    top_k: int = 3,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    raw_outputs, provider_errors = asyncio.run(
        fetch_model_outputs(query, gemini_model, ollama_model, ollama_url, top_k)
    )

    candidates: List[Candidate] = []
    for model_name, payload in raw_outputs.items():
        candidates.extend(parse_candidates(model_name, payload))

    if not candidates:
        raise RuntimeError("No candidates were returned by the upstream models")

    constraints = parse_query_constraints(query)
    scored = score_candidates(query, candidates, get_embedding_model())
    return build_output(query, constraints, raw_outputs, provider_errors, scored)


def main() -> None:
    args = parse_args()
    output = run_filter(
        args.query,
        top_k=args.top_k,
        gemini_model=args.gemini_model,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()