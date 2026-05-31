import os

from flask import Flask, jsonify, render_template, request

from angel_filter import DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, run_filter


app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        default_top_k=3,
        default_gemini_model=DEFAULT_GEMINI_MODEL,
        default_ollama_model=DEFAULT_OLLAMA_MODEL,
        default_ollama_url=DEFAULT_OLLAMA_URL,
    )


@app.get("/api/health")
def health() -> tuple:
    return (
        jsonify(
            {
                "status": "ok",
                "gemini_model": DEFAULT_GEMINI_MODEL,
                "ollama_model": DEFAULT_OLLAMA_MODEL,
                "ollama_url": DEFAULT_OLLAMA_URL,
            }
        ),
        200,
    )


@app.post("/api/filter")
def filter_query() -> tuple:
    payload = request.get_json(silent=True) or request.form
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    top_k_value = payload.get("top_k", 3)
    try:
        top_k = max(1, min(int(top_k_value), 10))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    gemini_model = str(payload.get("gemini_model", DEFAULT_GEMINI_MODEL)).strip() or DEFAULT_GEMINI_MODEL
    ollama_model = str(payload.get("ollama_model", DEFAULT_OLLAMA_MODEL)).strip() or DEFAULT_OLLAMA_MODEL
    ollama_url = str(payload.get("ollama_url", DEFAULT_OLLAMA_URL)).strip() or DEFAULT_OLLAMA_URL

    try:
        result = run_filter(
            query,
            top_k=top_k,
            gemini_model=gemini_model,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=True)