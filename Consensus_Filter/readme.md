# Consensus Filter

A Flask web app that sends a query to both **Gemini** (Google) and **Ollama** (local), embeds the responses using `sentence-transformers`, and filters them to the most semantically consistent results. Includes a Manim animation visualizing the pipeline.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- A Google Gemini API key

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in this directory:

```
GEMINI_API_KEY=your_key_here
```

## Run the Filter site

```powershell
python app.py
```

Open http://127.0.0.1:8000

## Render the pipeline animation

```powershell
manim pipeline_animation.py ConsensusFilterScene -ql
```

Output will be at `media/videos/pipeline_animation/480p15/ConsensusFilterScene.mp4`

To play it:

```powershell
Start-Process .\media\videos\pipeline_animation\480p15\ConsensusFilterScene.mp4
```



