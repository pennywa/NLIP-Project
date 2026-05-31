#!/bin/bash
# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
PORT=8005
python3.12 -m uvicorn angel_filter.server:app --reload --port $PORT
