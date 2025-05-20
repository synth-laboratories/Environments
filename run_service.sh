#!/bin/bash

# This script is intended to be run from the root of the "Environments" project directory.

echo "Ensuring current directory is script's directory: $(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" &> /dev/null && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null

PROJECT_ROOT="$(pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "Starting Environment Service from $PROJECT_ROOT using Uvicorn directly with venv python..."

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python interpreter not found in .venv at $VENV_PYTHON"
    echo "Please ensure the virtual environment exists and is populated (e.g., run 'uv venv' and 'uv sync')"
    exit 1
fi

# Run uvicorn using the Python interpreter from the .venv
"$VENV_PYTHON" -m uvicorn service.app:app --host 0.0.0.0 --port 8000

echo "Environment Service stopped." 