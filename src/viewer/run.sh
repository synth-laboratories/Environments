#!/bin/bash
# Start both backend and frontend for the trace viewer

echo "🚀 Starting Synth Trace Viewer v2..."

# Kill any existing processes on our ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start backend in background
echo "Starting backend on port 8000..."
cd "$(dirname "$0")"
uv run python -m uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend on port 3000..."
cd frontend
if [ ! -d ".web" ]; then
    echo "Initializing Reflex app..."
    uv run reflex init
fi

# Run frontend (this will block)
uv run reflex run

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null || true