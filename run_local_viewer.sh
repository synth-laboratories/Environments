#!/bin/bash
# Run the local trace viewer with Reflex only

echo "🚀 Starting Synth Trace Viewer (Reflex Only)..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Store the project root directory
PROJECT_ROOT=$(pwd)

# Navigate to viewer directory
cd src/viewer

# Check if DuckDB database exists, if not run migration
if [ ! -f "synth_eval.duckdb" ]; then
    echo "📊 Database not found, running migration..."
    cd "$PROJECT_ROOT"
    uv run python src/viewer/migrate_db.py
    cd src/viewer
    echo ""
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8200 | xargs kill -9 2>/dev/null || true
lsof -ti:3200 | xargs kill -9 2>/dev/null || true

echo ""
echo "================================================"
echo "🎉 Trace Viewer is starting!"
echo ""
echo "📡 Reflex Backend + Frontend: http://localhost:3200"
echo "🖥️  API will be available at: http://localhost:8200"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Start Reflex app
echo "🚀 Starting Reflex app on ports 3200 (frontend) and 8200 (backend)..."
cd "$PROJECT_ROOT/src/viewer/frontend"

# Initialize Reflex if needed
if [ ! -d ".web" ]; then
    echo "📦 Initializing Reflex app (first time setup)..."
    uv run reflex init --loglevel warning
fi

# Run Reflex (this will block and handle both frontend and backend)
uv run reflex run --loglevel info