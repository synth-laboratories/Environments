#!/bin/bash
# Script to run the Streamlit viewer with proper database configuration

# Get the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Set the database path environment variable if not already set
if [ -z "$TRACE_DB" ]; then
    export TRACE_DB="$PROJECT_ROOT/synth_eval.duckdb"
    echo "Setting TRACE_DB to: $TRACE_DB"
else
    echo "Using existing TRACE_DB: $TRACE_DB"
fi

# Verify database exists
if [ ! -f "$TRACE_DB" ]; then
    echo "⚠️  Warning: Database file not found at $TRACE_DB"
    echo "Please ensure the database exists or set TRACE_DB to the correct path."
    exit 1
fi

echo "✅ Database found at: $TRACE_DB"

# Change to viewer directory
cd "$PROJECT_ROOT/src/viewer" || exit 1

# Run Streamlit
echo "Starting Streamlit viewer..."
uv run streamlit run streamlit_app.py --server.port 8501 --server.address localhost