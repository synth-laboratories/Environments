#!/bin/bash
# Script to run the Reflex viewer with proper database configuration

# Get the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

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

# Change to frontend directory
cd "$SCRIPT_DIR/frontend" || exit 1

# Run Reflex
echo "Starting Reflex viewer..."
uv run reflex run