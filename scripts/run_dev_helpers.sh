#!/bin/bash

# Dev helpers runner script
# This script runs various development helpers to update project metrics

set -e

echo "🚀 Running development helpers..."

# Check if we have the required dependencies
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    exit 1
fi

if ! command -v uvx &> /dev/null; then
    echo "❌ Error: uvx is not available. Please ensure uv is properly installed."
    exit 1
fi

# Install required dependencies if not present
echo "📦 Installing required dependencies..."
uv add --dev requests pytest pytest-cov toml

# Run the README metrics updater
echo "📊 Updating README metrics..."
uv run python dev/update_readme_metrics.py

echo "✅ All dev helpers completed successfully!"
