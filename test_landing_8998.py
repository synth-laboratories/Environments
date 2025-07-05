#!/usr/bin/env python3
"""Test the landing page on port 8998"""

import asyncio
from src.synth_env.viewer.integrated_server import run_integrated_server

print("Starting integrated server with landing page on port 8998...")
print("Navigate to http://localhost:8998")
print("You should see:")
print("  - A header saying 'Synth Environment Evaluations'")
print("  - Environment cards (Crafter with 12 evaluations)")
print("  - Click on Crafter to see the list of evaluations")
print("  - Click 'View' on any evaluation to see the new viewer")
print("\nPress Ctrl+C to stop\n")

asyncio.run(run_integrated_server(port=8998))