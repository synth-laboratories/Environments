#!/usr/bin/env python3
"""Test the landing page directly"""

import asyncio
from src.synth_env.viewer.integrated_server import run_integrated_server

print("Starting integrated server with landing page...")
print("Navigate to http://localhost:8999")
print("You should see:")
print("  - A header saying 'Synth Environment Evaluations'")
print("  - Environment cards (Crafter with 12 evaluations)")
print("  - Click on Crafter to see the list of evaluations")
print("\nPress Ctrl+C to stop\n")

asyncio.run(run_integrated_server(port=8999))