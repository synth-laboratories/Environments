#!/usr/bin/env python3
"""Simple test of import functionality."""

from pathlib import Path
import json
from datetime import datetime
from src.synth_env.db_v2 import SynthEvalDB

# Create test database
db = SynthEvalDB("test_simple_v2.duckdb")

# Find one Crafter evaluation
eval_dir = Path("src/evals/crafter")
runs = list(eval_dir.glob("run_*"))[:1]  # Get first run

if not runs:
    print("No runs found")
    exit(1)

run_dir = runs[0]
print(f"Testing with: {run_dir}")

# Load evaluation summary
summary_path = run_dir / "evaluation_summary.json"
with open(summary_path, 'r') as f:
    summary = json.load(f)

# Extract data
eval_meta = summary.get("evaluation_metadata", {})
timestamp_str = eval_meta.get("timestamp", "")
try:
    timestamp = datetime.fromisoformat(timestamp_str)
except:
    timestamp = datetime.now()

# Insert evaluation
print("Inserting evaluation...")
eval_id = db.insert_evaluation(
    env_name="crafter",
    run_id=run_dir.name,
    timestamp=timestamp,
    models=summary.get("models_evaluated", []),
    difficulties=summary.get("difficulties_evaluated", []),
    num_trajectories=eval_meta.get("num_trajectories", 0),
    success_rate=summary.get("aggregate_results", {}).get("success_rate", 0.0),
    avg_achievements=summary.get("aggregate_results", {}).get("avg_achievements", 0.0),
    metadata=eval_meta
)

print(f"Inserted evaluation with ID: {eval_id}")

# Check what's in the database
with db.connection() as con:
    envs = con.execute("SELECT * FROM environments").fetchall()
    print(f"\nEnvironments: {envs}")
    
    evals = con.execute("SELECT * FROM evaluations").fetchall()  
    print(f"\nEvaluations: {len(evals)}")
    if evals:
        print(f"First eval: {evals[0]}")

print("\n✅ Test passed!")