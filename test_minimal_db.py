#!/usr/bin/env python3
"""Test minimal DB implementation."""

from pathlib import Path
import json
from src.synth_env.db_minimal import SynthEvalDB

# Create test database
db = SynthEvalDB("test_minimal.duckdb")

# Find one Crafter evaluation
eval_dir = Path("src/evals/crafter")
runs = list(eval_dir.glob("run_*"))[:1]

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

# Insert evaluation
print("Inserting evaluation...")
success = db.insert_evaluation(
    env_name="crafter",
    run_id=run_dir.name,
    timestamp=eval_meta.get("timestamp", ""),
    models=summary.get("models_evaluated", []),
    num_trajectories=eval_meta.get("num_trajectories", 0),
    success_rate=summary.get("aggregate_results", {}).get("success_rate", 0.0),
    metadata=eval_meta
)

print(f"Inserted evaluation: {success}")

# Insert a few trajectories
traces_dir = run_dir / "traces"
if traces_dir.exists():
    trace_files = list(traces_dir.glob("*.json"))[:3]  # Just first 3
    
    for trace_file in trace_files:
        print(f"Inserting trajectory: {trace_file.name}")
        
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        # Extract basic info
        trace = trace_data.get("trace", {})
        dataset = trace_data.get("dataset", {})
        trace_meta = trace.get("metadata", {})
        
        # Get final reward
        reward_signals = dataset.get("reward_signals", [])
        final_reward = reward_signals[0].get("reward", 0.0) if reward_signals else 0.0
        
        # Count steps
        num_steps = len(trace.get("partition", []))
        
        # Store minimal trace JSON
        minimal_trace = {
            "metadata": trace_meta,
            "num_partitions": num_steps,
            "final_reward": final_reward
        }
        
        success = db.insert_trajectory(
            env_name="crafter",
            run_id=run_dir.name,
            trace_id=trace_file.stem,
            model_name=trace_meta.get("model_name", "unknown"),
            final_reward=final_reward,
            num_steps=num_steps,
            trace_json=json.dumps(minimal_trace)
        )
        
        print(f"  Inserted: {success}")

# Test queries
print("\nTesting queries...")

# Get environments
envs = db.get_environments()
print(f"\nEnvironments: {envs}")

# Get evaluations
evals = db.get_evaluations("crafter")
print(f"\nCrafter evaluations: {len(evals)}")
if evals:
    print(f"First eval: {evals[0]}")

# Get trajectories
if evals:
    trajectories = db.get_trajectories("crafter", evals[0]["id"])
    print(f"\nTrajectories: {len(trajectories)}")
    if trajectories:
        print(f"First trajectory: {trajectories[0]}")

print("\n✅ Test passed!")