#!/usr/bin/env python3
"""
Launch the DuckDB-backed viewer with minimal implementation.
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

# First import some Crafter data
from src.synth_env.db_minimal import SynthEvalDB
import json

print("🔄 Importing Crafter evaluations to DuckDB...")

db = SynthEvalDB("synth_eval_viewer.duckdb")

# Import first 3 Crafter evaluations
eval_dir = Path("src/evals/crafter")
runs = list(eval_dir.glob("run_*"))[:3]

imported_count = 0
for run_dir in runs:
    print(f"  Importing {run_dir.name}...")
    
    # Load evaluation summary
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.exists():
        continue
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    eval_meta = summary.get("evaluation_metadata", {})
    
    # Insert evaluation
    db.insert_evaluation(
        env_name="crafter",
        run_id=run_dir.name,
        timestamp=eval_meta.get("timestamp", ""),
        models=summary.get("models_evaluated", []),
        num_trajectories=eval_meta.get("num_trajectories", 0),
        success_rate=summary.get("aggregate_results", {}).get("success_rate", 0.0),
        metadata=eval_meta
    )
    
    # Import first few traces
    traces_dir = run_dir / "traces"
    if traces_dir.exists():
        trace_files = list(traces_dir.glob("*.json"))[:5]  # Just first 5 per run
        
        for trace_file in trace_files:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            trace = trace_data.get("trace", {})
            dataset = trace_data.get("dataset", {})
            trace_meta = trace.get("metadata", {})
            
            # Get final reward
            reward_signals = dataset.get("reward_signals", [])
            final_reward = reward_signals[0].get("reward", 0.0) if reward_signals else 0.0
            
            # Count steps
            num_steps = len(trace.get("partition", []))
            
            # Store full trace JSON for viewer
            db.insert_trajectory(
                env_name="crafter",
                run_id=run_dir.name,
                trace_id=trace_file.stem,
                model_name=trace_meta.get("model_name", "unknown"),
                final_reward=final_reward,
                num_steps=num_steps,
                trace_json=json.dumps(trace_data)  # Store full trace
            )
    
    imported_count += 1

print(f"✅ Imported {imported_count} evaluations")

# Now create a minimal DuckDB server
from src.synth_env.viewer.duckdb_minimal_server import run_minimal_duckdb_server

print("\n🚀 Starting DuckDB-backed viewer on port 8996...")
print("\n📌 Navigate to http://localhost:8996")
print("   You should see evaluations loaded from DuckDB")
print("\n   Press Ctrl+C to stop\n")

asyncio.run(run_minimal_duckdb_server(
    port=8996,
    db_path="synth_eval_viewer.duckdb"
))