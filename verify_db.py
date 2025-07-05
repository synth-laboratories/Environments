#!/usr/bin/env python3
"""Verify what's in the DuckDB database."""

from src.synth_env.db_minimal import SynthEvalDB

db = SynthEvalDB("synth_eval_viewer.duckdb")

print("=== ENVIRONMENTS ===")
envs = db.get_environments()
for env in envs:
    print(f"  {env}")

print("\n=== CRAFTER EVALUATIONS ===")
evals = db.get_evaluations("crafter")
for eval in evals:
    print(f"  {eval['id']}: {eval['num_trajectories']} trajectories")
    
    # Get trajectories for first eval
    if evals:
        trajs = db.get_trajectories("crafter", evals[0]["id"])
        print(f"\n  First eval trajectories: {len(trajs)}")
        for traj in trajs:
            print(f"    - {traj['id']}")