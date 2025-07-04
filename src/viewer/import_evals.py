#!/usr/bin/env python3
"""
Import evaluation directories into the new schema database.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import duckdb
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import csv

# Add viewer directory to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config
from db_schema import assert_valid_schema

def parse_evaluation_summary(summary_path: Path) -> Dict[str, Any]:
    """Parse evaluation summary JSON file."""
    with open(summary_path, 'r') as f:
        return json.load(f)

def import_evaluation_directory(eval_dir: Path, con: duckdb.DuckDBPyConnection) -> bool:
    """Import a single evaluation directory."""
    # Check if evaluation summary exists
    summary_path = eval_dir / "evaluation_summary.json"
    if not summary_path.exists():
        print(f"  ❌ No evaluation_summary.json found in {eval_dir}")
        return False
    
    try:
        # Parse evaluation summary
        summary = parse_evaluation_summary(summary_path)
        
        # Extract environment name from path or summary
        env_name = eval_dir.parent.name  # e.g., "crafter" from "evals/crafter/run_xxx"
        
        # Ensure environment exists
        env_result = con.execute("SELECT id FROM environments WHERE name = ?", [env_name]).fetchone()
        if not env_result:
            # Get next ID
            max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM environments").fetchone()[0]
            env_id = max_id + 1
            
            # Create environment
            con.execute("""
                INSERT INTO environments (id, name, display_name, description) 
                VALUES (?, ?, ?, ?)
            """, [env_id, env_name, env_name.title(), f"{env_name} environment"])
            print(f"  Created environment: {env_name} (id={env_id})")
        else:
            env_id = env_result[0]
        
        # Extract evaluation data
        run_id = eval_dir.name  # e.g., "run_20250703_182358"
        timestamp = datetime.strptime(run_id.split('_', 1)[1], "%Y%m%d_%H%M%S")
        
        # Read trajectories from CSV if it exists
        trajectories_csv = eval_dir / "trajectories.csv"
        trajectories = []
        
        if trajectories_csv.exists():
            with open(trajectories_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    traj = {
                        "trajectory_id": row["Trajectory ID"],
                        "model_name": row["Model"],
                        "difficulty": row["Difficulty"],
                        "seed": int(row["Seed"]),
                        "success": row["Success"] == "✓",
                        "num_steps": int(row["Steps"]),
                        "num_turns": int(row["Turns"]),
                        "reward": float(row["Reward"]),
                        "total_achievements": int(row["Total Achievements"]),
                        "achievements": row["Achievements"].split(", ") if row["Achievements"] else [],
                        "termination": row["Termination"]
                    }
                    trajectories.append(traj)
        
        # Get models and difficulties from trajectories or summary
        if trajectories:
            models = list(set(t["model_name"] for t in trajectories))
            difficulties = list(set(t["difficulty"] for t in trajectories))
            num_trajectories = len(trajectories)
            successes = sum(1 for t in trajectories if t["success"])
            success_rate = successes / num_trajectories if num_trajectories > 0 else 0.0
            avg_achievements = sum(t["total_achievements"] for t in trajectories) / num_trajectories if trajectories else 0.0
        else:
            # Fallback to summary data
            models = summary.get("models_evaluated", ["unknown"])
            difficulties = summary.get("difficulties_evaluated", ["unknown"])
            num_trajectories = summary.get("evaluation_metadata", {}).get("num_trajectories", 0)
            success_rate = 0.0
            avg_achievements = 0.0
        
        # Check if evaluation already exists
        existing_eval = con.execute("SELECT id FROM evaluations WHERE run_id = ?", [run_id]).fetchone()
        if existing_eval:
            print(f"  ⚠️  Evaluation {run_id} already exists, skipping...")
            return True
        
        # Get next evaluation ID
        max_eval_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM evaluations").fetchone()[0]
        eval_id = max_eval_id + 1
        
        # Insert evaluation
        con.execute("""
            INSERT INTO evaluations (
                id, env_id, run_id, timestamp, models_evaluated, 
                difficulties_evaluated, num_trajectories, 
                success_rate, avg_achievements, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            eval_id, env_id, run_id, timestamp, models, difficulties,
            num_trajectories, success_rate, avg_achievements,
            json.dumps(summary.get("metadata", {}))
        ])
        
        # Import trajectories
        traces_dir = eval_dir / "traces"
        if traces_dir.exists():
            for idx, traj in enumerate(trajectories):
                # Extract trajectory data
                trace_id = traj["trajectory_id"]
                model_name = traj["model_name"]
                difficulty = traj["difficulty"]
                seed = traj["seed"]
                success = traj["success"]
                final_reward = traj["reward"]
                num_steps = traj["num_steps"]
                achievements = traj["achievements"]
                
                # Get next trajectory ID
                max_traj_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM trajectories").fetchone()[0]
                traj_id = max_traj_id + 1
                
                # Insert trajectory
                con.execute("""
                    INSERT INTO trajectories (
                        id, eval_id, trace_id, model_name, difficulty,
                        seed, success, final_reward, num_steps,
                        achievements, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    traj_id, eval_id, trace_id, model_name, difficulty,
                    seed, success, final_reward, num_steps,
                    achievements, json.dumps(traj.get("metadata", {}))
                ])
                
                # Check if trace file exists
                trace_file = traces_dir / f"{trace_id}.json"
                if trace_file.exists():
                    # Get next trace ID
                    max_trace_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM traces").fetchone()[0]
                    trace_id_num = max_trace_id + 1
                    
                    # Insert trace reference
                    con.execute("""
                        INSERT INTO traces (
                            id, trajectory_id, parquet_path, trace_format, size_bytes
                        ) VALUES (?, ?, ?, ?, ?)
                    """, [
                        trace_id_num,
                        traj_id, 
                        str(trace_file.relative_to(Path.cwd())),
                        "json",  # Still in JSON format
                        trace_file.stat().st_size
                    ])
        
        print(f"  ✅ Imported {run_id}: {num_trajectories} trajectories, {success_rate:.1%} success rate")
        return True
        
    except Exception as e:
        print(f"  ❌ Error importing {eval_dir}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Import evaluation directories into database")
    parser.add_argument("--env", type=str, help="Environment name (e.g., crafter, minigrid)")
    parser.add_argument("--dir", type=str, help="Specific evaluation directory to import")
    parser.add_argument("--all", action="store_true", help="Import all evaluations")
    args = parser.parse_args()
    
    # Validate schema
    try:
        assert_valid_schema()
        print("✅ Database schema is valid\n")
    except AssertionError as e:
        print(f"❌ Schema validation failed: {e}")
        return 1
    
    # Connect to database
    con = duckdb.connect(db_config.db_path, read_only=False)
    
    try:
        # Determine what to import
        evals_base = Path("/Users/joshuapurtell/Documents/GitHub/Environments/src/evals")
        
        if args.dir:
            # Import specific directory
            eval_dir = Path(args.dir)
            if eval_dir.exists():
                print(f"Importing {eval_dir}...")
                import_evaluation_directory(eval_dir, con)
            else:
                print(f"❌ Directory not found: {eval_dir}")
        
        elif args.env:
            # Import all evaluations for specific environment
            env_dir = evals_base / args.env
            if env_dir.exists():
                run_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
                print(f"Found {len(run_dirs)} evaluation runs for {args.env}\n")
                
                for run_dir in tqdm(run_dirs, desc=f"Importing {args.env} evaluations"):
                    import_evaluation_directory(run_dir, con)
            else:
                print(f"❌ Environment directory not found: {env_dir}")
        
        elif args.all:
            # Import all evaluations
            for env_dir in evals_base.iterdir():
                if env_dir.is_dir():
                    run_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
                    if run_dirs:
                        print(f"\nImporting {len(run_dirs)} runs for {env_dir.name}...")
                        for run_dir in tqdm(run_dirs, desc=f"Importing {env_dir.name}"):
                            import_evaluation_directory(run_dir, con)
        
        else:
            print("Please specify --env, --dir, or --all")
            return 1
        
        # Commit changes
        con.commit()
        
        # Show summary
        print("\n=== Import Summary ===")
        env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
        eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        traj_count = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        trace_count = con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        
        print(f"Environments: {env_count}")
        print(f"Evaluations: {eval_count}")
        print(f"Trajectories: {traj_count}")
        print(f"Traces: {trace_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        con.rollback()
        return 1
    finally:
        con.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())