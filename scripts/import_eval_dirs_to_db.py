#!/usr/bin/env python3
"""
Import existing evaluation directories into DuckDB.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.synth_env.db import SynthEvalDB


def parse_evaluation_summary(summary_path: Path) -> Dict[str, Any]:
    """Parse evaluation summary JSON file."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def parse_trace_file(trace_path: Path) -> Dict[str, Any]:
    """Parse a trace JSON file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def extract_trace_metadata(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from trace data."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    
    # Get trace metadata
    trace_meta = trace.get("metadata", {})
    
    # Calculate statistics
    num_steps = len(trace.get("partition", []))
    
    # Get final reward from dataset
    reward_signals = dataset.get("reward_signals", [])
    final_reward = reward_signals[0].get("reward", 0.0) if reward_signals else 0.0
    
    # Determine success (you may need to adjust this based on environment)
    success = final_reward > 0.5  # Simple threshold, adjust as needed
    
    # Extract achievements if present
    achievements = []
    for partition in trace.get("partition", []):
        for event in partition.get("events", []):
            env_steps = event.get("environment_compute_steps", [])
            for step in env_steps:
                if step.get("step_type") == "EnvironmentLogStep":
                    logs = step.get("output", {}).get("logs", [])
                    for log in logs:
                        if "achievement" in log.lower():
                            achievements.append(log)
    
    return {
        "model_name": trace_meta.get("model_name", "unknown"),
        "difficulty": trace_meta.get("difficulty"),
        "seed": trace_meta.get("seed"),
        "success": success,
        "final_reward": final_reward,
        "num_steps": num_steps,
        "achievements": list(set(achievements)),  # Unique achievements
        "metadata": trace_meta
    }


def import_evaluation_directory(db: SynthEvalDB, eval_dir: Path, 
                               env_name: Optional[str] = None,
                               use_parquet: bool = True,
                               parquet_base_dir: Path = Path("src/evals_parquet")) -> int:
    """Import a single evaluation directory."""
    # Load evaluation summary
    summary_path = eval_dir / "evaluation_summary.json"
    if not summary_path.exists():
        print(f"  ⚠️  No evaluation_summary.json found in {eval_dir}")
        return 0
    
    summary = parse_evaluation_summary(summary_path)
    
    # Determine environment name
    if not env_name:
        env_name = summary.get("environment", eval_dir.parent.name).lower()
    
    # Extract evaluation metadata
    eval_meta = summary.get("evaluation_metadata", {})
    run_id = eval_dir.name
    
    # Parse timestamp
    timestamp_str = eval_meta.get("timestamp", "")
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except:
        # Fallback to directory modification time
        timestamp = datetime.fromtimestamp(eval_dir.stat().st_mtime)
    
    # Insert evaluation
    eval_id = db.insert_evaluation(
        env_name=env_name,
        run_id=run_id,
        timestamp=timestamp,
        models=summary.get("models_evaluated", []),
        difficulties=summary.get("difficulties_evaluated", []),
        num_trajectories=eval_meta.get("num_trajectories", 0),
        success_rate=summary.get("aggregate_results", {}).get("success_rate", 0.0),
        avg_achievements=summary.get("aggregate_results", {}).get("avg_achievements", 0.0),
        metadata=eval_meta
    )
    
    # Import traces
    traces_dir = eval_dir / "traces"
    if not traces_dir.exists():
        return 0
    
    trace_count = 0
    trace_files = list(traces_dir.glob("*.json"))
    
    for trace_file in tqdm(trace_files, desc=f"  Importing traces", leave=False):
        try:
            # Load trace data
            trace_data = parse_trace_file(trace_file)
            trace_id = trace_file.stem
            
            # Extract metadata
            trace_info = extract_trace_metadata(trace_data)
            
            # Insert trajectory
            trajectory_id = db.insert_trajectory(
                eval_id=eval_id,
                trace_id=trace_id,
                **trace_info
            )
            
            # Save as parquet if requested
            if use_parquet:
                parquet_dir = parquet_base_dir / env_name / run_id
                parquet_path = parquet_dir / f"{trace_id}.parquet"
                
                # Convert to parquet
                actual_path = db.save_trace_as_parquet(trace_data["trace"], parquet_path)
                
                # Insert trace reference
                db.insert_trace_parquet(
                    trajectory_id=trajectory_id,
                    parquet_path=str(actual_path.relative_to(Path.cwd()))
                )
            
            trace_count += 1
            
        except Exception as e:
            print(f"    ❌ Error importing trace {trace_file}: {e}")
    
    return trace_count


def main():
    parser = argparse.ArgumentParser(description="Import evaluation directories to DuckDB")
    parser.add_argument("--eval-dir", type=Path, default=Path("src/evals"),
                       help="Base directory containing evaluations")
    parser.add_argument("--db-path", type=Path, default=Path("synth_eval.duckdb"),
                       help="Path to DuckDB database file")
    parser.add_argument("--parquet-dir", type=Path, default=Path("src/evals_parquet"),
                       help="Directory to store parquet files")
    parser.add_argument("--env", type=str, help="Specific environment to import")
    parser.add_argument("--run", type=str, help="Specific run to import")
    parser.add_argument("--no-parquet", action="store_true",
                       help="Skip parquet conversion")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be imported without doing it")
    parser.add_argument("--limit", type=int, help="Limit number of evaluations to import")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN - No changes will be made")
    
    # Create database
    db = SynthEvalDB(args.db_path)
    print(f"📊 Using database: {args.db_path}")
    
    # Find evaluation directories
    eval_dirs = []
    
    if args.env and args.run:
        # Specific run
        eval_dir = args.eval_dir / args.env / args.run
        if eval_dir.exists():
            eval_dirs.append((args.env, eval_dir))
    elif args.env:
        # All runs for an environment
        env_dir = args.eval_dir / args.env
        if env_dir.exists():
            for run_dir in sorted(env_dir.glob("run_*")):
                eval_dirs.append((args.env, run_dir))
    else:
        # All environments
        for env_dir in sorted(args.eval_dir.iterdir()):
            if env_dir.is_dir():
                for run_dir in sorted(env_dir.glob("run_*")):
                    eval_dirs.append((env_dir.name, run_dir))
    
    # Apply limit if specified
    if args.limit:
        eval_dirs = eval_dirs[:args.limit]
    
    print(f"📁 Found {len(eval_dirs)} evaluation(s) to import")
    
    if args.dry_run:
        for env_name, eval_dir in eval_dirs:
            print(f"  Would import: {env_name}/{eval_dir.name}")
        return
    
    # Import evaluations
    total_traces = 0
    for env_name, eval_dir in tqdm(eval_dirs, desc="Importing evaluations"):
        print(f"\n🔄 Importing {env_name}/{eval_dir.name}")
        try:
            trace_count = import_evaluation_directory(
                db, eval_dir, env_name,
                use_parquet=not args.no_parquet,
                parquet_base_dir=args.parquet_dir
            )
            total_traces += trace_count
            print(f"  ✅ Imported {trace_count} traces")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✨ Import complete!")
    print(f"  📈 Imported {len(eval_dirs)} evaluations with {total_traces} total traces")
    
    # Show summary
    with db.connection() as con:
        env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
        eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        traj_count = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        
        print(f"\n📊 Database summary:")
        print(f"  Environments: {env_count}")
        print(f"  Evaluations:  {eval_count}")
        print(f"  Trajectories: {traj_count}")


if __name__ == "__main__":
    main()