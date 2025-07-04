#!/usr/bin/env python3
"""
Migrate existing evaluations to DuckDB for the new trace viewer.
"""
import duckdb
import json
from pathlib import Path
from datetime import datetime

def create_schema(con):
    """Create the database schema."""
    
    # Drop existing tables
    con.execute("DROP TABLE IF EXISTS traces")
    con.execute("DROP TABLE IF EXISTS trajectories")
    con.execute("DROP TABLE IF EXISTS evaluations")
    con.execute("DROP TABLE IF EXISTS environments")
    
    # Create tables
    con.execute("""
        CREATE TABLE IF NOT EXISTS environments (
            env_name VARCHAR PRIMARY KEY,
            display_name VARCHAR,
            description VARCHAR
        )
    """)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            env_name VARCHAR,
            run_id VARCHAR,
            eval_timestamp VARCHAR,
            models_evaluated VARCHAR,
            num_trajectories INTEGER,
            success_rate DOUBLE,
            metadata VARCHAR,
            PRIMARY KEY (env_name, run_id)
        )
    """)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS trajectories (
            env_name VARCHAR,
            run_id VARCHAR,
            trajectory_id VARCHAR PRIMARY KEY,
            model_name VARCHAR,
            seed INTEGER,
            total_reward DOUBLE,
            num_steps INTEGER,
            terminated_reason VARCHAR,
            metadata VARCHAR
        )
    """)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id VARCHAR PRIMARY KEY,
            env_name VARCHAR,
            run_id VARCHAR,
            trajectory_id VARCHAR,
            trace_data_json TEXT,
            trace_data_parquet VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    print("✅ Schema created")

def import_evaluations(con, base_path="../evals"):
    """Import all evaluations from filesystem."""
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_path}")
        return
    
    # Add environments
    for env_dir in base_dir.iterdir():
        if env_dir.is_dir():
            env_name = env_dir.name
            con.execute(
                "INSERT OR REPLACE INTO environments (env_name, display_name) VALUES (?, ?)",
                [env_name, env_name.title()]
            )
            print(f"✅ Added environment: {env_name}")
            
            # Process runs
            for run_dir in env_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    import_run(con, env_name, run_dir)

def import_run(con, env_name, run_dir):
    """Import a single evaluation run."""
    run_id = run_dir.name
    
    # Load evaluation summary if available
    summary_file = run_dir / "evaluation_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            
        con.execute("""
            INSERT OR REPLACE INTO evaluations 
            (env_name, run_id, eval_timestamp, models_evaluated, num_trajectories, success_rate, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            env_name,
            run_id,
            summary.get("timestamp", datetime.now().isoformat()),
            ",".join(summary.get("models", [])),
            summary.get("num_trajectories", 0),
            summary.get("success_rate", 0.0),
            json.dumps(summary.get("metadata", {}))
        ])
    else:
        # Create basic entry
        con.execute("""
            INSERT OR REPLACE INTO evaluations 
            (env_name, run_id, eval_timestamp, models_evaluated, num_trajectories, success_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            env_name,
            run_id,
            datetime.now().isoformat(),
            "unknown",
            0,
            0.0
        ])
    
    # Import traces
    traces_dir = run_dir / "traces"
    if traces_dir.exists():
        trace_count = 0
        for trace_file in traces_dir.glob("*.json"):
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                
                trace_id = trace_file.stem
                
                # Extract metadata
                trace_meta = trace_data.get("trace", {}).get("metadata", {})
                dataset = trace_data.get("dataset", {})
                
                # Insert trajectory
                con.execute("""
                    INSERT OR REPLACE INTO trajectories
                    (env_name, run_id, trajectory_id, model_name, total_reward, num_steps, terminated_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    env_name,
                    run_id,
                    trace_id,
                    trace_meta.get("model_name", "unknown"),
                    dataset.get("reward_signals", [{}])[0].get("reward", 0.0),
                    len(trace_data.get("trace", {}).get("partition", [])),
                    trace_meta.get("termination_reason", None)
                ])
                
                # Insert trace reference
                con.execute("""
                    INSERT OR REPLACE INTO traces
                    (trace_id, env_name, run_id, trajectory_id, trace_data_json)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    trace_id,
                    env_name,
                    run_id,
                    trace_id,
                    json.dumps(trace_data)
                ])
                
                trace_count += 1
                
            except Exception as e:
                print(f"⚠️  Failed to import {trace_file}: {e}")
        
        print(f"✅ Imported {run_id}: {trace_count} traces")

def main():
    """Main migration function."""
    db_path = "synth_eval.duckdb"
    
    print(f"🗄️  Migrating to DuckDB: {db_path}")
    
    con = duckdb.connect(db_path)
    
    # Create schema
    create_schema(con)
    
    # Import evaluations
    import_evaluations(con)
    
    # Show summary
    env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
    eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    trace_count = con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    
    print("\n📊 Migration Summary:")
    print(f"  - Environments: {env_count}")
    print(f"  - Evaluations: {eval_count}")
    print(f"  - Traces: {trace_count}")
    
    con.close()
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    main()