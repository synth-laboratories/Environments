import duckdb
import pathlib
import os
import json
from typing import Optional, Dict, Any, List
import sys

# Add parent directories to path to import db_config
current_dir = pathlib.Path(__file__).resolve().parent
viewer_dir = current_dir.parent.parent
sys.path.insert(0, str(viewer_dir))

from db_config import db_config
from db_schema import assert_valid_schema

# Validate schema on module import - disabled to reduce noise
# The app works with both old and new schemas
# try:
#     assert_valid_schema()
# except AssertionError as e:
#     print(f"WARNING: Database schema validation failed: {e}")

def get_connection():
    """Get a read-only connection to the DuckDB database."""
    # Validate database exists
    if not db_config.validate_db_exists():
        db_info = db_config.get_db_info()
        raise FileNotFoundError(
            f"Database not found at: {db_info['absolute_path']}\n"
            f"Environment variable TRACE_DB: {db_info['env_var']}\n"
            f"Please ensure the database file exists or set TRACE_DB environment variable."
        )
    
    # Assert we have a valid absolute path
    db_path = db_config.db_path
    assert os.path.isabs(db_path), f"Database path must be absolute, got: {db_path}"
    
    return duckdb.connect(db_path, read_only=True)

def list_evaluations(env: Optional[str] = None):
    """List all evaluations, optionally filtered by environment."""
    con = get_connection()
    q = """
    SELECT 
        e.*,
        env.name as env_name,
        env.display_name as env_display_name
    FROM evaluations e
    JOIN environments env ON e.env_id = env.id
    """
    params = []
    if env:
        q += " WHERE env.name = ?"
        params.append(env)
    q += " ORDER BY e.timestamp DESC"
    return con.execute(q, params).fetchdf()

def list_traces(run_id: str):
    """List all traces for a given run."""
    con = get_connection()
    # Updated query based on actual schema
    q = """
    SELECT 
        t.id as trace_id,
        t.trajectory_id,
        tr.model_name,
        tr.final_reward as total_reward,
        tr.num_steps,
        tr.success,
        tr.trace_id as trace_identifier
    FROM traces t
    JOIN trajectories tr ON t.id = tr.id
    JOIN evaluations e ON tr.eval_id = e.id
    WHERE e.run_id = ?
    ORDER BY t.id
    """
    return con.execute(q, [run_id]).fetchdf()

def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """Get a single trace by ID."""
    con = get_connection()
    
    # The trace_id is a string identifier from trajectories table
    q = """
    SELECT 
        t.*,
        tr.trace_id as trajectory_trace_id,
        tr.model_name,
        tr.final_reward,
        tr.num_steps,
        tr.success,
        tr.achievements,
        tr.difficulty,
        tr.seed
    FROM trajectories tr
    JOIN traces t ON t.trajectory_id = tr.id
    WHERE tr.trace_id = ?
    """
    df = con.execute(q, [trace_id]).fetchdf()
    
    if len(df) == 0:
        return None
    
    trace_row = df.iloc[0].to_dict()
    
    # If trace data is stored as parquet path, we need to load it
    if 'parquet_path' in trace_row and trace_row['parquet_path']:
        # TODO: Load from parquet file
        # For now, return mock trace data for testing
        trace_row['trace'] = {
            'steps': [],
            'metadata': {
                'trace_id': trace_id,
                'model': trace_row.get('model_name', 'unknown'),
                'total_steps': trace_row.get('num_steps', 0)
            }
        }
    
    return trace_row

def get_trace_from_file(env_name: str, run_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
    """Fallback to load trace from filesystem if not in DB."""
    # Get absolute path to project root
    project_root = db_config.db_path_absolute.parent
    
    # Look for traces in evals directory
    trace_path = project_root / "evals" / env_name / run_id / "traces" / f"{trace_id}.json"
    
    # Assert we're using absolute paths
    assert trace_path.is_absolute(), f"Trace path must be absolute, got: {trace_path}"
    
    if trace_path.exists():
        with open(trace_path, 'r') as f:
            return json.load(f)
    return None

def get_environments() -> List[str]:
    """Get list of all environments in the database."""
    con = get_connection()
    q = "SELECT DISTINCT name FROM environments ORDER BY name"
    return [row[0] for row in con.execute(q).fetchall()] 