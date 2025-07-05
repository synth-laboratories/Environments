import pathlib
import os
import json
from typing import Optional, Dict, Any, List
import sys

# Add parent directory to path to import db_config
current_dir = pathlib.Path(__file__).resolve().parent
viewer_dir = current_dir.parent
sys.path.insert(0, str(viewer_dir))

from db_config import db_config
from db_queries import get_environments, list_evaluations, list_traces, get_trace

# The SQLAlchemy-based functions are already imported above
# Just re-export them for backward compatibility

# Re-exported functions from db_queries:
# - get_environments()
# - list_evaluations(env: Optional[str] = None)
# - list_traces(run_id: str)
# - get_trace(trace_id: str)


def get_trace_from_file(
    env_name: str, run_id: str, trace_id: str
) -> Optional[Dict[str, Any]]:
    """Fallback to load trace from filesystem if not in DB."""
    # Get absolute path to project root
    project_root = pathlib.Path(db_config.db_path).parent

    # Look for traces in evals directory
    trace_path = (
        project_root / "evals" / env_name / run_id / "traces" / f"{trace_id}.json"
    )

    # Assert we're using absolute paths
    assert trace_path.is_absolute(), f"Trace path must be absolute, got: {trace_path}"

    if trace_path.exists():
        with open(trace_path, "r") as f:
            return json.load(f)
    return None
