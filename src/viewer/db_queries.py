"""
Database queries using SQLAlchemy models.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select

from db_models import Environment, Evaluation, Trajectory, Trace, get_session


def get_environments() -> List[str]:
    """Get list of all environment names."""
    session = get_session()
    try:
        environments = session.query(Environment.name).order_by(Environment.name).all()
        return [env[0] for env in environments]
    finally:
        session.close()


def list_evaluations(env: Optional[str] = None) -> pd.DataFrame:
    """List all evaluations, optionally filtered by environment."""
    session = get_session()
    try:
        query = session.query(
            Evaluation.id,
            Evaluation.run_id,
            Evaluation.timestamp,
            Evaluation.models_evaluated,
            Evaluation.difficulties_evaluated,
            Evaluation.num_trajectories,
            Evaluation.success_rate,
            Evaluation.avg_achievements,
            Environment.name.label("env_name"),
            Environment.display_name.label("env_display_name"),
        ).join(Environment)

        if env:
            query = query.filter(Environment.name == env)

        query = query.order_by(Evaluation.timestamp.desc())

        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()

        # Convert results to dictionary format
        data = []
        for row in results:
            data.append(
                {
                    "id": row.id,
                    "run_id": row.run_id,
                    "timestamp": row.timestamp,
                    "models_evaluated": row.models_evaluated,
                    "difficulties_evaluated": row.difficulties_evaluated,
                    "num_trajectories": row.num_trajectories,
                    "success_rate": row.success_rate,
                    "avg_achievements": row.avg_achievements,
                    "env_name": row.env_name,
                    "env_display_name": row.env_display_name,
                }
            )

        return pd.DataFrame(data)
    finally:
        session.close()


def list_traces(run_id: str) -> pd.DataFrame:
    """List all traces for a given run."""
    session = get_session()
    try:
        # Get evaluation by run_id
        evaluation = session.query(Evaluation).filter_by(run_id=run_id).first()
        if not evaluation:
            return pd.DataFrame()

        # Get trajectories for this evaluation
        trajectories = session.query(Trajectory).filter_by(eval_id=evaluation.id).all()

        data = []
        for traj in trajectories:
            data.append(
                {
                    "trace_id": traj.trace_id,
                    "trajectory_id": traj.id,
                    "model_name": traj.model_name,
                    "total_reward": traj.final_reward,
                    "num_steps": traj.num_steps,
                    "success": traj.success,
                    "trace_identifier": traj.trace_id,
                }
            )

        return pd.DataFrame(data)
    finally:
        session.close()


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """Get a single trace by ID."""
    session = get_session()
    try:
        # Find trajectory by trace_id
        trajectory = session.query(Trajectory).filter_by(trace_id=trace_id).first()
        if not trajectory:
            return None

        # Get associated trace if exists
        trace = session.query(Trace).filter_by(trajectory_id=trajectory.id).first()

        result = {
            "id": trajectory.id,
            "trajectory_id": trajectory.id,
            "trace_id": trajectory.trace_id,
            "model_name": trajectory.model_name,
            "difficulty": trajectory.difficulty,
            "seed": trajectory.seed,
            "success": trajectory.success,
            "final_reward": trajectory.final_reward,
            "num_steps": trajectory.num_steps,
            "achievements": trajectory.achievements or [],
            "trajectory_trace_id": trajectory.trace_id,
        }

        if trace:
            result.update(
                {
                    "parquet_path": trace.parquet_path,
                    "trace_format": trace.trace_format,
                    "size_bytes": trace.size_bytes,
                }
            )
            # Add mock trace data for now
            result["trace"] = {
                "steps": [],
                "metadata": {
                    "trace_id": trace_id,
                    "model": trajectory.model_name,
                    "total_steps": trajectory.num_steps,
                },
            }

        return result
    finally:
        session.close()


def get_trace_from_file(
    env_name: str, run_id: str, trace_id: str
) -> Optional[Dict[str, Any]]:
    """Fallback to load trace from filesystem if not in DB."""
    from pathlib import Path
    import json

    # Get absolute path to project root
    project_root = Path(__file__).resolve().parent.parent.parent

    # Look for traces in evals directory
    trace_path = (
        project_root / "evals" / env_name / run_id / "traces" / f"{trace_id}.json"
    )

    if trace_path.exists():
        with open(trace_path, "r") as f:
            return json.load(f)
    return None
