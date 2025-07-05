from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class Evaluation(BaseModel):
    env_name: str
    run_id: str
    eval_timestamp: str
    models_evaluated: str
    num_trajectories: int
    success_rate: float
    metadata: Optional[str] = None


class TraceMeta(BaseModel):
    trace_id: str
    trajectory_id: str
    model_name: str
    total_reward: float
    num_steps: int
    terminated_reason: Optional[str] = None


class Trace(TraceMeta):
    trace: Dict[str, Any]  # Full trace JSON data
    env_name: Optional[str] = None
    run_id: Optional[str] = None


class TraceEvent(BaseModel):
    event_type: str
    trace_id: str
    run_id: Optional[str] = None
    timestamp: datetime = datetime.now()


class ViewerConfig(BaseModel):
    env_name: str
    viewer_type: str
    custom_config: Optional[Dict[str, Any]] = None
