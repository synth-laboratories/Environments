from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from uuid import UUID, uuid4

from src.tasks.core import TaskInstance, TaskInstanceMetadata, Impetus, Intent, Task


minigrid_task = Task(
    global_premises="MiniGrid navigation tasks",
    global_constraints="",
    global_objectives="Reach the goal square",
    shared_env_params={},
)


@dataclass
class MiniGridTaskInstanceMetadata(TaskInstanceMetadata):
    env_id: str


@dataclass
class MiniGridTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        data = asdict(self)
        if isinstance(data.get("id"), UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "MiniGridTaskInstance":
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except Exception:
                pass
        if "impetus" in data:
            data["impetus"] = Impetus(**data["impetus"])
        if "intent" in data:
            intent_data = data["intent"]
            intent_data["deterministic_eval_functions"] = []
            data["intent"] = Intent(**intent_data)
        if "metadata" in data:
            data["metadata"] = MiniGridTaskInstanceMetadata(**data["metadata"])
        keep = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in keep})


# Simple single-instance taskset
DEFAULT_TASK_INSTANCE = MiniGridTaskInstance(
    id=uuid4(),
    impetus=Impetus(instructions="Navigate to the goal square."),
    intent=Intent(rubric="Reach the goal", gold_trajectories=None, gold_state_diff={}),
    metadata=MiniGridTaskInstanceMetadata(env_id="MiniGrid-Empty-8x8-v0"),
    is_reproducible=True,
    initial_engine_snapshot=None,
)
