from __future__ import annotations

from dataclasses import dataclass, asdict
from uuid import UUID, uuid4
from typing import List

from src.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
)

# Basic task description
TASK = Task(
    global_premises="Interact with a bsuite environment.",
    global_constraints="Follow the bsuite action space.",
    global_objectives="Maximise cumulative reward.",
    shared_env_params={},
)


@dataclass
class BSuiteTaskInstance(TaskInstance):
    bsuite_id: str

    async def serialize(self) -> dict:
        d = asdict(self)
        d["id"] = str(self.id)
        return d

    @classmethod
    async def deserialize(cls, data: dict) -> "BSuiteTaskInstance":
        data = dict(data)
        data["id"] = UUID(data["id"])
        return cls(**data)


def default_bsuite_ids() -> List[str]:
    return ["catch/0", "cartpole/0"]


async def create_bsuite_taskset() -> TaskInstanceSet:
    instances: List[BSuiteTaskInstance] = []
    for bid in default_bsuite_ids():
        inst = BSuiteTaskInstance(
            id=uuid4(),
            bsuite_id=bid,
            impetus=Impetus(instructions=f"Solve bsuite task {bid}"),
            intent=Intent(rubric="Return maximisation", gold_trajectories=None, gold_state_diff={}),
            metadata=TaskInstanceMetadata(),
            is_reproducible=False,
            initial_engine_snapshot=None,
        )
        instances.append(inst)

    split_info = SplitInfo(val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=False)

    return TaskInstanceSet(
        name="bsuite_text",
        description="Minimal bsuite text task set",
        instances=instances,
        split_info=split_info,
    )
