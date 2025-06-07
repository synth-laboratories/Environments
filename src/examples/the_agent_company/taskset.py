from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

from src.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    Impetus,
    Intent,
    SplitInfo,
)

TASK_LIST_URL = (
    "https://github.com/TheAgentCompany/"
    "TheAgentCompany/releases/download/1.0.0/tasks.md"
)
LOCAL_TASKS = Path(__file__).parent / "data" / "tasks.md"

tac_task = Task(
    global_premises="Complete real-world digital tasks using provided Docker images.",
    global_constraints="",
    global_objectives="Finish the task successfully.",
    shared_env_params={},
)


@dataclass
class TACMetadata(TaskInstanceMetadata):
    image: str


@dataclass
class TACTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        return {"id": str(self.id), "image": self.metadata.image}

    @classmethod
    async def deserialize(cls, data: dict) -> "TACTaskInstance":
        return cls(
            id=uuid4(),
            impetus=Impetus(instructions=""),
            intent=Intent(rubric="", gold_trajectories=None, gold_state_diff={}),
            metadata=TACMetadata(image=data["image"]),
            is_reproducible=False,
            initial_engine_snapshot=None,
        )


async def create_tac_taskset(max_tasks: int | None = None) -> TaskInstanceSet:
    if LOCAL_TASKS.exists():
        lines = [line.strip() for line in LOCAL_TASKS.read_text().splitlines() if line.strip()]
    else:
        with urllib.request.urlopen(TASK_LIST_URL) as f:
            lines = [line.decode("utf-8").strip() for line in f.readlines() if line.strip()]
        LOCAL_TASKS.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_TASKS.write_text("\n".join(lines))
    if max_tasks:
        lines = lines[:max_tasks]

    instances: List[TACTaskInstance] = []
    for image in lines:
        instance = TACTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions=f"Run task {image}"),
            intent=Intent(
                rubric="Complete the task", gold_trajectories=None, gold_state_diff={}
            ),
            metadata=TACMetadata(image=image),
            is_reproducible=False,
            initial_engine_snapshot=None,
        )
        instances.append(instance)

    split_info = SplitInfo(
        val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=False
    )

    return TaskInstanceSet(
        name="The Agent Company Benchmark",
        description="Tasks from TheAgentCompany benchmark.",
        instances=instances,
        split_info=split_info,
    )
