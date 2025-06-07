from dataclasses import dataclass
from uuid import uuid4, UUID
from typing import List

from datasets import load_dataset

from src.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
)
from examples.scicode.schema import SciCodeTaskInstance, SciCodeTaskInstanceMetadata


scicode_task = Task(
    global_premises="Answer programming questions from the SciCode dataset",
    global_constraints="",
    global_objectives="Provide the correct code or explanation",
    shared_env_params={},
)


async def create_scicode_taskset(max_instances: int = 100) -> TaskInstanceSet:
    ds = load_dataset("allenai/scicode", split="train")
    instances: List[SciCodeTaskInstance] = []
    for i, ex in enumerate(ds):
        if i >= max_instances:
            break
        inst_id = UUID(str(ex.get("id", uuid4()))) if ex.get("id") else uuid4()
        impetus = Impetus(instructions="Solve the SciCode task.")
        intent = Intent(rubric={}, gold_trajectories=None, gold_state_diff={})
        metadata = SciCodeTaskInstanceMetadata(
            category=ex.get("category"),
            difficulty=ex.get("difficulty"),
            solution=ex.get("solution") or ex.get("answer"),
        )
        instance = SciCodeTaskInstance(
            id=inst_id,
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        instances.append(instance)

    split_info = SplitInfo(
        val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=False
    )
    return TaskInstanceSet(
        name="SciCode TaskSet",
        description="Tasks derived from the AllenAI SciCode dataset",
        instances=instances,
        split_info=split_info,
    )
