from __future__ import annotations
from dataclasses import dataclass
import uuid

from src.tasks.core import Task, TaskInstance, Impetus, Intent, TaskInstanceMetadata
from .schema import TreasureHunterTaskInstance, TreasureHunterTaskInstanceMetadata

TASK = Task(
    global_premises="Simple text world where the goal is to find a hidden treasure.",
    global_constraints="You can move north, south, east, west and take the treasure when found.",
    global_objectives="Retrieve the treasure in as few steps as possible.",
    shared_env_params={},
)

@dataclass
class _DefaultMetadata(TreasureHunterTaskInstanceMetadata):
    pass

INSTANCE = TreasureHunterTaskInstance(
    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    impetus=Impetus(instructions="Find the treasure."),
    intent=Intent(rubric="Treasure obtained", gold_trajectories=None, gold_state_diff={"has_treasure": True}),
    metadata=_DefaultMetadata(),
    is_reproducible=True,
    initial_engine_snapshot=None,
)
