from __future__ import annotations

from dataclasses import dataclass
import uuid

from src.tasks.core import Task, TaskInstance, Impetus, Intent, TaskInstanceMetadata

TASK = Task(
    global_premises="Explore the NetHack dungeon and survive.",
    global_constraints="Standard NetHack rules apply.",
    global_objectives="Reach the first downstairs.",
    shared_env_params={},
)


@dataclass
class NethackTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": None,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {},
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": None,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> "NethackTaskInstance":
        return cls(
            id=uuid.UUID(data["id"]),
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=None,
                gold_state_diff=data["intent"]["gold_state_diff"],
            ),
            metadata=TaskInstanceMetadata(),
            is_reproducible=data["is_reproducible"],
            initial_engine_snapshot=None,
        )


INSTANCE = NethackTaskInstance(
    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    impetus=Impetus(instructions="Descend into the dungeon and find the downstairs."),
    intent=Intent(
        rubric="Find the downstairs on the first level.",
        gold_trajectories=None,
        gold_state_diff={},
    ),
    metadata=TaskInstanceMetadata(),
    is_reproducible=True,
    initial_engine_snapshot=None,
)
