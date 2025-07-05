"""MiniGrid TaskSet implementation.

This module provides task generation and management for MiniGrid environments,
including procedural generation and task categorization.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from synth_env.tasks.api import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
)


@dataclass
class MiniGridTaskInstanceMetadata(TaskInstanceMetadata):
    """Metadata for a MiniGrid task instance."""

    env_name: str
    grid_size: Tuple[int, int]
    difficulty: str  # "easy", "medium", "hard"
    has_key: bool = False
    has_door: bool = False
    has_lava: bool = False
    num_objects: int = 0
    optimal_path_length: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class MiniGridTaskInstance(TaskInstance):
    """A specific MiniGrid task instance."""

    async def serialize(self) -> dict:
        """Serialize the task instance to dict."""
        return {
            "id": str(self.id),
            "impetus": {
                "instructions": self.impetus.instructions,
            },
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": self.intent.gold_trajectories,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {
                "env_name": self.metadata.env_name,
                "grid_size": list(self.metadata.grid_size),
                "difficulty": self.metadata.difficulty,
                "has_key": self.metadata.has_key,
                "has_door": self.metadata.has_door,
                "has_lava": self.metadata.has_lava,
                "num_objects": self.metadata.num_objects,
                "optimal_path_length": self.metadata.optimal_path_length,
                "seed": self.metadata.seed,
            },
            "is_reproducible": self.is_reproducible,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> "MiniGridTaskInstance":
        """Deserialize a task instance from dict."""
        return cls(
            id=uuid4() if "id" not in data else data["id"],
            impetus=Impetus(
                instructions=data["impetus"]["instructions"],
            ),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=data["intent"].get("gold_trajectories"),
                gold_state_diff=data["intent"].get("gold_state_diff", {}),
            ),
            metadata=MiniGridTaskInstanceMetadata(
                env_name=data["metadata"]["env_name"],
                grid_size=tuple(data["metadata"]["grid_size"]),
                difficulty=data["metadata"]["difficulty"],
                has_key=data["metadata"].get("has_key", False),
                has_door=data["metadata"].get("has_door", False),
                has_lava=data["metadata"].get("has_lava", False),
                num_objects=data["metadata"].get("num_objects", 0),
                optimal_path_length=data["metadata"].get("optimal_path_length"),
                seed=data["metadata"].get("seed"),
            ),
            is_reproducible=data.get("is_reproducible", True),
        )


# Predefined environment configurations
ENVIRONMENTS = {
    "easy": [
        ("MiniGrid-Empty-5x5-v0", (5, 5)),
        ("MiniGrid-Empty-6x6-v0", (6, 6)),
        ("MiniGrid-Empty-8x8-v0", (8, 8)),
        ("MiniGrid-FourRooms-v0", (19, 19)),
    ],
    "medium": [
        ("MiniGrid-DoorKey-5x5-v0", (5, 5)),
        ("MiniGrid-DoorKey-6x6-v0", (6, 6)),
        ("MiniGrid-DoorKey-8x8-v0", (8, 8)),
        ("MiniGrid-Unlock-v0", (8, 8)),
        ("MiniGrid-UnlockPickup-v0", (8, 8)),
    ],
    "hard": [
        ("MiniGrid-DoorKey-16x16-v0", (16, 16)),
        ("MiniGrid-MultiRoom-N2-S4-v0", (15, 15)),
        ("MiniGrid-MultiRoom-N4-S5-v0", (19, 19)),
        ("MiniGrid-MultiRoom-N6-v0", (25, 25)),
        ("MiniGrid-LavaGapS5-v0", (5, 7)),
        ("MiniGrid-LavaGapS6-v0", (6, 8)),
        ("MiniGrid-LavaGapS7-v0", (7, 9)),
        ("MiniGrid-LavaCrossingS9N1-v0", (9, 9)),
        ("MiniGrid-LavaCrossingS9N2-v0", (9, 9)),
        ("MiniGrid-LavaCrossingS9N3-v0", (9, 9)),
    ],
}


async def create_minigrid_taskset(
    num_tasks_per_difficulty: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
) -> TaskInstanceSet:
    """Generate MiniGrid task instances.

    Args:
        num_tasks_per_difficulty: Number of tasks to generate for each difficulty.
            Defaults to {"easy": 10, "medium": 10, "hard": 10}
        seed: Random seed for reproducibility

    Returns:
        TaskInstanceSet with train/val/test splits
    """
    if num_tasks_per_difficulty is None:
        num_tasks_per_difficulty = {"easy": 10, "medium": 10, "hard": 10}

    if seed is not None:
        random.seed(seed)

    instances = []

    for difficulty, num_tasks in num_tasks_per_difficulty.items():
        if difficulty not in ENVIRONMENTS:
            continue

        envs = ENVIRONMENTS[difficulty]

        for i in range(num_tasks):
            # Select random environment
            env_name, grid_size = random.choice(envs)

            # Determine features based on environment name
            has_key = "DoorKey" in env_name or "Unlock" in env_name
            has_door = "Door" in env_name or "Room" in env_name
            has_lava = "Lava" in env_name

            # Estimate number of objects
            num_objects = 0
            if has_key:
                num_objects += 1
            if has_door:
                num_objects += 1
            if "Pickup" in env_name:
                num_objects += 1

            # Create task-specific instructions with clear symbol explanations
            instructions = f"Navigate the {grid_size[0]}x{grid_size[1]} grid to reach the goal marked with 'G'."

            # Add specific instructions based on environment features
            if has_lava:
                instructions += " Avoid stepping on lava tiles marked with 'L' as they will end your mission."
            if has_key and has_door:
                instructions += " You must first pick up the key marked with 'K', then use it to unlock doors marked with 'D'."
            elif has_door:
                instructions += (
                    " Navigate through doors marked with 'D' to reach different rooms."
                )

            # Add general navigation help
            if env_name.startswith("MiniGrid-Empty"):
                instructions += " The grid contains walls (#) that block movement and empty spaces (.) you can move through."
            elif "FourRooms" in env_name:
                instructions += " The grid is divided into four rooms connected by openings. Find the path between rooms to reach the goal."
            elif "MultiRoom" in env_name:
                instructions += " Navigate through multiple connected rooms to find and reach the goal."

            # Always remind about the goal and exploration
            instructions += " Note: You have limited vision and may need to explore the maze to find the goal. Look for the green goal square marked with 'G' - it may not be visible initially, so explore systematically to discover it."

            # Create rubric
            rubric = {
                "goal": f"Successfully complete the {env_name} environment by reaching the goal.",
                "success_criteria": [
                    "Reach the goal tile marked with 'G'",
                    "Avoid illegal moves or getting stuck",
                ],
            }

            if has_lava:
                rubric["success_criteria"].append("Do not step on lava tiles")
            if has_key:
                rubric["success_criteria"].append(
                    "Pick up the key before attempting to open doors"
                )

            # Generate unique seed for this task
            task_seed = random.randint(0, 1000000)

            # Create task instance
            impetus = Impetus(instructions=instructions)
            intent = Intent(
                rubric=rubric,
                gold_trajectories=None,
                gold_state_diff={},
            )

            metadata = MiniGridTaskInstanceMetadata(
                env_name=env_name,
                grid_size=grid_size,
                difficulty=difficulty,
                has_key=has_key,
                has_door=has_door,
                has_lava=has_lava,
                num_objects=num_objects,
                seed=task_seed,
            )

            instance = MiniGridTaskInstance(
                id=uuid4(),
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )

            instances.append(instance)

    # Create splits (70% train, 15% val, 15% test)
    n_total = len(instances)
    n_val = max(1, int(n_total * 0.15))
    n_test = max(1, int(n_total * 0.15))

    # Shuffle and split
    random.shuffle(instances)

    val_ids = {inst.id for inst in instances[:n_val]}
    test_ids = {inst.id for inst in instances[n_val : n_val + n_test]}

    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="MiniGrid TaskSet",
        description="Diverse MiniGrid navigation tasks across multiple environments and difficulties",
        instances=instances,
        split_info=split_info,
    )


# Default task instance for quick testing
DEFAULT_MINIGRID_TASK = MiniGridTaskInstance(
    id=uuid4(),
    impetus=Impetus(
        instructions="Navigate the 5x5 grid to reach the goal marked with 'G'. The grid contains walls (#) that block movement and empty spaces (.) you can move through. You have limited vision - the goal may not be visible initially. Explore the maze systematically to find the green goal square marked with 'G', then navigate there to complete the task.",
    ),
    intent=Intent(
        rubric={
            "goal": "Successfully reach the goal tile in the MiniGrid-Empty-5x5-v0 environment.",
            "success_criteria": [
                "Reach the goal tile marked with 'G'",
                "Complete the task efficiently",
            ],
        },
        gold_trajectories=None,
        gold_state_diff={},
    ),
    metadata=MiniGridTaskInstanceMetadata(
        env_name="MiniGrid-Empty-5x5-v0",
        grid_size=(5, 5),
        difficulty="easy",
        seed=42,
    ),
    is_reproducible=True,
    initial_engine_snapshot=None,
)


# Module-level export for compatibility
taskset = create_minigrid_taskset
