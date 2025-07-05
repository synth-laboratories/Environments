"""
AlgoTune task instance definitions for the synth-env framework.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from uuid import UUID
import uuid

from synth_env.tasks.core import (
    TaskInstance,
    TaskInstanceMetadata,
    Impetus,
    Intent,
    Task,
)


@dataclass
class AlgoTuneTaskInstanceMetadata(TaskInstanceMetadata):
    """Metadata specific to AlgoTune tasks."""

    task_name: str
    problem_size: int
    random_seed: int
    target_speedup: float = 1.5  # Target speedup over baseline


@dataclass
class AlgoTuneTaskInstance(TaskInstance):
    """Task instance for AlgoTune optimization challenges."""

    async def serialize(self) -> Dict[str, Any]:
        """Serialize the task instance to a dictionary."""
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            if "deterministic_eval_functions" in data["intent"]:
                data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "AlgoTuneTaskInstance":
        """Deserialize a task instance from a dictionary."""
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except (ValueError, TypeError, AttributeError):
                # If not a valid UUID, generate a new one
                data["id"] = uuid.uuid4()

        # Reconstruct the nested dataclasses
        if isinstance(data.get("metadata"), dict):
            data["metadata"] = AlgoTuneTaskInstanceMetadata(**data["metadata"])
        if isinstance(data.get("impetus"), dict):
            data["impetus"] = Impetus(**data["impetus"])
        if isinstance(data.get("intent"), dict):
            intent_data = data["intent"].copy()
            intent_data.pop("deterministic_eval_functions", None)
            data["intent"] = Intent(**intent_data)

        return cls(**data)


def create_algotune_task_instance(
    task_name: str,
    problem_size: int = 64,
    random_seed: int = 42,
    target_speedup: float = 1.5,
    task_id: Optional[UUID] = None,
) -> AlgoTuneTaskInstance:
    """
    Create an AlgoTune task instance.

    Args:
        task_name: Name of the AlgoTune task (e.g., "matrix_multiplication")
        problem_size: Size parameter for problem generation
        random_seed: Random seed for reproducibility
        target_speedup: Target speedup over baseline
        task_id: Optional UUID for the task

    Returns:
        AlgoTuneTaskInstance ready for use with AlgoTuneEnvironment
    """
    if task_id is None:
        task_id = uuid.uuid4()

    # Create task metadata
    metadata = AlgoTuneTaskInstanceMetadata(
        task_name=task_name,
        problem_size=problem_size,
        random_seed=random_seed,
        target_speedup=target_speedup,
    )

    # Create impetus with instructions
    impetus = Impetus(
        instructions=f"Optimize the {task_name} algorithm to achieve at least {target_speedup}x speedup "
        f"over the baseline implementation. Your solution must pass all correctness tests."
    )

    # Create intent (simplified for now)
    intent = Intent(
        rubric={"speedup": target_speedup, "correctness": True},
        gold_trajectories=None,
        gold_state_diff={},
        deterministic_eval_functions=[],
    )

    return AlgoTuneTaskInstance(
        id=task_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )


# Pre-defined task instances for common AlgoTune challenges
ALGOTUNE_TASK_PRESETS = {
    "matrix_mult_small": {
        "task_name": "matrix_multiplication",
        "problem_size": 32,
        "target_speedup": 1.2,
    },
    "matrix_mult_large": {
        "task_name": "matrix_multiplication",
        "problem_size": 256,
        "target_speedup": 2.0,
    },
    "qr_factorization": {
        "task_name": "qr_factorization",
        "problem_size": 128,
        "target_speedup": 1.5,
    },
    "convex_hull": {
        "task_name": "convex_hull",
        "problem_size": 1000,
        "target_speedup": 5.0,
    },
    "pagerank": {"task_name": "pagerank", "problem_size": 100, "target_speedup": 2.0},
    "svd_small": {"task_name": "svd", "problem_size": 64, "target_speedup": 1.3},
}
