from dataclasses import dataclass
import uuid
from src.tasks.core import TaskInstance, Impetus, Intent, TaskInstanceMetadata

@dataclass
class MiniHackTaskInstance(TaskInstance):
    env_id: str = "MiniHack-Room-5x5-v0"
    max_steps: int = 50


INSTANCE = MiniHackTaskInstance(
    id=uuid.uuid4(),
    impetus=Impetus(instructions="Explore the dungeon and reach the goal."),
    intent=Intent(rubric="Collect the goal", gold_trajectories=None, gold_state_diff={}),
    metadata=TaskInstanceMetadata(),
    is_reproducible=True,
    initial_engine_snapshot=None,
)
