from uuid import UUID
import uuid
from typing import List

from synth_env.tasks.core import (
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
    TaskInstanceMetadataFilter,
)
from synth_env.examples.math.engine import load_tasks
from synth_env.examples.math.schema import (
    HendryksTaskInstance,
    HendryksTaskInstanceMetadata,
)


# --- Metadata Filters ---
class HendryksSubjectFilter(TaskInstanceMetadataFilter):
    def __init__(self, subjects: List[str]):
        self.subjects = subjects

    def __call__(self, instance: HendryksTaskInstance) -> bool:
        if hasattr(instance.metadata, "subject"):
            return instance.metadata.subject in self.subjects
        return False


class HendryksLevelFilter(TaskInstanceMetadataFilter):
    def __init__(self, levels: List[str]):
        self.levels = levels

    def __call__(self, instance: HendryksTaskInstance) -> bool:
        if hasattr(instance.metadata, "level"):
            return instance.metadata.level in self.levels
        return False


async def create_hendryks_taskset() -> TaskInstanceSet:
    """Generate a TaskInstanceSet for Hendryks math problems."""
    tasks = load_tasks()
    instances: List[HendryksTaskInstance] = []
    for t in tasks:
        inst_id_str = t.get("id")
        inst_id = None
        try:
            # Attempt to parse as UUID if it's a valid UUID string
            inst_id = UUID(inst_id_str)
        except (ValueError, TypeError, AttributeError):
            # If not a valid UUID string, or if inst_id_str is None, create a new one
            # This behavior might need adjustment based on how IDs are expected downstream
            inst_id = uuid.uuid4()  # Or handle error, or try uuid.UUID(t["id"]) as before if that was intended

        impetus = Impetus(instructions="Solve the math problem.")
        intent = Intent(rubric={}, gold_trajectories=None, gold_state_diff={})

        # Ensure metadata from loaded task 't' is correctly passed
        meta_subject = t.get("metadata", {}).get("subject")
        meta_category = t.get("metadata", {}).get("category")
        meta_level = t.get("metadata", {}).get("level")
        solution = t.get("solution")

        metadata = HendryksTaskInstanceMetadata(
            subject=meta_subject,
            category=meta_category,
            level=meta_level,
            solution=solution,
        )
        instance = HendryksTaskInstance(
            id=inst_id,
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,  # Assuming true for dataset-derived tasks
            initial_engine_snapshot=None,  # Math env is stateless, no initial snapshot needed
        )
        instances.append(instance)

    # No explicit splits defined here, but SplitInfo is required by TaskInstanceSet
    split_info = SplitInfo(
        val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=False
    )

    return TaskInstanceSet(
        name="Hendryks Math TaskSet",
        description="Task set generated from EleutherAI Hendrycks Math dataset",
        instances=instances,
        split_info=split_info,
    )
