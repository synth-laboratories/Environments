from typing import List
from uuid import uuid4

from src.tasks.core import (
    Task,
    TaskInstanceSet,
    TaskInstanceMetadataFilter,
    SplitInfo,
    Impetus,
    Intent,
)
from examples.memory_text.schema import (
    MemoryTextTaskInstance,
    MemoryTextTaskInstanceMetadata,
)

# Global description of the task family
TASK = Task(
    global_premises="Memorize sequences of digits and recall them exactly.",
    global_constraints="",
    global_objectives="Correctly repeat the shown sequence.",
    shared_env_params={},
)


class SequenceLengthFilter(TaskInstanceMetadataFilter):
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, instance: MemoryTextTaskInstance) -> bool:
        return getattr(instance.metadata, "sequence_length", None) == self.length


async def create_memory_text_taskset(
    num_instances: int = 20, sequence_length: int = 5
) -> TaskInstanceSet:
    """Generate a simple TaskInstanceSet for TextMemoryGym."""
    instances: List[MemoryTextTaskInstance] = []
    for _ in range(num_instances):
        impetus = Impetus(instructions="Recall the digits in order.")
        intent = Intent(rubric={"goal": "Repeat the sequence"})
        metadata = MemoryTextTaskInstanceMetadata(sequence_length=sequence_length)
        instance = MemoryTextTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        instances.append(instance)

    # Simple 80/10/10 split
    n = len(instances)
    val_ids = {inst.id for inst in instances[int(0.8 * n) : int(0.9 * n)]}
    test_ids = {inst.id for inst in instances[int(0.9 * n) :]}
    split = SplitInfo(
        val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True
    )

    return TaskInstanceSet(
        name="TextMemoryGym TaskSet",
        description="Digit sequence recall tasks.",
        instances=instances,
        split_info=split,
    )
