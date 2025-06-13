from synth_env.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceMetadataFilter,
    TaskInstanceSet,
)
from uuid import uuid4, UUID
from synth_env.tasks.core import SplitInfo, Impetus, Intent
from synth_env.examples.sokoban.engine_helpers.room_utils import (
    generate_room,
    get_shortest_action_path,
)
from dataclasses import dataclass, asdict, fields
from typing import Tuple
import os

sokoban_task = Task(
    global_premises="Procedural Sokoban task generation",
    global_constraints="",
    global_objectives="Push all boxes onto target locations",
    shared_env_params={},
)

# Configuration parameters
NUM_INSTANCES_PER_DIFFICULTY = 10
SEED_START = 42
DIFFICULTY_CONFIGS = {
    "easy": {
        "num_boxes": 1,
        "dim_room": (5, 5),
        "max_steps": 120,
        "impetus_prompt": "Solve this simple Sokoban puzzle by pushing the box onto the target.",
    },
    "medium": {
        "num_boxes": 2,
        "dim_room": (7, 7),
        "max_steps": 200,
        "impetus_prompt": "Solve this Sokoban puzzle by pushing the 2 boxes onto the targets.",
    },
    "hard": {
        "num_boxes": 4,
        "dim_room": (10, 10),
        "max_steps": 300,
        "impetus_prompt": "Solve this challenging Sokoban puzzle by pushing the 4 boxes onto the targets.",
    },
}


@dataclass
class SokobanTaskInstanceMetadata(TaskInstanceMetadata):
    difficulty: str
    num_boxes: int
    dim_room: Tuple[int, int]
    max_steps: int
    shortest_path_length: int
    seed: int
    generation_params: str


@dataclass
class SokobanTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            if "deterministic_eval_functions" in data["intent"]:
                data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "SokobanTaskInstance":
        """Gracefully accept non-UUID ids (e.g. 'demo-mcts')."""
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except (ValueError, TypeError, AttributeError):
                pass  # keep original string

        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])

        if "intent" in data and isinstance(data["intent"], dict):
            intent_data = data["intent"]
            intent_data["deterministic_eval_functions"] = []
            if (
                "gold_trajectories" in intent_data
                and intent_data["gold_trajectories"] is not None
            ):
                pass
            data["intent"] = Intent(**intent_data)

        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = SokobanTaskInstanceMetadata(**data["metadata"])

        constructor_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in constructor_field_names}

        return cls(**filtered_data)


async def create_sokoban_taskset() -> TaskInstanceSet:
    """Generates Sokoban task instances wrapped in a TaskInstanceSet."""
    instances = []
    current_seed = SEED_START

    for difficulty, config in DIFFICULTY_CONFIGS.items():
        for i in range(NUM_INSTANCES_PER_DIFFICULTY):
            instance_id = uuid4()
            room_structure, room_state, _, _ = generate_room(
                dim=config["dim_room"],
                initial_seed=current_seed,
                num_boxes=config["num_boxes"],
                search_depth=config["max_steps"],
            )
            shortest_actions = get_shortest_action_path(
                room_structure, room_state, MAX_DEPTH=config["max_steps"]
            )
            path_length = len(shortest_actions)

            impetus = Impetus(instructions=config["impetus_prompt"])
            intent = Intent(
                rubric={"goal": "Push all boxes onto target locations."},
                gold_trajectories=None,
                gold_state_diff={},
            )
            metadata = SokobanTaskInstanceMetadata(
                difficulty=difficulty,
                num_boxes=config["num_boxes"],
                dim_room=config["dim_room"],
                max_steps=config["max_steps"],
                shortest_path_length=path_length,
                seed=current_seed,
                generation_params=f"dim={config['dim_room']}, boxes={config['num_boxes']}, steps={config['max_steps']}",
            )

            task_instance = SokobanTaskInstance(
                id=instance_id,
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(task_instance)
            current_seed += 1

    class NumBoxesFilter(TaskInstanceMetadataFilter):
        def __init__(self, num_boxes):
            self.num_boxes = num_boxes

        def __call__(self, instance):
            if hasattr(instance.metadata, "num_boxes"):
                return instance.metadata.num_boxes == self.num_boxes
            return False

    class DimRoomFilter(TaskInstanceMetadataFilter):
        def __init__(self, dim_room):
            self.dim_room = dim_room

        def __call__(self, instance):
            if hasattr(instance.metadata, "dim_room"):
                return instance.metadata.dim_room == self.dim_room
            return False

    class PathLengthFilter(TaskInstanceMetadataFilter):
        def __init__(self, min_length=None, max_length=None):
            self.min_length = min_length
            self.max_length = max_length

        def __call__(self, instance):
            if not hasattr(instance.metadata, "shortest_path_length"):
                return False
            length = instance.metadata.shortest_path_length
            if self.min_length is not None and length < self.min_length:
                return False
            if self.max_length is not None and length > self.max_length:
                return False
            return True

    val_filter = NumBoxesFilter(2)
    test_filter = PathLengthFilter(max_length=10)
    val_ids = {inst.id for inst in instances if val_filter(inst)}
    # remove anything already tagged as validation
    test_ids = {
        inst.id for inst in instances if test_filter(inst) and inst.id not in val_ids
    }
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Sokoban Procedural TaskSet",
        description="Procedurally generated Sokoban tasks with varying difficulty.",
        instances=instances,
        split_info=split_info,
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    import json
    import os

    NUM_INSTANCES_PER_DIFFICULTY = 2
    # Updated path to examples/sokoban/dataset/instances.json
    OUTPUT_FILE_PATH = "dataset/instances.json"

    async def main():
        taskset = await create_sokoban_taskset()

        serialized = await asyncio.gather(
            *(inst.serialize() for inst in taskset.instances)
        )

        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"Serialized {len(serialized)} instances to {OUTPUT_FILE_PATH}")

        with open(OUTPUT_FILE_PATH, "r") as f:
            read_serialized_data = json.load(f)

        deserialized = await asyncio.gather(
            *(SokobanTaskInstance.deserialize(data) for data in read_serialized_data)
        )
        print(f"Deserialized {len(deserialized)} instances.")

        if any(inst is None for inst in deserialized):
            print("Error: Deserialization returned None for some instances.")
            for i, inst in enumerate(deserialized):
                if inst is None:
                    print(
                        f"Instance at index {i} is None. Serialized data: {read_serialized_data[i]}"
                    )
            return

        val_ids = taskset.split_info.val_instance_ids
        test_ids = taskset.split_info.test_instance_ids
        all_ids = {inst.id for inst in deserialized}
        train_ids = all_ids - val_ids - test_ids

        train = [inst for inst in deserialized if inst.id in train_ids]
        val = [inst for inst in deserialized if inst.id in val_ids]
        test = [inst for inst in deserialized if inst.id in test_ids]

        print(f"Train set ({len(train)} instances): {[str(i.id) for i in train]}")
        print(f"Val set ({len(val)} instances): {[str(i.id) for i in val]}")
        print(f"Test set ({len(test)} instances): {[str(i.id) for i in test]}")

    asyncio.run(main())
