"""
Test suite for SokobanEngine A* solver and replay functionality.
"""

import asyncio, heapq, json, numpy as np, itertools
from pathlib import Path
from typing import Any, List, Tuple
import pytest # Added pytest import
from uuid import uuid4 # Added UUID import

from examples.sokoban.engine import (
    SokobanEngine,
    # package_sokoban_env_from_engine_snapshot, # Not directly used in test logic
    SokobanEngineSnapshot, # Not directly used in test logic
) 
from src.tasks.core import TaskInstance, Impetus, Intent, TaskInstanceMetadata # TaskInstanceMetadata for base
from examples.sokoban.taskset import SokobanTaskInstanceMetadata, SokobanTaskInstance # Import the concrete class for task instances


# ---------- Test Data: A Simple Solvable Snapshot ------------------------ #
# This snapshot represents a very simple Sokoban puzzle.
# Player (5) at (2,1), Box (4) at (1,2), Target (2) at (1,1)
# Solution: Move Up (push box to target)
SIMPLE_TEST_SNAPSHOT = {
    "dim_room": [4, 4],  # Dimensions of the room
    "room_fixed": [
        [0,0,0,0],
        [0,2,1,0], # Target at (1,1), Empty at (1,2) initially (box will be here)
        [0,1,0,0], # Empty at (2,1) initially (player will be here)
        [0,0,0,0]
    ],
    "room_state": [
        [0,0,0,0],
        [0,1,4,0], # Box at (1,2)
        [0,5,0,0], # Player at (2,1)
        [0,0,0,0]
    ],
    "boxes_on_target": 0,
    "step_count": 0,
    "max_steps": 10,
    "num_boxes": 1,
    "player_pos": [2,1] # Added for clarity, though SokobanEngine re-derives it
}

# ---------- Solver/Replay Logic (from original script) ------------------- #

def solved(env) -> bool:
    """All targets covered?"""
    return env.package_sokoban_env.boxes_on_target == np.sum(
        env.package_sokoban_env.room_fixed == 2
    )

def heuristic(env) -> int:
    # Simple heuristic: number of boxes not on targets
    # More sophisticated heuristics could use Manhattan distance to nearest target
    return np.sum(env.package_sokoban_env.room_fixed == 2) - env.package_sokoban_env.boxes_on_target

async def astar(engine: SokobanEngine, max_nodes: int = 1000) -> List[int]:
    """A* search using engine._serialize_engine()/ _deserialize_engine()."""
    # Capture initial state and set up a counter for tie-breaking
    start_snap: SokobanEngineSnapshot = await engine._serialize_engine()
    counter = itertools.count()
    frontier: List[Tuple[int, int, SokobanEngineSnapshot, List[int]]] = [
        (heuristic(engine), next(counter), start_snap, [])
    ]
    seen: set[str] = set()

    nodes_expanded = 0
    while frontier and nodes_expanded < max_nodes:
        f, _, snap, path = heapq.heappop(frontier)
        # Restore engine from snapshot
        engine = await engine._deserialize_engine(snap)
        key = json.dumps(snap.engine_snapshot, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if solved(engine):
            return path
        nodes_expanded += 1

        # Iterate possible actions by index since action_space is Discrete
        for action in range(engine.package_sokoban_env.action_space.n):
            # Step engine forward
            await engine._step_engine(action)
            child_snap = await engine._serialize_engine()
            f2 = f + 1 + heuristic(engine)
            heapq.heappush(frontier, (f2, next(counter), child_snap, path + [action]))
    return []

async def replay(engine: SokobanEngine, start_snap: SokobanEngineSnapshot, actions: List[int]) -> bool:
    """Re-run actions from start snapshot and verify solved state."""
    # Restore engine to initial snapshot
    engine = await engine._deserialize_engine(start_snap)
    for a in actions:
        await engine._step_engine(a)
    return solved(engine)

# ---------- Pytest Test Function ----------------------------------------- #

@pytest.mark.asyncio
async def test_solve_and_replay_simple_snapshot():
    """Tests A* solver and replay functionality with a simple snapshot."""
    # Create a dummy TaskInstance with the test snapshot
    # Using SokobanTaskInstanceMetadata as it's now more defined
    dummy_metadata = SokobanTaskInstanceMetadata(
        difficulty="easy",
        num_boxes=SIMPLE_TEST_SNAPSHOT["num_boxes"],
        dim_room=(len(SIMPLE_TEST_SNAPSHOT["room_fixed"]), len(SIMPLE_TEST_SNAPSHOT["room_fixed"][0])),
        max_steps=SIMPLE_TEST_SNAPSHOT["max_steps"],
        shortest_path_length=-1, # Not known/relevant for this test setup
        seed=-1, # Not relevant for this test setup
        generation_params="test_snapshot"
    )
    task_instance = SokobanTaskInstance(
        id=uuid4(), # Generate a UUID for the id
        impetus=Impetus(instructions="solve test puzzle"),
        intent=Intent(rubric={"goal": "cover all targets"}, gold_trajectories=None, gold_state_diff={}),
        metadata=dummy_metadata, # Use the specific metadata object
        is_reproducible=True,
        initial_engine_snapshot=SIMPLE_TEST_SNAPSHOT,
    )

    engine = SokobanEngine(task_instance)
    await engine._reset_engine()  # hydrates from snapshot
    root_snap = await engine._serialize_engine()

    # Find a plan using A*
    plan = await astar(engine)
    # Print solution length for manual verification
    print(f"Solution length: {len(plan)} actions")
    assert plan, "A* should find a solution for the simple snapshot."
    assert len(plan) > 0, "Plan should not be empty."

    # Replay the plan and verify it solves the puzzle
    is_verified = await replay(engine, root_snap, plan)
    # Print verification result for manual checking
    print(f"Verified solved state: {is_verified}")
    assert is_verified, "Replaying the plan should result in a solved state."
