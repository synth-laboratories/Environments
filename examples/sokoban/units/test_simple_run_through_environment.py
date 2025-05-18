"""
test_sokoban_environment.py – A*‑style search and replay, but through the
*SokobanEnvironment* API (initialize/step/checkpoint) rather than talking to
SokobanEngine directly.
"""

import asyncio, heapq, itertools, json
from typing import List, Tuple, Any, Dict
from uuid import uuid4

import numpy as np
import pytest

# ––––– app imports ––––– #
from examples.sokoban.environment import SokobanEnvironment  # <- your wrapper
from examples.sokoban.engine import SokobanEngineSnapshot  # same snapshot type
from src.environment.tools import EnvToolCall  # call interface

from examples.sokoban.taskset import (
    SokobanTaskInstanceMetadata,
    SokobanTaskInstance,
)
from src.tasks.core import TaskInstance, Impetus, Intent


# ---------------- test fixture snapshot ---------------------------------- #
SIMPLE_SNAPSHOT: Dict[str, Any] = {
    "dim_room": [4, 4],
    "room_fixed": [[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    "room_state": [[0, 0, 0, 0], [0, 1, 4, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


# helper: tiny wrapper so we don’t depend on full EnvToolCall implementation
class Move(EnvToolCall):  # type: ignore[misc]
    def __init__(self, action: int):
        self.action = action


# ---------------- utility functions -------------------------------------- #
def solved(environment: SokobanEnvironment) -> bool:
    env = environment.engine.package_sokoban_env
    return env.boxes_on_target == np.sum(env.room_fixed == 2)


def heuristic(environment: SokobanEnvironment) -> int:
    env = environment.engine.package_sokoban_env
    return np.sum(env.room_fixed == 2) - env.boxes_on_target


async def astar(env: SokobanEnvironment, max_nodes: int = 500) -> List[int]:
    """A* that uses env._serialize_engine / _deserialize_engine + env.step."""
    start_snap: SokobanEngineSnapshot = await env._serialize_engine()
    frontier: List[Tuple[int, int, SokobanEngineSnapshot, List[int]]] = []
    counter = itertools.count()
    frontier.append((heuristic(env), next(counter), start_snap, []))
    seen: set[str] = set()

    nodes = 0
    while frontier and nodes < max_nodes:
        f, _, snap, path = heapq.heappop(frontier)
        env = await SokobanEnvironment._deserialize_engine(snap)
        key = json.dumps(snap.engine_snapshot, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if solved(env):
            return path
        nodes += 1

        for action in range(env.engine.package_sokoban_env.action_space.n):
            await env.step([[Move(action)]])
            child_snap = await env._serialize_engine()
            heapq.heappush(
                frontier,
                (
                    len(path) + 1 + heuristic(env),
                    next(counter),
                    child_snap,
                    path + [action],
                ),
            )
    return []


async def replay(
    env: SokobanEnvironment, start: SokobanEngineSnapshot, plan: List[int]
) -> bool:
    env = await SokobanEnvironment._deserialize_engine(start)
    for a in plan:
        await env.step([[Move(a)]])
    return solved(env)


# ----------------------------- test -------------------------------------- #
@pytest.mark.asyncio
async def test_environment_solve_and_replay():
    # build minimal TaskInstance
    meta = SokobanTaskInstanceMetadata(
        difficulty="easy",
        num_boxes=1,
        dim_room=(4, 4),
        max_steps=10,
        shortest_path_length=-1,
        seed=-1,
        generation_params="unit‑test",
    )
    ti = SokobanTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=SIMPLE_SNAPSHOT,
    )

    env = SokobanEnvironment(ti)
    await env.initialize()
    root_snapshot = await env._serialize_engine()

    # plan search
    plan = await astar(env)
    assert plan, "Environment A* failed to find a plan"

    # verify replay
    ok = await replay(env, root_snapshot, plan)
    assert ok, "Plan failed to solve puzzle on replay"
