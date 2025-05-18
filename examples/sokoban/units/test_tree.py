#!/usr/bin/env python3
"""
mcts_sokoban_env_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tiny Monte-Carlo-Tree-Search demo that
  • wraps a 4×4 toy Sokoban level in `SokobanEnvironment`
  • stores every state in a FilesystemSnapshotStore
  • expands / rolls-out with a TrajectoryTreeStore
  • returns the most visited root-child as the "plan"

Run with pytest: pytest Environments/examples/sokoban/units/test_tree.py
"""

import asyncio, gzip, math, pickle, random, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore, TrajectorySnapshot
from examples.sokoban.taskset import SokobanTaskInstance, SokobanTaskInstanceMetadata
from examples.sokoban.engine import SokobanEngineSnapshot  # only a type
from src.tasks.core import Impetus, Intent
from examples.sokoban.environment import SokobanEnvironment

# ─────────────────────────── toy level ──────────────────────────────── #

SNAP = {
    "dim_room": [4, 4],
    "room_fixed": [  # target is in the top-right corner
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "room_state": [  # player under the box
        [0, 0, 0, 0],
        [0, 4, 0, 0],  # box starts at (1,1)
        [5, 0, 0, 0],  # player at (2,0)
        [0, 0, 0, 0],
    ],
    "boxes_on_target": 0,
    "num_boxes": 1,
    "max_steps": 30,
}

# ─────────────────────────── env wrapper ────────────────────────────── #
# (import placed here to avoid circulars; uses the code you pasted)

# ─────────────────────────── helpers ─────────────────────────────────── #


def solved(env: SokobanEnvironment) -> bool:
    """All targets covered?"""
    eng = env.engine
    return eng.package_sokoban_env.boxes_on_target == np.sum(
        eng.package_sokoban_env.room_fixed == 2
    )


def ucb(child_stats, parent_visits, c: float = 1.4) -> float:
    if child_stats["visits"] == 0:
        return float("inf")
    return child_stats["value"] / child_stats["visits"] + c * math.sqrt(
        math.log(parent_visits) / child_stats["visits"]
    )


# ╭─────────────────────────── MCTS core ───────────────────────────────╮ #


async def mcts(
    tree: TrajectoryTreeStore,
    root_id: str,
    *,
    rollouts: int = 300,
    max_depth: int = 30,
) -> List[int]:
    """
    Very small ("vanilla") MCTS loop – no prior policy, no virtual-loss, etc.
    Works with any domain that can be snapshotted / deserialised through
    `SokobanEnvironment._deserialize_engine`.
    """
    stats: Dict[str, Dict[str, float]] = {root_id: dict(visits=0, value=0.0)}

    for _ in range(rollouts):
        # ── SELECTION ────────────────────────────────────────────────
        path: List[str] = [root_id]
        node_id: str = root_id
        depth: int = 0

        while tree.get_children(node_id):
            parent_vis = stats[node_id]["visits"]
            node_id = max(
                tree.get_children(node_id),
                key=lambda cid: ucb(
                    stats.get(cid, {"visits": 0, "value": 0}), parent_vis
                ),
            )
            path.append(node_id)
            depth += 1
            if depth >= max_depth:
                break

        # ── EXPANSION / SIMULATION ──────────────────────────────────
        # ‣ re-hydrate environment from the selected leaf snapshot
        leaf_blob = tree.load_snapshot_blob(node_id)
        leaf_snap = pickle.loads(gzip.decompress(leaf_blob))
        env: SokobanEnvironment = await SokobanEnvironment._deserialize_engine(
            leaf_snap
        )

        if solved(env):
            reward = 1.0
        else:
            # pick one random legal action for expansion
            legal = list(range(env.engine.package_sokoban_env.action_space.n))
            random.shuffle(legal)
            action = legal[0]
            await env.engine._step_engine(action)

            # materialise new child node
            child_snap = await env._serialize_engine()
            child_blob = gzip.compress(pickle.dumps(child_snap))
            child_id = tree.add_child(
                node_id,
                child_blob,
                action=action,
                reward=0.0,
                terminated=False,
                info={},
            )
            stats.setdefault(child_id, dict(visits=0, value=0.0))
            path.append(child_id)

            # random rollout from that child
            reward = 0.0
            steps = 0
            while not solved(env) and steps < max_depth:
                await env.engine._step_engine(random.choice(legal))
                steps += 1
            if solved(env):
                reward = 1.0

        # ── BACK-PROP ────────────────────────────────────────────────
        for nid in reversed(path):
            rec = stats.setdefault(nid, dict(visits=0, value=0.0))
            rec["visits"] += 1
            rec["value"] += reward

    # choose the most visited root-child as our "plan root"
    best_child = max(
        tree.get_children(root_id), key=lambda cid: stats[cid]["visits"], default=None
    )
    plan: List[int] = []
    cur = best_child
    while cur and cur != root_id:
        par = tree.get_parent(cur)
        plan.append(tree.graph[par][cur]["action"])
        cur = par
    return list(reversed(plan))


# ╰────────────────────────────────────────────────────────────────────╯ #

# ─────────────────────────── driver / demo ──────────────────────────── #


@pytest.mark.asyncio
async def test_mcts_sokoban_run(tmp_path: Path) -> None:
    # 1. build task-instance + env
    inst = SokobanTaskInstance(
        id="demo-mcts",
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=SokobanTaskInstanceMetadata(
            difficulty="easy",
            num_boxes=1,
            dim_room=(4, 4),
            max_steps=10,
            shortest_path_length=-1,
            seed=0,
            generation_params="demo",
        ),
        is_reproducible=True,
        initial_engine_snapshot=SNAP,
    )
    env = SokobanEnvironment(inst)
    await env.initialize()

    # 2. root snapshot → tree
    snap_store_path = tmp_path / "mcts_snaps"
    snap_store = FilesystemSnapshotStore(snap_store_path)
    tree = TrajectoryTreeStore(snap_store)

    root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
    root_id = tree.add_root(root_blob)

    # 3. run search
    plan = await mcts(tree, root_id, rollouts=500) # Increased rollouts for robustness
    print(f"MCTS plan: {plan}")
    assert len(plan) > 1, "MCTS plan should be longer than 1 action for this puzzle."

    # 4. verify
    ver_env = await SokobanEnvironment._deserialize_engine(
        pickle.loads(gzip.decompress(root_blob))
    )
    for a in plan:
        await ver_env.engine._step_engine(a)
    solved_status = solved(ver_env)
    print(f"Solved status: {solved_status}")
    assert solved_status, "The MCTS plan should solve the puzzle"


# Removed if __name__ == "__main__": block as pytest handles execution
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(test_mcts_sokoban_run(Path(tmpdir)))
