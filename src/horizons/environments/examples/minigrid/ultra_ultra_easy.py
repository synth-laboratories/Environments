"""Generator for ultra_ultra_easy MiniGrid tasks (~7-step median).

Strategy:
- Use the simplest, realistic environment: MiniGrid-Empty-5x5-v0 (topology: walls + 1 goal)
- Randomly scan seeds and compute the shortest path length (grid steps) from agent to goal
- Keep seeds whose path length is within a small band (e.g., 5..9) to target ~7 median
- Build MiniGridPuzzle objects with difficulty="ultra_ultra_easy" and realistic metadata

This produces deterministic, valid tasks because we pin the Gym seed per puzzle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import gymnasium as gym

from horizons.environments.examples.minigrid.puzzle_loader import MiniGridPuzzle
from horizons.environments.examples.minigrid.taskset import (
    MiniGridTaskInstance,
    MiniGridTaskInstanceMetadata,
)
from horizons.environments.tasks.api import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstanceSet,
)
from pathlib import Path
import json


def _find_goal_pos(unwrapped) -> Optional[Tuple[int, int]]:
    grid = unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                return (x, y)
    return None


def _bfs_shortest_path_len(unwrapped, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    width, height = unwrapped.grid.width, unwrapped.grid.height
    walls: Set[Tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "wall":
                walls.add((x, y))

    def neighbors(p: Tuple[int, int]):
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield (nx, ny)

    q: Deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
    seen: Set[Tuple[int, int]] = {start}
    while q:
        pos, d = q.popleft()
        if pos == goal:
            return d
        for nb in neighbors(pos):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _estimate_steps(unwrapped) -> Optional[int]:
    """Estimate steps as shortest path length; simple, realistic for Empty envs."""
    agent = tuple(unwrapped.agent_pos) if unwrapped.agent_pos is not None else None
    goal = _find_goal_pos(unwrapped)
    if agent is None or goal is None:
        return None
    return _bfs_shortest_path_len(unwrapped, agent, goal)


def create_ultra_ultra_easy_taskset(
    num_instances: int = 30,
    target_range: Tuple[int, int] = (6, 8),
    max_candidates: int = 5000,
    env_names: Optional[List[str]] = None,
) -> TaskInstanceSet:
    """Create a set of ~7-step median tasks by sampling and filtering.

    Args:
        num_instances: how many tasks to generate
        target_range: inclusive [min,max] for BFS steps to accept
        max_candidates: cap on seed search to avoid long loops

    Returns:
        TaskInstanceSet with the generated tasks
    """
    env_names = env_names or [
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-Empty-Random-5x5-v0",
    ]
    env_index = 0
    env = gym.make(env_names[env_index])

    accepted: List[MiniGridPuzzle] = []
    seen_seeds: Set[int] = set()
    seed = 0
    while len(accepted) < num_instances and seed < max_candidates:
        if seed in seen_seeds:
            seed += 1
            continue
        seen_seeds.add(seed)

        # Rotate environments to diversify distances
        env_name = env_names[env_index]
        env_index = (env_index + 1) % len(env_names)
        if env.spec and env.spec.id != env_name:
            env.close()
            env = gym.make(env_name)
        env.reset(seed=seed)
        unwrapped = env.unwrapped
        steps = _estimate_steps(unwrapped)
        if steps is None:
            seed += 1
            continue

        if target_range[0] <= steps <= target_range[1]:
            turn_penalty = 1  # account for at least one turn on average
            puzzle = MiniGridPuzzle(
                id=f"ultra_ultra_easy_{seed:04d}",
                environment_name=env_name,
                difficulty="ultra_ultra_easy",
                seed=seed,
                grid_size=(unwrapped.grid.width, unwrapped.grid.height),
                mission_description="Reach the goal tile (G) in the smallest number of moves.",
                has_key=False,
                has_door=False,
                has_lava=False,
                has_multi_room=False,
                num_objects=0,
                complexity_score=max(0.0, (unwrapped.grid.width * unwrapped.grid.height) / 100.0),
                estimated_steps=int(steps + turn_penalty),
            )
            accepted.append(puzzle)
        seed += 1

    # Convert to TaskInstanceSet
    instances: List[MiniGridTaskInstance] = []
    for p in accepted:
        metadata = MiniGridTaskInstanceMetadata(
            env_name=p.environment_name,
            grid_size=p.grid_size,
            difficulty=p.difficulty,
            has_key=p.has_key,
            has_door=p.has_door,
            has_lava=p.has_lava,
            num_objects=p.num_objects,
            optimal_path_length=p.estimated_steps,
            seed=p.seed,
        )
        instance = MiniGridTaskInstance(
            id=uuid4(),
            impetus=Impetus(
                instructions=(
                    "Navigate to the goal (G). Use 'left'/'right' to turn and 'forward' to move."
                )
            ),
            intent=Intent(
                rubric={
                    "goal": "Reach the goal tile efficiently",
                    "success_criteria": ["Reach goal", "Avoid repeating blocked moves"],
                },
                gold_trajectories=None,
                gold_state_diff={},
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        instances.append(instance)

    # Simple 80/10/10 split
    n_total = len(instances)
    n_val = max(1, n_total // 10)
    n_test = max(1, n_total // 10)
    val_ids = {inst.id for inst in instances[:n_val]}
    test_ids = {inst.id for inst in instances[n_val : n_val + n_test]}
    split_info = SplitInfo(val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True)

    return TaskInstanceSet(
        name="MiniGrid UltraUltraEasy",
        description=(
            f"Auto-generated ultra_ultra_easy tasks from {env_name} with path length in {target_range}, "
            f"count={len(instances)}"
        ),
        instances=instances,
        split_info=split_info,
    )


def export_ultra_ultra_easy_json(
    out_path: Path | str = None,
    num_instances: int = 30,
    target_range: Tuple[int, int] = (6, 8),
    max_candidates: int = 5000,
) -> Path:
    """Generate and export puzzles to a persistent JSON for distribution.

    The file matches the shape used by other loaders: {"metadata": {...}, "puzzles": {difficulty: [..]}}
    """
    ts = create_ultra_ultra_easy_taskset(
        num_instances=num_instances, target_range=target_range, max_candidates=max_candidates
    )
    puzzles = []
    for inst in ts.instances:
        # Reconstruct a MiniGridPuzzle-equivalent dict from instance metadata
        puzzles.append(
            {
                "id": f"ultra_ultra_easy_{inst.metadata.seed:04d}",
                "environment_name": inst.metadata.env_name,
                "difficulty": "ultra_ultra_easy",
                "seed": inst.metadata.seed,
                "grid_size": list(inst.metadata.grid_size),
                "mission_description": inst.impetus.instructions,
                "has_key": inst.metadata.has_key,
                "has_door": inst.metadata.has_door,
                "has_lava": inst.metadata.has_lava,
                "has_multi_room": False,
                "num_objects": inst.metadata.num_objects,
                "complexity_score": max(
                    0.0, (inst.metadata.grid_size[0] * inst.metadata.grid_size[1]) / 100.0
                ),
                "estimated_steps": inst.metadata.optimal_path_length or 0,
            }
        )

    payload = {
        "metadata": {
            "name": "ultra_ultra_easy",
            "description": "Auto-generated short-path MiniGrid tasks (median ~7 moves)",
            "count": len(puzzles),
        },
        "puzzles": {"ultra_ultra_easy": puzzles},
    }

    if out_path is None:
        out_path = Path(__file__).parent / "ultra_ultra_easy_puzzles.json"
    else:
        out_path = Path(out_path)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path
