"""MiniGrid Debug Runner

Run with uv:
  uv run python -m horizons.environments.examples.minigrid.agent_demos.minigrid_debug_runner --actions right forward forward

You can also specify a difficulty and index into the curated deterministic puzzles.
"""

import argparse
import asyncio
from typing import List

from horizons.environments.examples.minigrid.environment import MiniGridEnvironment
from horizons.environments.examples.minigrid.puzzle_loader import get_puzzle_by_index
from horizons.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK, create_minigrid_task_from_puzzle


async def run(actions: List[str], difficulty: str | None, index: int | None):
    if difficulty and index is not None:
        puzzle = get_puzzle_by_index(difficulty, index)
        if puzzle is None:
            raise SystemExit(f"No puzzle for difficulty={difficulty} index={index}")
        task = await create_minigrid_task_from_puzzle(puzzle)
    else:
        task = DEFAULT_MINIGRID_TASK

    env = MiniGridEnvironment(task)
    obs = await env.initialize()
    print("=== INIT ===")
    print("env:", {k: obs.get(k) for k in ("env_name", "seed", "difficulty")})
    print("agent:", obs.get("agent_debug"))
    print("front_cell:", obs.get("front_cell"))
    print("available_actions:", obs.get("available_actions"))
    print("terminated/truncated:", obs.get("terminated"), obs.get("truncated"))

    if actions:
        print("\n=== STEP (multi-action) ===")
        tool_call = {"tool": "minigrid_act", "args": {"actions": actions}}
        obs2 = await env.step(tool_call)
        print("executed:", obs2.get("executed_actions"))
        print("agent:", obs2.get("agent_debug"))
        print("front_cell:", obs2.get("front_cell"))
        print("terminated/truncated:", obs2.get("terminated"), obs2.get("truncated"))

    fin = await env.terminate()
    print("\n=== FINAL ===")
    print("final_position:", fin.get("final_position"))
    print("total_steps:", fin.get("total_steps"))
    print("total_reward:", fin.get("total_reward"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="*", default=[], help="Actions to execute consecutively")
    ap.add_argument("--difficulty", choices=["ultra_easy", "easy", "medium", "hard"], default=None)
    ap.add_argument("--index", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args.actions, args.difficulty, args.index))


if __name__ == "__main__":
    main()

