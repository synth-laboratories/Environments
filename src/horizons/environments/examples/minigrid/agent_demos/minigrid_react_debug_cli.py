"""MiniGrid ReAct Agent Debug CLI

Run a short evaluation with detailed debug logs.

Examples:
  uv run python -m horizons.environments.examples.minigrid.agent_demos.minigrid_react_debug_cli \
      --difficulty easy --num-tasks 2 --verbose

  uv run python -m horizons.environments.examples.minigrid.agent_demos.minigrid_react_debug_cli \
      --model gpt-5-nano --difficulty medium --num-tasks 1
"""

import argparse
import asyncio
from typing import Any, Dict

from .minigrid_react_agent import eval_minigrid_react


async def _amain(model: str, difficulty: str, num_tasks: int, verbose: bool) -> Dict[str, Any]:
    return await eval_minigrid_react(
        model_name=model, difficulty=difficulty, num_tasks=num_tasks, verbose=verbose
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-nano", help="LLM model name (default: gpt-5-nano)")
    ap.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Task difficulty",
    )
    ap.add_argument("--num-tasks", type=int, default=1, help="Number of tasks to run")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    asyncio.run(_amain(args.model, args.difficulty, args.num_tasks, args.verbose))


if __name__ == "__main__":
    main()

