#!/usr/bin/env python3
"""
TicTacToe Environment Demo

This script demonstrates how to use the TicTacToe environment
in the Horizons AI framework.
"""

import asyncio
from uuid import uuid4

from horizons.environments.tasks.core import Impetus, Intent, TaskInstance, TaskInstanceMetadata
from horizons.environments.examples.tictactoe.environment import TicTacToeEnvironment
from horizons.environments.environment.tools import EnvToolCall


async def main():
    """Demonstrate TicTacToe environment usage."""
    print("🎮 TicTacToe Environment Demo")
    print("=" * 40)

    # Create a task instance
    task_instance = TaskInstance(
        id=uuid4(),
        impetus=Impetus("Play TicTacToe and try to win!"),
        intent=Intent(
            rubric={"goal": "win the game"},
            gold_trajectories=None,
            gold_state_diff={}
        ),
        metadata=TaskInstanceMetadata(),
        is_reproducible=True,
        initial_engine_snapshot=None
    )

    # Create and initialize environment
    print("🏁 Initializing TicTacToe environment...")
    env = TicTacToeEnvironment(task_instance)
    obs = await env.initialize()

    print("📊 Initial board state:")
    print(obs.public_observation['board_text'])
    print(f"🎯 Current player: {obs.public_observation['current_player']}")
    print(f"📈 Move count: {obs.public_observation['move_count']}")
    print()

    # Make some moves
    moves = [
        ("A", 1, "X"),  # X plays A1
        ("B", 1, "O"),  # O plays B1
        ("A", 2, "X"),  # X plays A2
        ("B", 2, "O"),  # O plays B2
        ("A", 3, "X"),  # X plays A3 (wins!)
    ]

    for i, (letter, number, player) in enumerate(moves, 1):
        print(f"🎲 Move {i}: {player} plays {letter}{number}")

        # Create tool call
        tool_call = EnvToolCall(
            tool="interact",
            args={"letter": letter, "number": number}
        )

        # Validate and execute move
        try:
            env.validate_tool_calls([tool_call])
            obs = await env.step([tool_call])

            # Display board state
            print("📊 Board after move:")
            print(obs.public_observation['board_text'])
            print(f"🎯 Current player: {obs.public_observation['current_player']}")
            print(f"📈 Move count: {obs.public_observation['move_count']}")

            # Check for winner
            if obs.public_observation['winner']:
                print(f"🏆 Winner: {obs.public_observation['winner']}!")
                break

            print()

        except ValueError as e:
            print(f"❌ Invalid move: {e}")
            break

    # Terminate environment
    print("🏁 Terminating environment...")
    final_obs = await env.terminate()
    print(f"💡 Final message: {final_obs.public_observation['message']}")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
