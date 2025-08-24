#!/usr/bin/env python3
"""
TicTacToe Environment Validation Script

This script validates that the TicTacToe environment is properly
set up and integrated with the Horizons AI framework.
"""

import asyncio
import sys
from uuid import uuid4

from horizons.environments.tasks.core import Impetus, Intent, TaskInstance, TaskInstanceMetadata
from horizons.environments.examples.tictactoe.environment import TicTacToeEnvironment
from horizons.environments.environment.tools import EnvToolCall


async def validate_environment():
    """Validate TicTacToe environment functionality."""
    print("🔍 Validating TicTacToe Environment Setup")
    print("=" * 50)

    # Create task instance
    task_instance = TaskInstance(
        id=uuid4(),
        impetus=Impetus("Validate TicTacToe environment functionality."),
        intent=Intent(
            rubric={"validation": "pass all tests"},
            gold_trajectories=None,
            gold_state_diff={}
        ),
        metadata=TaskInstanceMetadata(),
        is_reproducible=True,
        initial_engine_snapshot=None
    )

    try:
        # Test 1: Environment Creation
        print("✅ Test 1: Environment Creation")
        env = TicTacToeEnvironment(task_instance)
        print("   ✓ TicTacToeEnvironment created successfully")

        # Test 2: Environment Initialization
        print("✅ Test 2: Environment Initialization")
        obs = await env.initialize()
        print("   ✓ Environment initialized successfully")
        print(f"   ✓ Initial observation keys: {list(obs.public_observation.keys())}")

        # Test 3: Board State Validation
        print("✅ Test 3: Board State Validation")
        board_text = obs.public_observation.get('board_text', '')
        assert board_text, "Board text missing from observation"
        print("   ✓ Board text present and valid")
        print("   ✓ Initial board state:")
        print(f"   {board_text.replace(chr(10), chr(10) + '   ')}")

        # Test 4: Valid Move Execution
        print("✅ Test 4: Valid Move Execution")
        tool_call = EnvToolCall(
            tool="interact",
            args={"letter": "A", "number": 1}
        )
        env.validate_tool_calls([tool_call])
        obs = await env.step([tool_call])
        print("   ✓ Valid move executed successfully")
        print(f"   ✓ Move count: {obs.public_observation['move_count']}")
        print(f"   ✓ Last move: {obs.public_observation['last_move']}")

        # Test 5: Invalid Move Validation
        print("✅ Test 5: Invalid Move Validation")
        invalid_tool_call = EnvToolCall(
            tool="interact",
            args={"letter": "D", "number": 1}  # Invalid letter
        )
        try:
            env.validate_tool_calls([invalid_tool_call])
            print("   ❌ Should have raised validation error")
            return False
        except ValueError as e:
            print(f"   ✓ Invalid move properly rejected: {e}")

        # Test 6: Environment Termination
        print("✅ Test 6: Environment Termination")
        final_obs = await env.terminate()
        print("   ✓ Environment terminated successfully")
        print(f"   ✓ Termination message: {final_obs.public_observation.get('message', 'N/A')}")

        # Test 7: Checkpoint Functionality
        print("✅ Test 7: Checkpoint Functionality")
        env2 = TicTacToeEnvironment(task_instance)
        await env2.initialize()
        await env2.step([EnvToolCall(tool="interact", args={"letter": "B", "number": 2})])
        checkpoint_obs = await env2.checkpoint()
        print("   ✓ Checkpoint created successfully")
        print(f"   ✓ Checkpoint contains: {list(checkpoint_obs.public_observation.keys())}")

        print("\n🎉 All validation tests passed!")
        print("✅ TicTacToe environment is properly integrated with Horizons AI")
        return True

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main validation function."""
    success = await validate_environment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
