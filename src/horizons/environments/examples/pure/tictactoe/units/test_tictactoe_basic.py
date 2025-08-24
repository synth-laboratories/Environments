"""Basic tests for TicTacToe environment."""

import pytest
import asyncio
from uuid import uuid4

from horizons.environments.tasks.core import Impetus, Intent, TaskInstance, TaskInstanceMetadata
from horizons.environments.examples.tictactoe.environment import TicTacToeEnvironment
from horizons.environments.examples.tictactoe.engine import COORD_TO_IDX
from horizons.environments.environment.tools import EnvToolCall


class TestTicTacToeBasic:
    """Basic functionality tests for TicTacToe environment."""

    @pytest.fixture
    def task_instance(self):
        """Create a basic task instance for testing."""
        return TaskInstance(
            id=uuid4(),
            impetus=Impetus("Play TicTacToe and win the game."),
            intent=Intent(rubric={"goal": "win the game"}),
            metadata=TaskInstanceMetadata(),
            is_reproducible=True,
            initial_engine_snapshot=None
        )

    @pytest.mark.asyncio
    async def test_environment_initialization(self, task_instance):
        """Test that environment initializes correctly."""
        env = TicTacToeEnvironment(task_instance)
        obs = await env.initialize()

        # Check that observation has expected structure
        assert obs is not None
        assert hasattr(obs, 'public_observation')
        assert 'board_text' in obs.public_observation

    @pytest.mark.asyncio
    async def test_valid_move(self, task_instance):
        """Test making a valid move."""
        env = TicTacToeEnvironment(task_instance)
        await env.initialize()

        # Make a valid move
        tool_call = EnvToolCall(
            name="interact",
            args={"letter": "A", "number": 1}
        )

        # Validate the move
        env.validate_tool_calls([tool_call])

        # Execute the move
        obs = await env.step([tool_call])

        # Check that move was made
        assert obs is not None
        assert obs.public_observation['move_count'] == 1
        assert obs.public_observation['last_move'] == "A1"

    @pytest.mark.asyncio
    async def test_invalid_move_validation(self, task_instance):
        """Test that invalid moves are rejected."""
        env = TicTacToeEnvironment(task_instance)
        await env.initialize()

        # Try an invalid move (out of bounds)
        tool_call = EnvToolCall(
            name="interact",
            args={"letter": "D", "number": 1}  # D is invalid
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="Invalid letter"):
            env.validate_tool_calls([tool_call])

    @pytest.mark.asyncio
    async def test_game_completion(self, task_instance):
        """Test that game can reach completion."""
        env = TicTacToeEnvironment(task_instance)
        await env.initialize()

        # Make moves to complete a row
        moves = [
            ("A", 1), ("B", 1),  # X: A1, O: B1
            ("A", 2), ("B", 2),  # X: A2, O: B2
            ("A", 3)             # X: A3 (win)
        ]

        for i, (letter, number) in enumerate(moves):
            tool_call = EnvToolCall(
                name="interact",
                args={"letter": letter, "number": number}
            )

            env.validate_tool_calls([tool_call])
            obs = await env.step([tool_call])

            # After 5 moves, X should win
            if i == 4:  # After X's third move
                assert obs.public_observation['winner'] == "X"
                assert obs.public_observation['terminated'] is True

    @pytest.mark.asyncio
    async def test_environment_termination(self, task_instance):
        """Test environment termination."""
        env = TicTacToeEnvironment(task_instance)
        await env.initialize()

        # Terminate the environment
        obs = await env.terminate()

        # Check termination state
        assert obs.public_observation['terminated'] is True
        assert 'message' in obs.public_observation

    @pytest.mark.asyncio
    async def test_checkpoint_functionality(self, task_instance):
        """Test checkpoint creation and restoration."""
        env = TicTacToeEnvironment(task_instance)
        await env.initialize()

        # Make a move
        tool_call = EnvToolCall(
            name="interact",
            args={"letter": "A", "number": 1}
        )
        await env.step([tool_call])

        # Create checkpoint
        checkpoint_obs = await env.checkpoint()
        assert checkpoint_obs is not None

        # Checkpoint should contain game state
        assert 'board_text' in checkpoint_obs.public_observation
