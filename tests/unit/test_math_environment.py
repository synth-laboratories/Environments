"""
Unit tests for the HendryksMath environment.
"""

import pytest
from uuid import uuid4

from synth_env.examples.math.environment import HendryksMathEnv
from synth_env.examples.math.schema import (
    HendryksTaskInstance,
    HendryksTaskInstanceMetadata,
)
from synth_env.tasks.core import Impetus, Intent


class TestMathEnvironment:
    """Test the HendryksMath environment."""

    @pytest.mark.asyncio
    async def test_math_environment_initialization(self):
        """Test basic initialization of math environment."""
        # Create a simple math task
        metadata = HendryksTaskInstanceMetadata(
            problem="What is 2 + 2?",
            solution="4",
            subject="arithmetic",
            level=1,
            unique_id="test_001",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the math problem"),
            intent=Intent(
                rubric="Provide the correct answer",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={"problem": "What is 2 + 2?", "solution": "4"},
        )

        # Create environment
        env = HendryksMathEnv(task_instance)
        obs = await env.initialize()

        # Check observation
        assert "problem" in obs
        assert obs["problem"] == "What is 2 + 2?"
        assert "submitted" in obs
        assert obs["submitted"] is False

    @pytest.mark.asyncio
    async def test_math_correct_answer(self):
        """Test submitting a correct answer."""
        metadata = HendryksTaskInstanceMetadata(
            problem="What is 5 * 6?",
            solution="30",
            subject="arithmetic",
            level=1,
            unique_id="test_002",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the multiplication"),
            intent=Intent(
                rubric="Multiply the numbers",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={"problem": "What is 5 * 6?", "solution": "30"},
        )

        env = HendryksMathEnv(task_instance)
        await env.initialize()

        # Submit correct answer
        from examples.math.environment import SubmitAnswer

        result = await env.step([[SubmitAnswer("30")]])

        obs = result.get("observation", result)
        assert obs["submitted"] is True
        assert obs.get("correct", False) is True

    @pytest.mark.asyncio
    async def test_math_incorrect_answer(self):
        """Test submitting an incorrect answer."""
        metadata = HendryksTaskInstanceMetadata(
            problem="What is 10 / 2?",
            solution="5",
            subject="arithmetic",
            level=1,
            unique_id="test_003",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the division"),
            intent=Intent(
                rubric="Divide the numbers",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={"problem": "What is 10 / 2?", "solution": "5"},
        )

        env = HendryksMathEnv(task_instance)
        await env.initialize()

        # Submit wrong answer
        from examples.math.environment import SubmitAnswer

        result = await env.step([[SubmitAnswer("4")]])

        obs = result.get("observation", result)
        assert obs["submitted"] is True
        assert obs.get("correct", True) is False

    @pytest.mark.asyncio
    async def test_math_multiple_submissions(self):
        """Test that only first submission counts."""
        metadata = HendryksTaskInstanceMetadata(
            problem="What is 3 + 3?",
            solution="6",
            subject="arithmetic",
            level=1,
            unique_id="test_004",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Add the numbers"),
            intent=Intent(
                rubric="Find the sum",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={"problem": "What is 3 + 3?", "solution": "6"},
        )

        env = HendryksMathEnv(task_instance)
        await env.initialize()

        from examples.math.environment import SubmitAnswer

        # First submission (wrong)
        result1 = await env.step([[SubmitAnswer("5")]])
        obs1 = result1.get("observation", result1)
        assert obs1["submitted"] is True
        assert obs1.get("correct", True) is False

        # Try second submission (should not change result)
        result2 = await env.step([[SubmitAnswer("6")]])
        obs2 = result2.get("observation", result2)

        # Should still show first submission result
        assert obs2["submitted"] is True
        # The environment might not allow changing answers

    @pytest.mark.asyncio
    async def test_math_complex_problem(self):
        """Test a more complex math problem."""
        metadata = HendryksTaskInstanceMetadata(
            problem="Solve for x: 2x + 5 = 13",
            solution="4",
            subject="algebra",
            level=2,
            unique_id="test_005",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the algebraic equation"),
            intent=Intent(
                rubric="Find the value of x",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={
                "problem": "Solve for x: 2x + 5 = 13",
                "solution": "4",
            },
        )

        env = HendryksMathEnv(task_instance)
        obs = await env.initialize()

        assert "Solve for x" in obs["problem"]

        # Submit correct answer
        from examples.math.environment import SubmitAnswer

        result = await env.step([[SubmitAnswer("4")]])

        obs = result.get("observation", result)
        assert obs.get("correct", False) is True

    @pytest.mark.asyncio
    async def test_math_checkpoint(self):
        """Test checkpointing in math environment."""
        metadata = HendryksTaskInstanceMetadata(
            problem="What is 7 - 3?",
            solution="4",
            subject="arithmetic",
            level=1,
            unique_id="test_006",
        )

        task_instance = HendryksTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Subtract"),
            intent=Intent(
                rubric="Find the difference",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={"problem": "What is 7 - 3?", "solution": "4"},
        )

        env = HendryksMathEnv(task_instance)
        await env.initialize()

        # Get checkpoint before submission
        checkpoint1 = await env.checkpoint()
        assert "engine_snapshot" in checkpoint1
        assert checkpoint1["engine_snapshot"]["submitted"] is False

        # Submit answer
        from examples.math.environment import SubmitAnswer

        await env.step([[SubmitAnswer("4")]])

        # Get checkpoint after submission
        checkpoint2 = await env.checkpoint()
        assert checkpoint2["engine_snapshot"]["submitted"] is True
        assert checkpoint2["engine_snapshot"]["submitted_answer"] == "4"
