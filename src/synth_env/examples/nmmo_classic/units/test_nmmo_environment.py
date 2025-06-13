"""Tests for Neural MMO Classic Environment."""

import pytest
import asyncio
import numpy as np

from synth_env.examples.nmmo_classic.environment import NeuralMMOEnvironment
from synth_env.examples.nmmo_classic.taskset import (
    NeuralMMOTaskInstance,
    NeuralMMOTaskInstanceMetadata,
)
from synth_env.tasks.core import Impetus, Intent
from synth_env.environment.tools import EnvToolCall


@pytest.fixture
def sample_task_instance():
    """Create a sample task instance for testing."""
    metadata = NeuralMMOTaskInstanceMetadata(
        difficulty="easy",
        seed=42,
        map_size=128,
        season="spring",
        resource_density=0.5,
        water_pct=0.1,
        hostiles_25=0,
        forage_tiles_25=20,
        spawn_biome="grass",
    )

    return NeuralMMOTaskInstance(
        id="test-nmmo-instance",
        impetus=Impetus(instructions="Test survival in Neural MMO"),
        intent=Intent(rubric={"goal": "Survive and score points"}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        config={"seed": 42, "map_size": 128, "tick_limit": 100},
    )


@pytest.mark.asyncio
class TestNeuralMMOEnvironment:
    """Test the Neural MMO Environment."""

    async def test_environment_initialization_vector_mode(self, sample_task_instance):
        """Test environment initialization in vector mode."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        obs = await env.initialize()

        # Check observation structure
        assert isinstance(obs, dict)
        assert "tick" in obs
        assert "position" in obs
        assert "health" in obs
        assert "stamina" in obs
        assert "inventory" in obs
        assert "terrain" in obs
        assert "visible_entities" in obs
        assert "reward_last" in obs
        assert "total_reward" in obs
        assert "terminated" in obs
        assert "truncated" in obs

        # Check that image is not included in vector mode
        assert "image" not in obs

    async def test_environment_initialization_image_mode(self, sample_task_instance):
        """Test environment initialization in image mode."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="image")

        # Initialize environment
        obs = await env.initialize()

        # Check observation structure includes image
        assert isinstance(obs, dict)
        assert "image" in obs

        # Check image properties
        image = obs["image"]
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Height, Width, Channels
        assert image.shape[2] == 3  # RGB
        assert image.dtype == np.uint8

    async def test_environment_step_vector_mode(self, sample_task_instance):
        """Test environment stepping in vector mode."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        await env.initialize()

        # Create a simple NMMO action (no-op or basic move)
        action = {
            "Move": {"Direction": 0, "Target": [0, 0]},  # Simple no-op action
        }

        # Create tool call
        tool_call = EnvToolCall(name="nmmo_action", args={"action": action})

        # Step environment
        obs = await env.step([[tool_call]])

        # Check observation structure
        assert isinstance(obs, dict)
        assert "tick" in obs
        assert obs["tick"] >= 0

        # Check that we got valid position data
        assert "position" in obs
        assert isinstance(obs["position"], (tuple, list))
        assert len(obs["position"]) == 2

    async def test_environment_step_image_mode(self, sample_task_instance):
        """Test environment stepping in image mode."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="image")

        # Initialize environment
        await env.initialize()

        # Create a simple NMMO action
        action = {
            "Move": {"Direction": 0, "Target": [0, 0]},
        }

        # Create tool call
        tool_call = EnvToolCall(name="nmmo_action", args={"action": action})

        # Step environment
        obs = await env.step([[tool_call]])

        # Check that image is included
        assert "image" in obs
        image = obs["image"]
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3

    async def test_environment_multiple_steps(self, sample_task_instance):
        """Test multiple environment steps."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        obs_init = await env.initialize()
        initial_tick = obs_init["tick"]

        # Take several steps
        for i in range(3):
            action = {
                "Move": {"Direction": i % 4, "Target": [0, 0]},  # Different directions
            }

            tool_call = EnvToolCall(name="nmmo_action", args={"action": action})

            obs = await env.step([[tool_call]])

            # Check that tick is advancing
            assert obs["tick"] >= initial_tick + i

            # Check that we're not terminated yet (should survive a few steps)
            if i < 2:  # Allow for early termination in later steps
                assert not obs["terminated"]

    async def test_environment_checkpoint(self, sample_task_instance):
        """Test environment checkpointing."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        await env.initialize()

        # Take a step
        action = {"Move": {"Direction": 0, "Target": [0, 0]}}
        tool_call = EnvToolCall(name="nmmo_action", args={"action": action})
        await env.step([[tool_call]])

        # Checkpoint
        checkpoint_obs = await env.checkpoint()

        # Check checkpoint structure
        assert isinstance(checkpoint_obs, dict)
        assert "tick" in checkpoint_obs
        assert "position" in checkpoint_obs

    async def test_invalid_tool_call(self, sample_task_instance):
        """Test handling of invalid tool calls."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        await env.initialize()

        # Create invalid tool call
        tool_call = EnvToolCall(name="invalid_tool", args={"some": "data"})

        # Should raise error for unknown tool
        with pytest.raises(ValueError, match="Unknown tool"):
            await env.step([[tool_call]])

    async def test_environment_termination(self, sample_task_instance):
        """Test that environment can terminate."""
        env = NeuralMMOEnvironment(sample_task_instance, observation_mode="vector")

        # Initialize environment
        await env.initialize()

        # Test termination call
        termination_obs = await env.terminate()

        assert isinstance(termination_obs, dict)
        assert "terminated" in termination_obs
        assert termination_obs["terminated"] is True


if __name__ == "__main__":
    # Simple manual test
    async def manual_test():
        metadata = NeuralMMOTaskInstanceMetadata(
            difficulty="easy",
            seed=42,
            map_size=128,
            season="spring",
            resource_density=0.5,
            water_pct=0.1,
            hostiles_25=0,
            forage_tiles_25=20,
            spawn_biome="grass",
        )

        task_instance = NeuralMMOTaskInstance(
            id="manual-test",
            impetus=Impetus(instructions="Manual test"),
            intent=Intent(rubric={"goal": "Test"}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
            config={"seed": 42, "map_size": 128, "tick_limit": 100},
        )

        # Test vector mode
        print("Testing vector mode...")
        env_vector = NeuralMMOEnvironment(task_instance, observation_mode="vector")
        obs = await env_vector.initialize()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(
            f"Tick: {obs['tick']}, Position: {obs['position']}, Health: {obs['health']}"
        )

        # Test image mode
        print("\nTesting image mode...")
        env_image = NeuralMMOEnvironment(task_instance, observation_mode="image")
        obs = await env_image.initialize()
        print(f"Image mode observation keys: {list(obs.keys())}")
        if "image" in obs:
            print(f"Image shape: {obs['image'].shape}")

        print("Manual test completed successfully!")

    asyncio.run(manual_test())
