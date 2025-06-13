"""Simple demo of Neural MMO Classic Environment.

This demo shows how to use the Neural MMO environment with both
vector and image observation modes.
"""

import asyncio
from typing import Dict, Any

# Try to import NMMO components
try:
    from synth_env.examples.nmmo_classic.environment import NeuralMMOEnvironment
    from synth_env.examples.nmmo_classic.taskset import (
        NeuralMMOTaskInstance,
        NeuralMMOTaskInstanceMetadata,
    )
    from synth_env.tasks.core import Impetus, Intent
    from synth_env.environment.tools import EnvToolCall

    NMMO_AVAILABLE = True
except ImportError as e:
    print(f"NMMO dependencies not available: {e}")
    print("Please install with: cd src/examples/nmmo_classic && pip install -e .")
    NMMO_AVAILABLE = False


def create_sample_task() -> "NeuralMMOTaskInstance":
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
        id="demo-task",
        impetus=Impetus(instructions="Survive and explore the Neural MMO world"),
        intent=Intent(rubric={"goal": "Maximize score and survival time"}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        config={"seed": 42, "map_size": 128, "tick_limit": 100},
    )


def create_simple_action() -> Dict[str, Any]:
    """Create a simple NMMO action."""
    return {
        "Move": {
            "Direction": 0,  # North
            "Target": [0, 0],  # No specific target
        }
    }


async def demo_vector_mode():
    """Demonstrate the environment in vector observation mode."""
    print("\n=== Vector Mode Demo ===")

    if not NMMO_AVAILABLE:
        print("NMMO not available, skipping demo")
        return

    # Create environment
    task = create_sample_task()
    env = NeuralMMOEnvironment(task, observation_mode="vector")

    try:
        # Initialize environment
        print("Initializing environment...")
        obs = await env.initialize()

        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Tick: {obs['tick']}")
        print(f"Position: {obs['position']}")
        print(f"Health: {obs['health']}")
        print(f"Stamina: {obs['stamina']}")
        print(f"Terrain shape: {obs['terrain'].shape}")

        # Take a few steps
        for step in range(3):
            print(f"\nStep {step + 1}:")

            action = create_simple_action()
            tool_call = EnvToolCall(name="nmmo_action", args={"action": action})

            obs = await env.step([[tool_call]])

            print(f"  Tick: {obs['tick']}")
            print(f"  Position: {obs['position']}")
            print(f"  Health: {obs['health']}")
            print(f"  Reward: {obs['reward_last']}")
            print(f"  Terminated: {obs['terminated']}")

            if obs["terminated"]:
                print("  Episode terminated!")
                break

        print("Vector mode demo completed successfully!")

    except Exception as e:
        print(f"Error in vector mode demo: {e}")
        import traceback

        traceback.print_exc()


async def demo_image_mode():
    """Demonstrate the environment in image observation mode."""
    print("\n=== Image Mode Demo ===")

    if not NMMO_AVAILABLE:
        print("NMMO not available, skipping demo")
        return

    # Create environment
    task = create_sample_task()
    env = NeuralMMOEnvironment(task, observation_mode="image")

    try:
        # Initialize environment
        print("Initializing environment...")
        obs = await env.initialize()

        print(f"Observation keys: {list(obs.keys())}")

        if "image" in obs:
            image = obs["image"]
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Image value range: {image.min()} - {image.max()}")
        else:
            print("No image in observation!")

        # Take one step to see image changes
        print("\nTaking one step...")
        action = create_simple_action()
        tool_call = EnvToolCall(name="nmmo_action", args={"action": action})

        obs = await env.step([[tool_call]])

        if "image" in obs:
            print(f"Updated image shape: {obs['image'].shape}")

        print("Image mode demo completed successfully!")

    except Exception as e:
        print(f"Error in image mode demo: {e}")
        import traceback

        traceback.print_exc()


async def demo_checkpointing():
    """Demonstrate environment checkpointing capabilities."""
    print("\n=== Checkpointing Demo ===")

    if not NMMO_AVAILABLE:
        print("NMMO not available, skipping demo")
        return

    # Create environment
    task = create_sample_task()
    env = NeuralMMOEnvironment(task, observation_mode="vector")

    try:
        # Initialize and take a step
        await env.initialize()

        action = create_simple_action()
        tool_call = EnvToolCall(name="nmmo_action", args={"action": action})
        obs_after_step = await env.step([[tool_call]])

        print(f"After step - Tick: {obs_after_step['tick']}")

        # Create checkpoint
        checkpoint_obs = await env.checkpoint()
        print(f"Checkpoint - Tick: {checkpoint_obs['tick']}")
        print(f"Checkpoint - Position: {checkpoint_obs['position']}")

        print("Checkpointing demo completed successfully!")

    except Exception as e:
        print(f"Error in checkpointing demo: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all demos."""
    print("Neural MMO Classic Environment Demo")
    print("===================================")

    if not NMMO_AVAILABLE:
        print("\nTo run this demo, install NMMO dependencies:")
        print("cd src/examples/nmmo_classic && pip install -e .")
        print("\nOr install manually:")
        print("pip install nmmo>=2.1.2 pufferlib>=2.0.6 pillow>=10.0.0")
        return

    await demo_vector_mode()
    await demo_image_mode()
    await demo_checkpointing()

    print("\nAll demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
