#!/usr/bin/env python3
"""
Integration tests for GPT-5-nano playing each environment to completion.
Tests all three environments: Crafter, MiniGrid, and Sokoban.
Each test runs for up to 10 steps and verifies the environment service works correctly.
"""

import asyncio
import os
import sys
import uuid
from typing import Any, Dict, List

import pytest
from httpx import AsyncClient
from pydantic import BaseModel, Field

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool

# --- Service Configuration ---
SERVICE_BASE_URL = "http://localhost:8901"
MODEL_NAME = "gpt-5-nano"
MAX_STEPS = 10
TIMEOUT_SECONDS = 30

# Environment configurations
ENV_CONFIGS = {
    "crafter": {
        "service_name": "CrafterClassic",
        "difficulty": "easy",
        "expected_actions": [0, 1, 2, 3, 4, 5]  # Integer actions: move_left, move_right, move_up, move_down, do, sleep
    },
    "minigrid": {
        "service_name": "MiniGrid",
        "difficulty": "easy",
        "expected_actions": ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    },
    "sokoban": {
        "service_name": "Sokoban",
        "difficulty": "ultra_easy",
        "expected_actions": ["push_up", "push_down", "push_left", "push_right", "move_up", "move_down", "move_left", "move_right"]
    }
}


# --- Tool Definitions ---
class GameActionArgs(BaseModel):
    """Arguments for game actions."""
    action: str = Field(..., description="The action to perform")


class GameActionTool(BaseTool):
    """Tool for executing game actions."""

    def __init__(self, available_actions: List[str]):
        super().__init__(
            name="interact",
            description="Execute an action in the game environment",
            args_schema=GameActionArgs,
            available_actions=available_actions
        )
        self.available_actions = available_actions

    def run(self, action: str) -> Dict[str, Any]:
        """Execute the action and return result."""
        if action not in self.available_actions:
            return {
                "error": f"Invalid action: {action}. Available actions: {self.available_actions}",
                "success": False
            }

        return {
            "action": action,
            "success": True,
            "message": f"Executed action: {action}"
        }


async def create_environment_session(client: AsyncClient, service_env_name: str, difficulty: str) -> Dict[str, Any]:
    """Create a new environment session and return the full response."""
    # Use the /env/{env_name}/initialize endpoint instead of the deprecated create endpoint
    response = await client.post(
        f"{SERVICE_BASE_URL}/env/{service_env_name}/initialize",
        json={"session_id": str(uuid.uuid4()), "difficulty": difficulty},
        timeout=TIMEOUT_SECONDS
    )
    assert response.status_code == 200, f"Failed to create environment: {response.text}"
    result = response.json()
    # The service returns 'env_id' instead of 'session_id'
    session_id = result.get("session_id") or result.get("env_id")
    assert session_id, f"No session_id or env_id in response: {result}"
    # Add session_id to result for consistency
    result["session_id"] = session_id
    return result


async def initialize_environment(client: AsyncClient, session_id: str) -> Dict[str, Any]:
    """Initialize the environment and get the first observation."""
    response = await client.post(
        f"{SERVICE_BASE_URL}/env/initialize",
        json={"session_id": session_id},
        timeout=TIMEOUT_SECONDS
    )
    assert response.status_code == 200, f"Failed to initialize environment: {response.text}"
    result = response.json()
    assert "public_observation" in result, f"No public_observation in response: {result}"
    return result


async def step_environment(client: AsyncClient, service_env_name: str, session_id: str, action: str, request_id: str = None) -> Dict[str, Any]:
    """Take a step in the environment."""
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Different environments have different action formats
    if service_env_name == "Sokoban":
        # Sokoban expects action as integer in a specific format
        # Convert string action to integer (e.g., "push_up" -> 0, "push_down" -> 1, etc.)
        action_mapping = {
            "push_up": 0, "push_down": 1, "push_left": 2, "push_right": 3,
            "move_up": 4, "move_down": 5, "move_left": 6, "move_right": 7
        }
        action_int = action_mapping.get(action, 0)  # Default to push_up if not found

        # Use standard tool_calls format for Sokoban
        payload = {
            "env_id": session_id,
            "action": {
                "tool_calls": [{
                    "tool": "interact",
                    "args": {"action": action_int}
                }]
            },
            "request_id": request_id
        }
    else:
        # Crafter and MiniGrid use tool_calls format with string actions
        tool_name = "interact"  # Default for Crafter
        if service_env_name == "MiniGrid":
            tool_name = "minigrid_act"

        payload = {
            "env_id": session_id,
            "action": {"tool_calls": [{"tool": tool_name, "args": {"action": action}}]},
            "request_id": request_id
        }

    response = await client.post(
        f"{SERVICE_BASE_URL}/env/{service_env_name}/step",
        json=payload,
        timeout=TIMEOUT_SECONDS
    )
    assert response.status_code == 200, f"Failed to step environment: {response.text}"
    result = response.json()
    # The service returns observation directly, not under public_observation
    assert "observation" in result, f"No observation in response: {result}"
    return result


async def terminate_environment(client: AsyncClient, service_env_name: str, session_id: str) -> Dict[str, Any]:
    """Terminate the environment session."""
    response = await client.post(
        f"{SERVICE_BASE_URL}/env/{service_env_name}/terminate",
        json={"env_id": session_id, "action": {"tool_calls": []}},
        timeout=TIMEOUT_SECONDS
    )
    assert response.status_code == 200, f"Failed to terminate environment: {response.text}"
    result = response.json()
    # The service might return observation directly or under different keys
    assert "observation" in result or "public_observation" in result, f"No observation in response: {result}"
    return result


async def run_gpt5_nano_episode(
    client: AsyncClient,
    env_name: str,
    difficulty: str,
    max_steps: int = MAX_STEPS
) -> Dict[str, Any]:
    """Run a complete episode with GPT-5-nano agent."""
    print(f"🎮 Starting {env_name} episode with GPT-5-nano (max {max_steps} steps)")

    # Get service environment name from config
    config = ENV_CONFIGS[env_name]
    service_env_name = config["service_name"]

    # Create environment (this also initializes it)
    init_result = await create_environment_session(client, service_env_name, difficulty)
    print(f"📝 Created session: {init_result['session_id']}")
    print("🚀 Environment initialized successfully")

    session_id = init_result['session_id']
    # The service returns observation directly
    current_observation = init_result.get('observation', {})

    # Create agent
    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=1.0)
    config = ENV_CONFIGS[env_name]
    agent = SimpleReActAgent(
        llm=llm,
        available_actions=config["expected_actions"],
        max_turns=max_steps,
        verbose=True
    )

    # Run episode
    episode_data = {
        "session_id": session_id,
        "env_name": env_name,
        "difficulty": difficulty,
        "model": MODEL_NAME,
        "steps": [],
        "total_reward": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "step_count": 0
    }

    current_observation = init_result.get('public_observation', init_result.get('observation', {}))
    episode_data["steps"].append({
        "step": 0,
        "observation": current_observation,
        "action": "initialize",
        "reward": 0.0
    })

    for step in range(1, max_steps + 1):
        print(f"🔄 Step {step}/{max_steps}")

        # Agent chooses action
        action = await agent.act(current_observation)

        # Execute action
        step_result = await step_environment(client, service_env_name, session_id, action)

        # Update episode data
        obs = step_result.get("observation", {})
        info = step_result.get("info", {})
        reward = step_result.get("reward") or 0.0
        terminated = info.get("terminated", False)
        truncated = info.get("truncated", False)

        episode_data["steps"].append({
            "step": step,
            "observation": obs,
            "action": action,
            "reward": reward
        })

        episode_data["total_reward"] += reward
        episode_data["step_count"] = step
        episode_data["terminated"] = terminated
        episode_data["truncated"] = truncated

        # Check if episode is done
        if terminated or truncated:
            episode_data["success"] = not truncated  # Assume success if not truncated
            print(f"🏁 Episode ended at step {step}: {'Success' if episode_data['success'] else 'Failed'}")
            break

        current_observation = obs

    # Terminate environment
    try:
        terminate_result = await terminate_environment(client, service_env_name, session_id)
        final_obs = terminate_result.get("observation", {})
        final_reward = terminate_result.get("reward", episode_data["total_reward"])
        episode_data["total_reward"] = final_reward
        print(f"💰 Final total reward: {episode_data['total_reward']}")
    except Exception as e:
        print(f"⚠️  Warning: Could not terminate environment cleanly: {e}")

    return episode_data


class SimpleReActAgent:
    """Simple ReAct agent for testing environments."""

    def __init__(self, llm, available_actions: List[str], max_turns: int = 10, verbose: bool = False):
        self.llm = llm
        self.available_actions = available_actions
        self.max_turns = max_turns
        self.verbose = verbose
        self.turn_count = 0

    async def act(self, observation: Dict[str, Any]) -> str:
        """Choose an action based on the current observation."""
        self.turn_count += 1

        # Format observation for the model
        obs_text = self._format_observation(observation)

        # Create prompt
        prompt = f"""You are playing a game. Here is the current observation:
{obs_text}

Available actions: {', '.join(str(action) for action in self.available_actions)}

Choose the next action to take. Respond with only the action number, nothing else."""

        if self.verbose:
            print(f"🤖 Agent prompt: {prompt}")

        # Get response from model
        try:
            response = await self.llm.respond_async(system_message="You are a helpful AI assistant.", user_message=prompt)
            if self.verbose:
                print(f"🤖 Model response object: {response}")
                print(f"🤖 Raw response: {response.raw_response}")
            action_text = str(response.raw_response).strip().lower()
            if self.verbose:
                print(f"🤖 Extracted action text: '{action_text}'")
            action = action_text

            # Clean up action name
            action = action.replace(" ", "_")

            # Validate action
            if action not in self.available_actions:
                # Try to find closest match
                for available_action in self.available_actions:
                    if available_action.lower() in action or action in available_action.lower():
                        action = available_action
                        break
                else:
                    # Default to first action if no match found
                    action = self.available_actions[0]
                    if self.verbose:
                        print(f"⚠️  Invalid action '{action}', using default: {action}")

            if self.verbose:
                print(f"🎯 Agent chose action: {action}")

            return action

        except Exception as e:
            if self.verbose:
                print(f"❌ Error getting action from model: {e}")
            # Return default action on error
            return self.available_actions[0]

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """Format the observation for the model."""
        if isinstance(observation, dict):
            # Extract key information
            formatted_parts = []

            if "mission" in observation:
                formatted_parts.append(f"Mission: {observation['mission']}")

            if "image" in observation:
                formatted_parts.append("Image: Available")

            if "grid" in observation:
                formatted_parts.append(f"Grid: {observation['grid']}")

            if "player_stats" in observation:
                stats = observation["player_stats"]
                formatted_parts.append(f"Player Stats: Health={stats.get('health', 0)}, Food={stats.get('food', 0)}, Drink={stats.get('drink', 0)}")

            if "terminated" in observation:
                formatted_parts.append(f"Terminated: {observation['terminated']}")

            if "truncated" in observation:
                formatted_parts.append(f"Truncated: {observation['truncated']}")

            if not formatted_parts:
                formatted_parts.append(str(observation))

            return "\n".join(formatted_parts)
        else:
            return str(observation)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gpt5_nano_crafter():
    """Test GPT-5-nano playing Crafter environment for up to 10 steps."""
    async with AsyncClient() as client:
        episode_data = await run_gpt5_nano_episode(client, "crafter", "easy", MAX_STEPS)

        # Verify episode ran successfully
        assert episode_data["step_count"] > 0, "Episode should have at least 1 step"
        assert episode_data["step_count"] <= MAX_STEPS, f"Episode should not exceed {MAX_STEPS} steps"
        assert len(episode_data["steps"]) == episode_data["step_count"] + 1, "Steps data should match step count"

        # Verify environment is working
        assert "session_id" in episode_data
        assert episode_data["env_name"] == "crafter"
        assert episode_data["model"] == MODEL_NAME

        print("✅ Crafter test completed successfully")
        print(f"   Steps: {episode_data['step_count']}")
        print(f"   Reward: {episode_data['total_reward']:.3f}")
        print(f"   Success: {episode_data['success']}")
        print(f"   Terminated: {episode_data['terminated']}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gpt5_nano_minigrid():
    """Test GPT-5-nano playing MiniGrid environment for up to 10 steps."""
    async with AsyncClient() as client:
        episode_data = await run_gpt5_nano_episode(client, "minigrid", "easy", MAX_STEPS)

        # Verify episode ran successfully
        assert episode_data["step_count"] > 0, "Episode should have at least 1 step"
        assert episode_data["step_count"] <= MAX_STEPS, f"Episode should not exceed {MAX_STEPS} steps"
        assert len(episode_data["steps"]) == episode_data["step_count"] + 1, "Steps data should match step count"

        # Verify environment is working
        assert "session_id" in episode_data
        assert episode_data["env_name"] == "minigrid"
        assert episode_data["model"] == MODEL_NAME

        print("✅ MiniGrid test completed successfully")
        print(f"   Steps: {episode_data['step_count']}")
        print(f"   Reward: {episode_data['total_reward']:.3f}")
        print(f"   Success: {episode_data['success']}")
        print(f"   Terminated: {episode_data['terminated']}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gpt5_nano_sokoban():
    """Test GPT-5-nano playing Sokoban environment for up to 10 steps."""
    async with AsyncClient() as client:
        episode_data = await run_gpt5_nano_episode(client, "sokoban", "ultra_easy", MAX_STEPS)

        # Verify episode ran successfully
        assert episode_data["step_count"] > 0, "Episode should have at least 1 step"
        assert episode_data["step_count"] <= MAX_STEPS, f"Episode should not exceed {MAX_STEPS} steps"
        assert len(episode_data["steps"]) == episode_data["step_count"] + 1, "Steps data should match step count"

        # Verify environment is working
        assert "session_id" in episode_data
        assert episode_data["env_name"] == "sokoban"
        assert episode_data["model"] == MODEL_NAME

        print("✅ Sokoban test completed successfully")
        print(f"   Steps: {episode_data['step_count']}")
        print(f"   Reward: {episode_data['total_reward']:.3f}")
        print(f"   Success: {episode_data['success']}")
        print(f"   Terminated: {episode_data['terminated']}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_environment_service_health():
    """Test that the environment service is running and healthy."""
    async with AsyncClient() as _client:
        # Test health endpoint
        response = await _client.get(f"{SERVICE_BASE_URL}/health", timeout=TIMEOUT_SECONDS)
        assert response.status_code == 200, f"Health check failed: {response.text}"

        result = response.json()
        assert "status" in result, f"No status in health response: {result}"
        assert result["status"] == "ok", f"Service not healthy: {result['status']}"

        print(f"✅ Service health check passed. Status: {result['status']}")


if __name__ == "__main__":
    # Run the tests directly
    print("🧪 Running GPT-5-nano environment integration tests...")
    print(f"Service URL: {SERVICE_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max steps per episode: {MAX_STEPS}")
    print("=" * 60)

    async def run_all_tests():
        async with AsyncClient() as _client:
            # First check service health
            print("🏥 Checking service health...")
            await test_environment_service_health()

            print("\n" + "=" * 60)
            print("🎮 Testing Crafter environment...")
            await test_gpt5_nano_crafter()

            print("\n" + "=" * 60)
            print("🎮 Testing MiniGrid environment...")
            await test_gpt5_nano_minigrid()

            print("\n" + "=" * 60)
            print("🎮 Testing Sokoban environment...")
            await test_gpt5_nano_sokoban()

            print("\n" + "=" * 60)
            print("🎉 All tests completed successfully!")

    asyncio.run(run_all_tests())

    print("\n" + "=" * 80)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print("✅ Successfully created integration tests for GPT-5-nano environments")
    print("✅ Tests can be run individually or all together")
    print("✅ Each test runs up to 10 steps per environment")
    print("✅ Tests verify environment service functionality")
    print("✅ Tests demonstrate agent-environment interaction")
    print("\nTo run specific environment tests:")
    print("  python -m pytest tests/integration/test_gpt5_nano_environments.py::test_gpt5_nano_crafter -v -s")
    print("  python -m pytest tests/integration/test_gpt5_nano_environments.py::test_gpt5_nano_minigrid -v -s")
    print("  python -m pytest tests/integration/test_gpt5_nano_environments.py::test_gpt5_nano_sokoban -v -s")
    print("\nTo run all environment tests:")
    print("  python -m pytest tests/integration/test_gpt5_nano_environments.py -v -s")
    print("=" * 80)
