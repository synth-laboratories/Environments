#!/usr/bin/env python3
"""
Test script to demonstrate both the current step method and the new gymnasium-compliant step method.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from horizons.environments.service.app import app
from starlette.testclient import TestClient

def test_both_step_methods():
    """Test both the current step method and gymnasium-compliant step method."""

    print("🚀 Testing both step methods...")
    print("=" * 60)

    # Create test client
    client = TestClient(app)

    # Initialize an environment
    print("1. Initializing Sokoban environment...")
    init_response = client.post('/env/Sokoban/initialize', json={
        'initial_state': {
            'dim_room': [4, 4],
            'room_fixed': [[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            'room_state': [[0, 0, 0, 0], [0, 1, 4, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
            'boxes_on_target': 0,
            'max_steps': 10,
            'num_boxes': 1
        }
    })

    if init_response.status_code != 200:
        print(f"❌ Failed to initialize environment: {init_response.text}")
        return

    env_id = init_response.json()['env_id']
    print(f"✅ Environment initialized with ID: {env_id}")

    # Test current step method
    print("\n2. Testing CURRENT step method...")
    step_response = client.post('/env/Sokoban/step', json={
        'env_id': env_id,
        'request_id': 'current-step-test',
        'action': {
            'tool_calls': [{'tool': 'interact', 'args': {'action': 1}}]  # DOWN action
        }
    })

    if step_response.status_code == 200:
        data = step_response.json()
        print(f"✅ Current step result: {list(data.keys())}")
        print(f"   - Observation type: {type(data['observation'])}")
        print(f"   - Reward: {data['reward']}")
        print(f"   - Done: {data['done']}")
        print(f"   - Info: {data['info']}")
    else:
        print(f"❌ Current step failed: {step_response.status_code} - {step_response.text}")

    # Get the environment instance to test gymnasium method
    print("\n3. Testing GYMNASIUM-COMPLIANT step method...")

    # Import the environment registry to get the environment
    from horizons.environments.environment.registry import get_environment_cls
    from horizons.environments.service.core_routes import instances

    try:
        # Get the environment instance
        env = instances[env_id]

        # Test gymnasium step method
        observation, reward, terminated, truncated, info = env.step_gymnasium(2)  # RIGHT action

        print("✅ Gymnasium step result:")
        print(f"   - Observation type: {type(observation)}")
        print(f"   - Reward: {reward}")
        print(f"   - Terminated: {terminated}")
        print(f"   - Truncated: {truncated}")
        print(f"   - Info: {info}")

        print("\n" + "=" * 60)
        print("🎉 Both step methods are working!")
        print("\n📋 Summary:")
        print("   - Current step: async def step(tool_calls) -> InternalObservation")
        print("   - Gymnasium step: def step_gymnasium(action) -> (obs, reward, terminated, truncated, info)")
        print("   - Both methods work alongside each other!")

    except Exception as e:
        print(f"❌ Gymnasium step test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_both_step_methods()
