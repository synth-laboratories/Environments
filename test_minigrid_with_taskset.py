import asyncio
from httpx import AsyncClient

async def test_minigrid_with_taskset():
    """Test MiniGrid with proper task configuration"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Check if we can get a MiniGrid taskset
        try:
            # Try to create a task instance with MiniGrid configuration
            task_data = {
                "env_name": "MiniGrid-Empty-5x5-v0",
                "seed": 42,
                "max_steps": 100
            }
            
            resp = await client.post(
                "/env/MiniGrid/initialize",
                json={"initial_state": task_data, "config": {}}
            )
            print(f"Initialize with task config status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"Error response: {resp.text}")
                return
                
            init_data = resp.json()
            instance_id = init_data["env_id"]
            initial_obs = init_data["observation"]
            print(f"✅ Created MiniGrid instance: {instance_id}")
            
            # Print observation structure
            print(f"🎮 Initial observation keys: {list(initial_obs.keys())}")
            
            # Print mission and grid if available
            if "mission" in initial_obs:
                print(f"   Mission: {initial_obs['mission']}")
            if "grid_text" in initial_obs:
                print(f"   Grid:\n{initial_obs['grid_text']}")
            
            # Try a simple action - forward
            step_resp = await client.post(
                "/env/MiniGrid/step",
                json={
                    "env_id": instance_id,
                    "request_id": "test_forward",
                    "action": {
                        "tool_calls": [{"tool": "minigrid_act", "args": {"action": "forward"}}]
                    }
                }
            )
            
            if step_resp.status_code == 200:
                step_data = step_resp.json()
                obs = step_data.get("observation", {})
                print(f"✅ Forward action successful!")
                print(f"   Reward: {obs.get('reward_last', 0)}")
                print(f"   Steps: {obs.get('step_count', 0)}")
            else:
                print(f"❌ Forward action failed: {step_resp.text}")
            
            # Cleanup
            await client.post("/env/MiniGrid/terminate", json={"env_id": instance_id})
            print("✅ MiniGrid test completed successfully")
            
        except Exception as e:
            print(f"❌ Error testing MiniGrid: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_minigrid_with_taskset())
