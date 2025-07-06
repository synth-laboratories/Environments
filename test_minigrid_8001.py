import asyncio
from httpx import AsyncClient

async def test_minigrid_with_default_task():
    """Test MiniGrid with the default task"""
    async with AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        # 1) Health check
        health = await client.get("/health")
        assert health.status_code == 200
        supported = health.json()["supported_environments"]
        print(f"✅ Supported environments: {supported}")
        assert "MiniGrid" in supported
        
        # 2) Try to use the default MiniGrid task from taskset
        try:
            from synth_env.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK
            task_data = await DEFAULT_MINIGRID_TASK.serialize()
            
            resp = await client.post(
                "/env/MiniGrid/initialize", 
                json={"initial_state": task_data, "config": {}}
            )
            print(f"Initialize with default task status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"Error response: {resp.text}")
                return
                
            init_data = resp.json()
            instance_id = init_data["env_id"]
            initial_obs = init_data["observation"]
            print(f"✅ Created MiniGrid instance: {instance_id}")
            
            # Print observation structure
            print(f"🎮 Initial observation keys: {list(initial_obs.keys())}")
            
            # Print some key observation data
            for key in ["mission", "step_count", "agent_pos", "agent_dir"]:
                if key in initial_obs:
                    print(f"   {key}: {initial_obs[key]}")
            
            # Check if we have a grid representation
            if "grid_text" in initial_obs:
                print(f"📋 Grid:\n{initial_obs['grid_text']}")
            
            # Try a simple forward action
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
                print(f"   New position: {obs.get('agent_pos', 'unknown')}")
                print(f"   Reward: {obs.get('reward_last', 0)}")
                print(f"   Steps: {obs.get('step_count', 0)}")
                print(f"   Terminated: {obs.get('terminated', False)}")
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
    asyncio.run(test_minigrid_with_default_task())
