import asyncio
from httpx import AsyncClient

async def test_minigrid_basic():
    """Test MiniGrid basic functionality through the service"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Health check
        health = await client.get("/health")
        assert health.status_code == 200
        supported = health.json()["supported_environments"]
        assert "MiniGrid" in supported
        print("✅ MiniGrid found in health check")
        
        # 2) Initialize a MiniGrid instance
        resp = await client.post(
            "/env/MiniGrid/initialize",
            json={"initial_state": {}, "config": {}}
        )
        print(f"Initialize response status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"Initialize response: {resp.text}")
            return
            
        init_data = resp.json()
        instance_id = init_data["env_id"]
        initial_obs = init_data["observation"]
        print(f"✅ Created MiniGrid instance: {instance_id}")
        
        # 3) Check observation structure
        print(f"🎮 Initial observation has {len(initial_obs)} keys:")
        for key in initial_obs.keys():
            print(f"   - {key}")
            
        # 4) Try a few actions
        # MiniGrid actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done
        test_actions = [2, 1, 2, 2]  # forward, right, forward, forward
        
        for i, action in enumerate(test_actions):
            step_resp = await client.post(
                "/env/MiniGrid/step",
                json={
                    "env_id": instance_id,
                    "request_id": f"test_step_{i}",
                    "action": {
                        "tool_calls": [{"tool": "minigrid_act", "args": {"action": action}}]
                    }
                }
            )
            
            if step_resp.status_code != 200:
                print(f"Step {i+1} failed: {step_resp.text}")
                continue
                
            step_data = step_resp.json()
            obs = step_data.get("observation", {})
            
            print(f"📋 After action {i+1} (action {action}):")
            print(f"   Keys: {list(obs.keys())}")
            print(f"   Terminated: {obs.get('terminated', False)}")
            print(f"   Reward: {obs.get('reward_last', 0)}")
            
            if obs.get("terminated"):
                print(f"🏁 Episode ended after {i+1} steps!")
                break
                
        # 5) Terminate the instance
        term_resp = await client.post(
            "/env/MiniGrid/terminate",
            json={"env_id": instance_id}
        )
        assert term_resp.status_code == 200
        print("✅ MiniGrid instance terminated successfully")

if __name__ == "__main__":
    asyncio.run(test_minigrid_basic())
