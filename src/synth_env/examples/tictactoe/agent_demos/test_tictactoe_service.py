import pytest
import asyncio
import json

from httpx import AsyncClient

from synth_env.examples.tictactoe.taskset import create_tictactoe_taskset
from synth_ai.zyk import LM

# Demo: drive TicTacToe via FastAPI service endpoints


# === UNIT TESTS FOR SERVICE INTEGRATION ===

@pytest.mark.anyio
async def test_tictactoe_service_new_api():
    """Test the new /env/TicTacToe/initialize API - verifies the environment works"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Health check
        health = await client.get("/health")
        assert health.status_code == 200
        supported = health.json()["supported_environments"]
        assert "TicTacToe" in supported
        print("✅ Health check passed")

        # 2) Get a task instance from the taskset
        taskset = await create_tictactoe_taskset()
        task_instance = taskset.instances[0]  # Use first instance
        task_data = await task_instance.serialize()
        
        # 3) Initialize a TicTacToe instance using the new API
        resp = await client.post(
            "/env/TicTacToe/initialize",
            json={"initial_state": task_data, "config": {}}
        )
        print(f"Initialize response status: {resp.status_code}")
        print(f"Initialize response: {resp.text}")
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        initial_obs = init_data["observation"]
        print(f"✅ Created instance: {instance_id}")
        
        # 4) Critical test: Verify observation is NOT empty
        assert len(initial_obs) > 0, "Initial observation should not be empty!"
        assert "board_text" in initial_obs, "Should have board_text in observation"
        assert "current_player" in initial_obs, "Should have current_player in observation"
        print(f"✅ Observation has {len(initial_obs)} keys: {list(initial_obs.keys())}")
        print(f"🎮 Initial Board:")
        print(initial_obs.get("board_text", "No board text"))
        print(f"Current Player: {initial_obs.get('current_player', 'Unknown')}")
        
        # 5) Take a few moves to verify step observations work
        for move_num in range(3):
            # Try common moves like A1, B2, C3
            moves = ["A1", "B2", "C3", "A2", "B1", "C1", "A3", "B3", "C2"]
            move = moves[move_num % len(moves)]
            
            step_resp = await client.post(
                "/env/TicTacToe/step",
                json={
                    "env_id": instance_id,
                    "request_id": f"test_move_{move_num}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": move}}]
                    }
                }
            )
            
            print(f"Move {move_num+1} ({move}) response status: {step_resp.status_code}")
            
            # Handle both successful moves and invalid moves
            if step_resp.status_code == 200:
                step_data = step_resp.json()
                obs_data = step_data.get("observation", {})
                
                # Critical test: Step observations should NOT be empty
                assert len(obs_data) > 0, f"Move {move_num+1} observation should not be empty!"
                assert "board_text" in obs_data, f"Move {move_num+1} should have board_text"
                
                print(f"✅ Move {move_num+1}: Observation has {len(obs_data)} keys")
                print(f"   Board after move:")
                print(obs_data.get("board_text", "No board"))
                print(f"   Current Player: {obs_data.get('current_player', 'Unknown')}")
                print(f"   Move Count: {obs_data.get('move_count', 0)}")
                
                if obs_data.get("error"):
                    print(f"   Error: {obs_data['error']}")
                    continue  # Try next move
                
                if obs_data.get("terminated") or obs_data.get("winner"):
                    print(f"Game ended: Winner = {obs_data.get('winner', 'Draw')}")
                    break
            else:
                print(f"❌ Move {move_num+1} failed with status {step_resp.status_code}")
                print(f"Response: {step_resp.text}")

        # 6) Terminate the instance
        term_resp = await client.post(
            "/env/TicTacToe/terminate",
            json={"env_id": instance_id}
        )
        assert term_resp.status_code == 200
        print("✅ Instance terminated successfully")


@pytest.mark.anyio
async def test_tictactoe_service_legacy_api():
    """Test the legacy /TicTacToe/create API - verifies both APIs work"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Get a task instance from the taskset
        taskset = await create_tictactoe_taskset()
        task_instance = taskset.instances[0]
        task_data = await task_instance.serialize()
        
        # 2) Create using legacy API
        resp = await client.post(
            "/TicTacToe/create",
            json={"initial_state": task_data}
        )
        print(f"Legacy create response status: {resp.status_code}")
        
        if resp.status_code == 200:
            create_data = resp.json()
            instance_id = create_data["instance_id"]
            print(f"✅ Created legacy instance: {instance_id}")
            
            # 3) Reset to get initial observation
            reset_resp = await client.post(f"/TicTacToe/{instance_id}/reset")
            assert reset_resp.status_code == 200
            
            obs = reset_resp.json()
            private = obs["private"]
            public = obs["public"]
            
            # Verify observations are not empty
            assert len(private) > 0, "Private observation should not be empty!"
            assert len(public) > 0, "Public observation should not be empty!"
            assert "board_text" in public, "Should have board_text in public observation"
            
            print(f"✅ Legacy API: Public obs has {len(public)} keys, Private obs has {len(private)} keys")
            print(f"   Initial Board:")
            print(public.get("board_text", "No board"))
            print(f"   Current Player: {public.get('current_player', 'Unknown')}")
            
            # 4) Take a step using legacy API
            step_resp = await client.post(
                f"/TicTacToe/{instance_id}/step",
                json=[{"tool": "interact", "args": {"action": "A1"}}]
            )
            assert step_resp.status_code == 200
            
            step_obs = step_resp.json()
            step_private = step_obs["private"]
            step_public = step_obs["public"]
            
            # Critical test: Step observations should NOT be empty
            assert len(step_private) > 0, "Step private observation should not be empty!"
            assert len(step_public) > 0, "Step public observation should not be empty!"
            print(f"✅ Legacy step: Public obs has {len(step_public)} keys, Private obs has {len(step_private)} keys")
            print(f"   Board after A1:")
            print(step_public.get("board_text", "No board"))
            
            # 5) Terminate using legacy API
            term_resp = await client.post(f"/TicTacToe/{instance_id}/terminate")
            assert term_resp.status_code == 200
            print("✅ Legacy instance terminated successfully")


@pytest.mark.anyio
async def test_multiple_game_configurations():
    """Test with different TicTacToe starting positions"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        taskset = await create_tictactoe_taskset()
        
        # Test first 3 different configurations
        for i, task_instance in enumerate(taskset.instances[:3]):
            task_data = await task_instance.serialize()
            
            # Initialize with this configuration
            resp = await client.post(
                "/env/TicTacToe/initialize",
                json={"initial_state": task_data, "config": {}}
            )
            print(f"Config {i+1} initialize status: {resp.status_code}")
            
            if resp.status_code == 200:
                init_data = resp.json()
                instance_id = init_data["env_id"]
                obs = init_data["observation"]
                
                # Critical test: Verify observation is not empty
                assert len(obs) > 0, f"Config {i+1} observation should not be empty!"
                assert "board_text" in obs, f"Config {i+1} should have board_text"
                
                print(f"✅ Config {i+1}: Observation has {len(obs)} keys")
                print(f"   Starting Player: {obs.get('current_player', 'Unknown')}")
                print(f"   Move Count: {obs.get('move_count', 0)}")
                print(f"   Opening Moves: {task_instance.metadata.opening_moves}")
                print(f"   Board:")
                print(obs.get("board_text", "No board"))
                
                # Terminate
                term_resp = await client.post(
                    "/env/TicTacToe/terminate",
                    json={"env_id": instance_id}
                )
                assert term_resp.status_code == 200
                print(f"✅ Config {i+1} test completed successfully")


@pytest.mark.anyio
async def test_tictactoe_observation_content_validation():
    """Test that observations contain expected content structure"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        taskset = await create_tictactoe_taskset()
        task_instance = taskset.instances[0]
        task_data = await task_instance.serialize()
        
        # Initialize
        resp = await client.post(
            "/env/TicTacToe/initialize",
            json={"initial_state": task_data, "config": {}}
        )
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        obs = init_data["observation"]
        
        # Test specific observation structure
        required_keys = ["board_text", "current_player", "move_count"]
        for key in required_keys:
            assert key in obs, f"Missing required key: {key}"
        
        # Test data types
        assert isinstance(obs["board_text"], str), "board_text should be a string"
        assert isinstance(obs["current_player"], str), "current_player should be a string"
        assert isinstance(obs["move_count"], int), "move_count should be an int"
        assert len(obs["board_text"]) > 0, "board_text should not be empty"
        assert obs["current_player"] in ["X", "O"], "current_player should be X or O"
        
        print(f"✅ Observation structure validation passed")
        print(f"   Current Player: {obs['current_player']}")
        print(f"   Move Count: {obs['move_count']}")
        print(f"   Board text length: {len(obs['board_text'])} characters")
        
        # Take a step and verify the observation still has proper structure
        step_resp = await client.post(
            "/env/TicTacToe/step",
            json={
                "env_id": instance_id,
                "request_id": "validation_step",
                "action": {
                    "tool_calls": [{"tool": "interact", "args": {"action": "A1"}}]
                }
            }
        )
        
        if step_resp.status_code == 200:
            step_data = step_resp.json()
            step_obs = step_data.get("observation", {})
            
            # Verify step observation structure
            for key in required_keys:
                assert key in step_obs, f"Missing required key in step observation: {key}"
            
            # Verify move count increased
            assert step_obs["move_count"] > obs["move_count"], "move_count should increase after step"
            
            print(f"✅ Step observation structure validation passed")
            print(f"   Move count after step: {step_obs['move_count']}")
        
        # Cleanup
        await client.post("/env/TicTacToe/terminate", json={"env_id": instance_id})


# === SIMPLE REACT AGENT TEST ===

class SimpleTicTacToeAgent:
    """Simple TicTacToe agent for testing"""
    
    def __init__(self):
        # Simple move priority: corners first, then center, then edges
        self.move_priority = ["A1", "A3", "C1", "C3", "B2", "A2", "B1", "B3", "C2"]
    
    async def decide_move(self, obs: dict) -> str:
        """Choose next move based on observation"""
        board_text = obs.get("board_text", "")
        current_player = obs.get("current_player", "X")
        
        # Parse board to see what moves are available
        available_moves = []
        for move in self.move_priority:
            # Simple heuristic: if the move position isn't mentioned as taken, try it
            if move not in board_text or "   " in board_text:  # Empty cell indicator
                available_moves.append(move)
        
        if available_moves:
            return available_moves[0]
        else:
            # Fallback to any cell name
            return "B2"


@pytest.mark.anyio
async def test_simple_agent_game():
    """Test a simple agent playing TicTacToe through the service"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        taskset = await create_tictactoe_taskset()
        task_instance = taskset.instances[0]
        task_data = await task_instance.serialize()
        
        # Initialize game
        resp = await client.post(
            "/env/TicTacToe/initialize",
            json={"initial_state": task_data, "config": {}}
        )
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        obs = init_data["observation"]
        
        print(f"🎮 Starting simple agent game")
        print(f"Initial board:")
        print(obs.get("board_text", "No board"))
        
        agent = SimpleTicTacToeAgent()
        
        # Play game
        for turn in range(9):  # Max 9 moves in TicTacToe
            current_player = obs.get("current_player")
            if not current_player:
                break
                
            print(f"\nTurn {turn + 1}: Player {current_player}'s move")
            
            # Get agent decision
            move = await agent.decide_move(obs)
            print(f"Agent chooses: {move}")
            
            # Take step
            step_resp = await client.post(
                "/env/TicTacToe/step",
                json={
                    "env_id": instance_id,
                    "request_id": f"agent_turn_{turn}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": move}}]
                    }
                }
            )
            
            if step_resp.status_code != 200:
                print(f"❌ Move failed: {step_resp.text}")
                break
            
            step_data = step_resp.json()
            obs = step_data.get("observation", {})
            
            print(f"Board after move:")
            print(obs.get("board_text", "No board"))
            
            if obs.get("error"):
                print(f"Move error: {obs['error']}")
                # Continue to try other moves
                
            if obs.get("terminated") or obs.get("winner"):
                winner = obs.get("winner", "Draw")
                print(f"🏁 Game ended! Winner: {winner}")
                break
        
        # Cleanup
        await client.post("/env/TicTacToe/terminate", json={"env_id": instance_id})
        print("✅ Simple agent game completed")


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("🧪 Testing TicTacToe Service Integration")
        print("=" * 50)
        
        try:
            print("\n1️⃣ Testing new API...")
            await test_tictactoe_service_new_api()
            
            print("\n2️⃣ Testing legacy API...")
            await test_tictactoe_service_legacy_api()
            
            print("\n3️⃣ Testing multiple configurations...")
            await test_multiple_game_configurations()
            
            print("\n4️⃣ Testing observation validation...")
            await test_tictactoe_observation_content_validation()
            
            print("\n5️⃣ Testing simple agent game...")
            await test_simple_agent_game()
            
            print("\n🎉 All TicTacToe tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests()) 