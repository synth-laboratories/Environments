import asyncio
from httpx import AsyncClient

async def test_draw_scenario():
    """Test a specific scenario that leads to a draw"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # Initialize
        resp = await client.post(
            "/env/TicTacToe/initialize",
            json={"initial_state": {}, "config": {}}
        )
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        
        # Force a draw scenario with these moves:
        # X: A1, A3, B2, C1, C2
        # O: A2, B1, B3, C3
        draw_moves = [
            "A1",  # X
            "A2",  # O
            "A3",  # X
            "B1",  # O
            "B2",  # X
            "B3",  # O
            "C1",  # X
            "C3",  # O
            "C2"   # X (final move, should be draw)
        ]
        
        print("🎯 Testing draw scenario with specific moves...")
        current_player = "X"
        
        for i, move in enumerate(draw_moves):
            step_resp = await client.post(
                "/env/TicTacToe/step",
                json={
                    "env_id": instance_id,
                    "request_id": f"draw_step_{i}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": move}}]
                    }
                }
            )
            assert step_resp.status_code == 200
            
            step_data = step_resp.json()
            obs = step_data.get("observation", {})
            
            print(f"📋 Move {i+1}: {current_player} plays {move}")
            print(f"   Board:\n{obs.get('board_text', 'N/A')}")
            print(f"   Winner: {obs.get('winner', 'N/A')}")
            print(f"   Terminated: {obs.get('terminated', False)}")
            print()
            
            if obs.get("terminated"):
                final_winner = obs.get("winner")
                if final_winner == "draw":
                    print("✅ Successfully created a draw!")
                    print(f"📋 Final board:\n{obs.get('board_text', 'N/A')}")
                elif final_winner in ["X", "O"]:
                    print(f"🏆 {final_winner} won the game!")
                    print(f"📋 Final board:\n{obs.get('board_text', 'N/A')}")
                else:
                    print(f"🤔 Game ended with winner: {final_winner}")
                break
                
            # Switch player for next move
            current_player = "O" if current_player == "X" else "X"
        
        # Cleanup
        await client.post("/env/TicTacToe/terminate", json={"env_id": instance_id})

if __name__ == "__main__":
    asyncio.run(test_draw_scenario())
