#!/usr/bin/env python3

import sys
import asyncio
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.examples.red.engine import PokemonRedEngine
from src.examples.red.taskset import INSTANCE as DEFAULT_TASK

async def test_updated_engine():
    """Test the updated Pokemon Red Engine with init state loading."""
    
    print("🔍 Creating Pokemon Red Engine with init state loading...")
    try:
        engine = PokemonRedEngine(DEFAULT_TASK, skip_rom_check=False)
        print("✅ Engine created successfully")

        print("\n📊 Extracting current state...")
        state = engine._extract_current_state()
        print(f"🔑 State keys: {list(state.keys())}")

        for key, value in state.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")

        has_valid_state = (
            isinstance(state.get("player_x"), int) and state["player_x"] > 0 or
            isinstance(state.get("player_y"), int) and state["player_y"] > 0 or
            isinstance(state.get("map_id"), int) and state["map_id"] > 0
        )

        print("\n🧪 Initial State Validation:")
        print("✅ Non-zero values detected! Init state loading worked!" if has_valid_state else "❌ Still getting zero values")

        print("\n🎮 Testing button press (RIGHT)...")
        old_x, old_y = state["player_x"], state["player_y"]

        priv_state, pub_state = await engine._step_engine({"button": "RIGHT", "frames": 1})
        
        print("\n📊 After Button Press:")
        print(f"  • Position: ({pub_state.player_x}, {pub_state.player_y})")
        print(f"  • Map ID: {pub_state.map_id}")
        print(f"  • Reward: {priv_state.reward_last_step}")

        if pub_state.player_x != old_x or pub_state.player_y != old_y:
            print("✅ Position changed! Movement working!")
        else:
            print("❌ No position change detected")

    except Exception as e:
        print(f"\n❌ Error encountered: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_updated_engine())
