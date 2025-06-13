#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.examples.red.engine import PokemonRedEngine
from src.examples.red.taskset import INSTANCE as DEFAULT_TASK


async def test_updated_engine():
    """Test the updated Pokemon Red Engine with init state loading"""

    print("🔍 Creating Pokemon Red Engine with init state loading...")
    try:
        engine = PokemonRedEngine(DEFAULT_TASK, skip_rom_check=False)
        print("✅ Engine created successfully")

        print("📊 Extracting current state...")
        state = engine._extract_current_state()

        print(f"🔑 Available state keys: {list(state.keys())}")
        print("📋 State contents:")
        for key, value in state.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")

        # Check if we have meaningful values now
        if state["player_x"] > 0 or state["player_y"] > 0 or state["map_id"] > 0:
            print("✅ Non-zero values detected! Init state loading worked!")
        else:
            print("❌ Still getting zero values")

        # Test a button press
        print("\n🎮 Testing button press...")
        old_x, old_y = state["player_x"], state["player_y"]

        # Test step engine with button press
        priv_state, pub_state = await engine._step_engine(
            {"button": "RIGHT", "frames": 1}
        )

        print("📊 After button press:")
        print(f"  Position: ({pub_state.player_x}, {pub_state.player_y})")
        print(f"  Map ID: {pub_state.map_id}")
        print(f"  Reward: {priv_state.reward_last_step}")

        if pub_state.player_x != old_x or pub_state.player_y != old_y:
            print("✅ Position changed! Movement working!")
        else:
            print("❌ No position change detected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_updated_engine())
