#!/usr/bin/env python3
"""Debug script to run a single MiniGrid trajectory and see what's happening."""

import asyncio
from src.synth_env.examples.minigrid.agent_demos.eval_framework import MiniGridEvalFramework

async def debug_single_trajectory():
    """Run a single trajectory with full debug output."""
    print("🔍 Debug: Running single Gemini trajectory")
    
    framework = MiniGridEvalFramework()
    
    try:
        result = await framework.run_single_trajectory(
            model_name="gemini-1.5-flash-latest",
            difficulty="easy",
            task_type="MiniGrid-Empty-5x5-v0",
            seed=1234,
            max_turns=5,  # Short for debugging
            collect_detailed_data=True
        )
        
        print(f"\n🎯 Result: {result}")
        print(f"Success: {result.success}")
        print(f"Steps: {result.total_steps}")
        print(f"Turns: {result.total_turns}")
        print(f"Termination: {result.termination_reason}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_single_trajectory()) 