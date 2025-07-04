"""
Test script for MiniGrid evaluation framework.
Quick test to verify the framework works correctly.
"""

import asyncio
from synth_env.examples.minigrid.agent_demos.eval_framework import run_minigrid_eval

async def main():
    """Test the MiniGrid evaluation framework with a simple setup."""
    
    # Test with a small configuration
    report = await run_minigrid_eval(
        model_names=["gpt-4o-mini"],
        difficulties=["easy"],
        task_types=["MiniGrid-Empty-6x6-v0"],
        num_trajectories=1,
        max_turns=20
    )
    
    print("\n✅ MiniGrid evaluation framework test completed!")
    print(f"Generated report with {len(report['raw_trajectory_results'])} trajectories")

if __name__ == "__main__":
    asyncio.run(main()) 