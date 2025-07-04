#!/usr/bin/env python3
"""
Demo showing how to use the pure success rate functionality in MiniGrid evaluation.

This shows the simplest way to get a single number that tells you if the agent
is getting the task right.
"""

import asyncio
from eval_framework import run_minigrid_eval, get_success_rate, get_pure_success_scores

async def demo_success_rates():
    """Demo showing different ways to access success rates."""
    
    print("🎯 MiniGrid Success Rate Demo")
    print("="*50)
    
    # Run a quick evaluation (you can replace with your actual models)
    print("Running evaluation...")
    report = await run_minigrid_eval(
        model_names=["gpt-4o-mini"],  # Replace with your model
        difficulties=["easy"],
        task_types=["Empty-5x5-v0"],  # Simple task
        num_trajectories=2,  # Quick test
        max_turns=20
    )
    
    print("\n" + "="*50)
    print("📊 ACCESSING SUCCESS RATES")
    print("="*50)
    
    # Method 1: Direct access from report
    print("\n1. Pure success scores from report:")
    if "pure_success_scores" in report:
        for model_condition, success_rate in report["pure_success_scores"].items():
            print(f"   {model_condition}: {success_rate:.1f}%")
    
    # Method 2: Using helper function
    print("\n2. Using get_success_rate() helper:")
    success_rate = get_success_rate(report, "gpt-4o-mini", "easy")
    print(f"   GPT-4o-mini (easy): {success_rate:.1f}%")
    
    # Method 3: Overall success rate across all difficulties
    overall_success = get_success_rate(report, "gpt-4o-mini")
    print(f"   GPT-4o-mini (overall): {overall_success:.1f}%")
    
    print("\n" + "="*50)
    print("✅ SUCCESS INTERPRETATION")
    print("="*50)
    print("• 100% = Perfect task completion")
    print("• 75%+ = Strong performance")
    print("• 50%+ = Moderate performance") 
    print("• 25%+ = Weak performance")
    print("• 0%   = No successful completions")
    
    return report

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_success_rates()) 