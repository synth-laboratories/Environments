#!/usr/bin/env python3
"""
Run real Gemini 1.5 Flash evaluation on MiniGrid.
"""

import sys
import os
sys.path.insert(0, 'src')

import asyncio
from synth_env.examples.minigrid.agent_demos.eval_framework import run_minigrid_eval, get_success_rate

async def run_real_gemini():
    """Run the actual Gemini evaluation."""
    
    print('🚀 Running REAL Gemini 1.5 Flash Evaluation')
    print('='*60)
    print('Model: gemini-1.5-flash-latest')
    print('Task: MiniGrid-Empty-5x5-v0 (simple navigation)')
    print('Difficulty: easy')
    print('Trajectories: 10 instances')
    print('='*60)
    
    try:
        report = await run_minigrid_eval(
            model_names=['gemini-1.5-flash-latest'],
            difficulties=['easy'],
            task_types=['MiniGrid-Empty-5x5-v0'],  # Correct environment name
            num_trajectories=10,  # 10 instances as requested
            max_turns=25
        )
        
        # Extract the key metric
        success_rate = get_success_rate(report, 'gemini-1.5-flash-latest', 'easy')
        
        print(f'\n🎯 REAL GEMINI SUCCESS RATE: {success_rate:.1f}%')
        print('='*60)
        
        if success_rate >= 80:
            print('✅ Excellent performance!')
        elif success_rate >= 60:
            print('✅ Good performance!')
        elif success_rate >= 40:
            print('⚠️  Moderate performance')
        else:
            print('❌ Needs improvement')
            
        return report
        
    except Exception as e:
        print(f'❌ Error running evaluation: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_real_gemini()) 