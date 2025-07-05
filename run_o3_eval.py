#!/usr/bin/env python3
"""
Run MiniGrid evaluation with the o3 LLM (3 trajectories).
"""

import sys
import os
sys.path.insert(0, 'src')

import asyncio
from synth_env.examples.minigrid.agent_demos.eval_framework import run_minigrid_eval, get_success_rate


async def run_o3_eval():
    """Run evaluation for the o3 model."""

    model_name = 'o3'
    print('🚀 Running o3 MiniGrid Evaluation')
    print('=' * 60)
    print(f'Model: {model_name}')
    print('Task: MiniGrid-Empty-5x5-v0 (simple navigation)')
    print('Difficulty: easy')
    print('Trajectories: 3 instances')
    print('=' * 60)

    report = await run_minigrid_eval(
        model_names=[model_name],
        difficulties=['easy'],
        task_types=['MiniGrid-Empty-5x5-v0'],
        num_trajectories=3,
        max_turns=25,
    )

    success_rate = get_success_rate(report, model_name, 'easy')

    print(f'\n🎯 o3 SUCCESS RATE: {success_rate:.1f}%')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(run_o3_eval()) 