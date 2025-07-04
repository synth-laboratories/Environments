#!/usr/bin/env python3
import asyncio
import subprocess
import sys
from eval_framework import run_crafter_eval

async def main():
    print("Starting Crafter evaluation...")
    print("=" * 50)
    
    try:
        # Run evaluation with shorter parameters for testing
        await run_crafter_eval(
            model_names=['gpt-4o-mini'],
            difficulties=['easy'],
            num_trajectories=1,
            max_turns=10
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    print("=" * 50)
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
