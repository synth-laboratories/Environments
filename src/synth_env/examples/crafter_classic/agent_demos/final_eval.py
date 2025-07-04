import asyncio
import logging
import numpy as np

# Suppress logging and numpy verbosity
logging.disable(logging.CRITICAL)
np.set_printoptions(threshold=0)

from eval_framework import run_crafter_eval

async def main():
    print("🎯 Starting final evaluation (50 steps, 5 trajectories)...")
    
    try:
        await run_crafter_eval(
            model_names=['gpt-4o-mini'],
            difficulties=['easy'],
            num_trajectories=5,
            max_turns=50
        )
    except Exception as e:
        print(f"Error: {e}")
    
    print("✅ Final evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())
