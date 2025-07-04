import asyncio
import logging
import sys
import os

# Suppress ALL logging
logging.disable(logging.CRITICAL)

# Suppress numpy array printing
import numpy as np
np.set_printoptions(threshold=0)

from eval_framework import run_crafter_eval

async def main():
    print("🎯 Starting minimal evaluation...")
    
    try:
        await run_crafter_eval(
            model_names=['gpt-4o-mini'],
            difficulties=['easy'],
            num_trajectories=1,
            max_turns=5  # Very short for testing
        )
    except Exception as e:
        print(f"Error: {e}")
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())
