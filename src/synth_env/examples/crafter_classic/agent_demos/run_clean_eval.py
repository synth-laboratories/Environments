#!/usr/bin/env python3
import asyncio
import sys
import os
from contextlib import redirect_stderr
from io import StringIO
from eval_framework import run_crafter_eval

async def main():
    print("Starting Crafter evaluation...")
    print("=" * 50)
    
    # Capture stderr to filter out debug messages
    stderr_capture = StringIO()
    
    try:
        with redirect_stderr(stderr_capture):
            # Run evaluation with proper parameters
            await run_crafter_eval(
                model_names=['gpt-4o-mini'],
                difficulties=['easy'],
                num_trajectories=5,
                max_turns=50
            )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    print("=" * 50)
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
