import asyncio
from eval_framework import run_crafter_eval

async def main():
    print("Starting evaluation...")
    await run_crafter_eval(
        model_names=['gpt-4o-mini'],
        difficulties=['easy'],
        num_trajectories=1,
        max_turns=10
    )
    print("Done!")

asyncio.run(main())
