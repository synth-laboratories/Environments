#!/usr/bin/env python3
"""
Run a single Crafter trajectory with image capture enabled.
"""

import asyncio
from src.synth_env.examples.crafter_classic.agent_demos.full_enchilada import run_full_enchilada_eval


async def main():
    print("🎮 Running Crafter evaluation with image capture enabled...")
    print("This will create a new evaluation with images in the traces.")
    
    await run_full_enchilada_eval(
        model_names=["gpt-4o-mini"],
        difficulties=["easy"],
        num_trajectories=1,  # Just one trajectory for testing
        max_turns=10,        # Shorter run for testing
        capture_images=True, # IMPORTANT: Enable image capture
        launch_viewer=False, # We'll launch manually
        output_dir=None      # Will create timestamped directory
    )
    
    print("\n✅ Evaluation complete!")
    print("Run the viewer with: python -m src.synth_env.viewer.unified_viewer_v2 <output_dir> --port 9001")


if __name__ == "__main__":
    asyncio.run(main())