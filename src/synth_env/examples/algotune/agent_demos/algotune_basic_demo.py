"""
Demo showing how to optimize AlgoTune tasks using the synth-env framework.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import asyncio
from synth_env.examples.algotune.environment import AlgoTuneEnvironment
from synth_env.examples.algotune.taskset import create_algotune_task_instance

# Example optimization attempts for different algorithms
OPTIMIZATION_ATTEMPTS = {
    "qr_factorization": """
import numpy as np
def solve(problem):
    A = problem["matrix"]
    # NumPy QR but single-precision for speed
    Q, R = np.linalg.qr(A.astype(np.float32), mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
""",
    "matrix_multiplication": """
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    # Use numpy's optimized BLAS routines
    return np.dot(A, B).tolist()
""",
    "convex_hull": """
import numpy as np
from scipy.spatial import ConvexHull
def solve(problem):
    points = problem["points"]
    # Use scipy's optimized implementation
    hull = ConvexHull(points)
    return {
        "hull_vertices": hull.vertices.tolist(),
        "hull_points": np.array(points)[hull.vertices].tolist()
    }
""",
}


async def demo_task_optimization(task_name: str, problem_size: int = 128):
    """Demo optimizing a specific AlgoTune task."""
    print(f"\n{'=' * 60}")
    print(f"Optimizing {task_name} (n={problem_size})")
    print("=" * 60)

    # Create a proper task instance
    task_instance = create_algotune_task_instance(
        task_name=task_name,
        problem_size=problem_size,
        random_seed=42,
        target_speedup=1.5,
    )

    # Create environment instance
    env = AlgoTuneEnvironment(task_instance)

    # Initialize environment
    obs = await env.initialize()
    print(f"\nInitial state:")
    print(f"  Task: {obs['public']['task']}")
    print(f"  Problem size: {obs['public']['n']}")
    print(f"  Baseline time: {obs['public']['baseline_time']:.4f}s")
    print(f"  Target speedup: {task_instance.metadata.target_speedup}x")

    # Submit optimization attempt
    if task_name in OPTIMIZATION_ATTEMPTS:
        print(f"\nSubmitting optimization attempt...")
        obs = await env.step({"code": OPTIMIZATION_ATTEMPTS[task_name]})

        if obs["public"]["ok"]:
            print(f"✅ Solution correct!")
            print(f"  Time: {obs['public']['elapsed']:.4f}s")
            print(f"  Speedup: {obs['public']['speedup_vs_baseline']:.2f}x")

            if (
                obs["public"]["speedup_vs_baseline"]
                >= task_instance.metadata.target_speedup
            ):
                print(f"  🎯 Target speedup achieved!")
            else:
                print(
                    f"  ⚠️  Below target speedup of {task_instance.metadata.target_speedup}x"
                )
        else:
            print(f"❌ Solution incorrect!")
            if "error" in obs["public"]:
                print(f"  Error: {obs['public']['error']}")

    await env.terminate()


async def main():
    """Run optimization demos for multiple tasks."""
    print("AlgoTune Environment Demo")
    print("Demonstrating algorithm optimization with proper task instances")

    # Demo 1: QR Factorization
    await demo_task_optimization("qr_factorization", problem_size=128)

    # Demo 2: Matrix Multiplication
    await demo_task_optimization("matrix_multiplication", problem_size=64)

    # Demo 3: Convex Hull
    await demo_task_optimization("convex_hull", problem_size=200)

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey observations:")
    print("- Task instances properly configured with metadata")
    print("- Environment extracts parameters from task instance")
    print("- Optimization attempts are evaluated for correctness and speed")
    print("- Speedup is calculated against baseline implementation")


if __name__ == "__main__":
    # Suppress AlgoTune logging
    import logging

    logging.getLogger().setLevel(logging.WARNING)

    asyncio.run(main())
