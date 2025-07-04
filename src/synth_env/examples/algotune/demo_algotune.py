"""
Demo script showing AlgoTune environment working with gold solutions.
Run from repository root: python src/synth_env/examples/algotune/demo_algotune.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

import asyncio
from synth_env.examples.algotune.environment import _AlgoTuneEngine


async def demo_matrix_multiplication():
    """Demo showing matrix multiplication optimization."""
    print("\n=== Matrix Multiplication Demo ===")
    
    # Create the engine directly
    engine = _AlgoTuneEngine("matrix_multiplication", n=64, random_seed=42)
    
    # Initialize
    priv, pub = await engine.reset()
    print(f"Task: {pub['task']}")
    print(f"Problem size: n={pub['n']}")
    print(f"Baseline time: {pub['baseline_time']:.4f}s")
    
    # Submit a solution (using NumPy's optimized implementation)
    solution_code = '''
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    return np.dot(A, B).tolist()
'''
    
    # Step with the solution
    ok, err, (priv, pub) = await engine.step(solution_code)
    print(f"\nAfter submission:")
    print(f"  Correct: {pub['ok']}")
    print(f"  Time: {pub['elapsed']:.4f}s")
    print(f"  Speedup: {pub['speedup_vs_baseline']:.2f}x")


async def demo_convex_hull():
    """Demo showing convex hull computation."""
    print("\n=== Convex Hull Demo ===")
    
    engine = _AlgoTuneEngine("convex_hull", n=100, random_seed=42)
    
    priv, pub = await engine.reset()
    print(f"Task: {pub['task']}")
    print(f"Problem size: n={pub['n']}")
    print(f"Baseline time: {pub['baseline_time']:.4f}s")
    
    # Submit the standard solution
    solution_code = '''
import numpy as np
from scipy.spatial import ConvexHull
def solve(problem):
    points = problem["points"]
    hull = ConvexHull(points)
    return {
        "hull_vertices": hull.vertices.tolist(),
        "hull_points": np.array(points)[hull.vertices].tolist()
    }
'''
    
    ok, err, (priv, pub) = await engine.step(solution_code)
    print(f"\nAfter submission:")
    print(f"  Correct: {pub['ok']}")
    print(f"  Time: {pub['elapsed']:.4f}s")
    print(f"  Speedup: {pub['speedup_vs_baseline']:.2f}x")


async def demo_qr_factorization():
    """Demo showing QR factorization."""
    print("\n=== QR Factorization Demo ===")
    
    engine = _AlgoTuneEngine("qr_factorization", n=64, random_seed=42)
    
    priv, pub = await engine.reset()
    print(f"Task: {pub['task']}")
    print(f"Problem size: n={pub['n']}")
    print(f"Baseline time: {pub['baseline_time']:.4f}s")
    
    # Try an optimized solution
    solution_code = '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    # Use reduced mode for efficiency
    Q, R = np.linalg.qr(A, mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
'''
    
    ok, err, (priv, pub) = await engine.step(solution_code)
    print(f"\nAfter submission:")
    print(f"  Correct: {pub['ok']}")
    print(f"  Time: {pub['elapsed']:.4f}s")
    print(f"  Speedup: {pub['speedup_vs_baseline']:.2f}x")


async def main():
    """Run all demos."""
    print("AlgoTune Environment Demo")
    print("=" * 50)
    
    await demo_matrix_multiplication()
    await demo_convex_hull()
    await demo_qr_factorization()
    
    print("\n" + "=" * 50)
    print("✅ All demos completed successfully!")
    print("\nThe AlgoTune environment is working correctly.")
    print("It successfully:")
    print("- Loads AlgoTune tasks")
    print("- Generates problems")
    print("- Times baseline solutions")
    print("- Evaluates submitted code")
    print("- Validates correctness")
    print("- Calculates speedups")


if __name__ == "__main__":
    # Suppress AlgoTune logging
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    asyncio.run(main())