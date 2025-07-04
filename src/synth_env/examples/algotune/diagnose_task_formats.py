"""
Diagnose the expected input/output formats for AlgoTune tasks.
Run from repository root: python src/synth_env/examples/algotune/diagnose_task_formats.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

import asyncio
import logging
from synth_env.examples.algotune.environment import _AlgoTuneEngine

# Suppress AlgoTune logging
logging.getLogger().setLevel(logging.ERROR)


async def diagnose_task(task_name: str, n: int = 32):
    """Diagnose a single task to understand its format."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print('='*60)
    
    try:
        engine = _AlgoTuneEngine(task_name, n=n, random_seed=42)
        print(f"✓ Task loaded successfully")
        print(f"  Baseline time: {engine.baseline_time:.6f}s")
        
        # Check problem format
        print(f"\nProblem keys: {list(engine.problem.keys())}")
        for key, value in engine.problem.items():
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):
                    print(f"  {key}: list of lists, shape [{len(value)}x{len(value[0])}]")
                else:
                    print(f"  {key}: list, length {len(value)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            elif hasattr(value, 'shape'):  # numpy array
                print(f"  {key}: {type(value).__name__} shape {value.shape}")
            elif isinstance(value, (bytes, str)):
                print(f"  {key}: {type(value).__name__}, length {len(value)}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Check baseline solution format
        print(f"\nBaseline solution type: {type(engine.baseline).__name__}")
        if isinstance(engine.baseline, dict):
            print(f"  Keys: {list(engine.baseline.keys())}")
            for key, value in engine.baseline.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"    {key}: list[{len(value)}]")
                elif isinstance(value, dict):
                    print(f"    {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"    {key}: {type(value).__name__}")
        elif isinstance(engine.baseline, list):
            print(f"  List length: {len(engine.baseline)}")
            if len(engine.baseline) > 0:
                print(f"  First element type: {type(engine.baseline[0]).__name__}")
        
    except Exception as e:
        print(f"✗ Failed to load task: {e}")


async def main():
    """Diagnose several tasks to understand their formats."""
    tasks_to_diagnose = [
        ("svd", 32),
        ("linear_system_solver", 32),
        ("eigenvalues_real", 32),
        ("convolve_1d", 64),
        ("sha256_hashing", 16),
        ("pagerank", 16),
        ("kmeans", 50),
        ("pca", 50),
        ("lasso", 32),
    ]
    
    print("AlgoTune Task Format Diagnosis")
    
    for task_name, n in tasks_to_diagnose:
        await diagnose_task(task_name, n)


if __name__ == "__main__":
    asyncio.run(main())