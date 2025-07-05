"""Unit tests for AlgoTune environment integration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

import asyncio
import pytest
from synth_env.examples.algotune.environment import AlgoTuneEnvironment, _AlgoTuneEngine
from synth_env.tasks.core import TaskInstance
from synth_env.environment.tools import EnvToolCall


class TestAlgoTuneEngine:
    """Test the AlgoTune engine wrapper."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test that engine initializes correctly with a valid task."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32, random_seed=42)
        assert engine.task_name == "matrix_multiplication"
        assert engine.n == 32
        assert engine.random_seed == 42
        assert engine.baseline_time > 0
        assert engine.best_time == float("inf")
        assert engine.attempts == 0

    @pytest.mark.asyncio
    async def test_engine_reset(self):
        """Test engine reset returns proper state."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32)
        priv, pub = await engine.reset()

        assert priv["terminated"] == False
        assert pub["task"] == "matrix_multiplication"
        assert pub["n"] == 32
        assert pub["baseline_time"] > 0
        assert pub["attempts"] == 0

    @pytest.mark.asyncio
    async def test_engine_step_with_valid_code(self):
        """Test engine step with valid solution code."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32)

        # Use the baseline solution as our test code
        code = """
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    return {"C": np.dot(A, B).tolist()}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert err is None
        assert pub["attempt"] == 1
        assert pub["ok"] == True
        assert pub["elapsed"] > 0
        assert pub["speedup_vs_baseline"] > 0

    @pytest.mark.asyncio
    async def test_engine_step_with_invalid_code(self):
        """Test engine step with code missing solve function."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32)

        code = """
def helper():
    pass
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == False
        assert err == "No `solve` function defined"
        assert pub == {}

    @pytest.mark.asyncio
    async def test_engine_step_with_incorrect_solution(self):
        """Test engine step with incorrect solution."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32)

        code = """
def solve(problem):
    # Return wrong result
    return {"C": [[0]]}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == False
        assert pub["attempt"] == 1
        assert pub["ok"] == False
        assert pub["speedup_vs_baseline"] == 0.0


class TestAlgoTuneEnvironment:
    """Test the AlgoTune environment wrapper."""

    @pytest.mark.asyncio
    async def test_environment_initialization(self):
        """Test environment initializes correctly."""
        task = TaskInstance(id="test-1", description="Test AlgoTune")
        env = AlgoTuneEnvironment(task, task_name="matrix_multiplication", n=32)

        obs = await env.initialize()
        assert "public" in obs
        assert "private" in obs
        assert obs["public"]["task"] == "matrix_multiplication"
        assert obs["public"]["baseline_time"] > 0

    @pytest.mark.asyncio
    async def test_environment_step_with_tool_call(self):
        """Test environment step with proper tool call."""
        task = TaskInstance(id="test-1", description="Test AlgoTune")
        env = AlgoTuneEnvironment(task, task_name="matrix_multiplication", n=32)
        await env.initialize()

        code = """
import numpy as np
def solve(problem):
    return {"C": np.dot(problem["A"], problem["B"]).tolist()}
"""
        tool_call = EnvToolCall(tool="optimise", args={"code": code})
        obs = await env.step(tool_call)

        assert "public" in obs
        assert obs["public"]["ok"] == True
        assert obs["public"]["speedup_vs_baseline"] > 0

    @pytest.mark.asyncio
    async def test_environment_step_with_dict_call(self):
        """Test environment step with dict-style call."""
        task = TaskInstance(id="test-1", description="Test AlgoTune")
        env = AlgoTuneEnvironment(task, task_name="matrix_multiplication", n=32)
        await env.initialize()

        code = """
def solve(problem):
    import numpy as np
    return {"C": np.dot(problem["A"], problem["B"]).tolist()}
"""
        obs = await env.step({"code": code})

        assert "public" in obs
        assert obs["public"]["ok"] == True

    @pytest.mark.asyncio
    async def test_environment_serialization(self):
        """Test environment serialization/deserialization."""
        task = TaskInstance(id="test-1", description="Test AlgoTune")
        env = AlgoTuneEnvironment(task, task_name="sorting", n=64, random_seed=123)

        # Serialize
        snapshot = await env._serialize_engine()
        assert snapshot["task"] == "sorting"
        assert snapshot["n"] == 64
        assert snapshot["seed"] == 123

        # Deserialize
        new_engine = await AlgoTuneEnvironment._deserialize_engine(task, snapshot)
        assert new_engine.task_name == "sorting"
        assert new_engine.n == 64
        assert new_engine.random_seed == 123


class TestAlgoTuneGoldSolutions:
    """Test that gold solutions work correctly for various tasks."""

    @pytest.mark.asyncio
    async def test_qr_factorization_gold(self):
        """Test QR factorization gold solution."""
        engine = _AlgoTuneEngine("qr_factorization", n=64)

        # Submit the gold solution
        code = """
import numpy as np
def solve(problem):
    A = problem["matrix"]
    Q, R = np.linalg.qr(A, mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert pub["ok"] == True
        assert pub["speedup_vs_baseline"] > 0.5  # Should be close to 1.0

    @pytest.mark.asyncio
    async def test_sorting_gold(self):
        """Test sorting gold solution."""
        engine = _AlgoTuneEngine("sorting", n=128)

        code = """
def solve(problem):
    arr = problem["array"]
    return {"sorted_array": sorted(arr)}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_binary_search_gold(self):
        """Test binary search gold solution."""
        engine = _AlgoTuneEngine("binary_search", n=128)

        code = """
def solve(problem):
    arr = problem["sorted_array"]
    target = problem["target"]
    
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return {"index": mid}
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return {"index": -1}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_convex_hull_gold(self):
        """Test convex hull gold solution."""
        engine = _AlgoTuneEngine("convex_hull", n=64)

        code = """
def solve(problem):
    points = problem["points"]
    
    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
    
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return {"hull": points}
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return {"hull": lower[:-1] + upper[:-1]}
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_maximum_subarray_gold(self):
        """Test maximum subarray (Kadane's algorithm) gold solution."""
        engine = _AlgoTuneEngine("maximum_subarray", n=128)

        code = """
def solve(problem):
    arr = problem["array"]
    
    max_ending_here = max_so_far = arr[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            temp_start = i
        else:
            max_ending_here = max_ending_here + arr[i]
        
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = temp_start
            end = i
    
    return {
        "max_sum": max_so_far,
        "start_index": start,
        "end_index": end
    }
"""
        ok, err, (priv, pub) = await engine.step(code)

        assert ok == True
        assert pub["ok"] == True


@pytest.mark.asyncio
async def test_multiple_attempts_tracking():
    """Test that multiple attempts are tracked correctly."""
    engine = _AlgoTuneEngine("sorting", n=64)

    # First attempt - baseline solution
    code1 = """
def solve(problem):
    return {"sorted_array": sorted(problem["array"])}
"""
    ok1, _, (_, pub1) = await engine.step(code1)
    assert pub1["attempt"] == 1
    assert ok1 == True

    # Second attempt - potentially optimized
    code2 = """
def solve(problem):
    arr = problem["array"][:]
    arr.sort()  # In-place sort might be faster
    return {"sorted_array": arr}
"""
    ok2, _, (_, pub2) = await engine.step(code2)
    assert pub2["attempt"] == 2
    assert ok2 == True
    assert pub2["best_speedup"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
