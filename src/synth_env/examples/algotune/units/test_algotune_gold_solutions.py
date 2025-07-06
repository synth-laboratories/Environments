"""Test gold solutions for various AlgoTune tasks to ensure correctness."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

import pytest
import asyncio
from synth_env.examples.algotune.environment import _AlgoTuneEngine


class TestAlgoTuneGoldSolutions:
    """Comprehensive tests for AlgoTune gold solutions across different domains."""

    # Linear Algebra Tasks
    @pytest.mark.asyncio
    async def test_matrix_multiplication_gold(self):
        """Test matrix multiplication gold solution."""
        engine = _AlgoTuneEngine("matrix_multiplication", n=32)
        code = """
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    return {"C": np.dot(A, B).tolist()}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True
        assert 0.8 <= pub["speedup_vs_baseline"] <= 1.2

    @pytest.mark.asyncio
    async def test_cholesky_factorization_gold(self):
        """Test Cholesky factorization gold solution."""
        engine = _AlgoTuneEngine("cholesky_factorization", n=32)
        code = """
import numpy as np
def solve(problem):
    A = problem["matrix"]
    L = np.linalg.cholesky(A)
    return {"L": L.tolist()}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_lu_factorization_gold(self):
        """Test LU factorization gold solution."""
        engine = _AlgoTuneEngine("lu_factorization", n=32)
        code = """
import numpy as np
import scipy.linalg
def solve(problem):
    A = problem["matrix"]
    P, L, U = scipy.linalg.lu(A)
    return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    # Sorting and Searching Tasks
    @pytest.mark.asyncio
    async def test_quicksort_gold(self):
        """Test quicksort gold solution."""
        engine = _AlgoTuneEngine("quicksort", n=128)
        code = """
def solve(problem):
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    
    return {"sorted_array": quicksort(problem["array"])}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_heapsort_gold(self):
        """Test heapsort gold solution."""
        engine = _AlgoTuneEngine("heapsort", n=128)
        code = """
def solve(problem):
    import heapq
    arr = problem["array"][:]
    heapq.heapify(arr)
    return {"sorted_array": [heapq.heappop(arr) for _ in range(len(arr))]}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    # Graph Algorithm Tasks
    @pytest.mark.asyncio
    async def test_dijkstra_gold(self):
        """Test Dijkstra's shortest path gold solution."""
        engine = _AlgoTuneEngine("dijkstra", n=32)
        code = """
import heapq
def solve(problem):
    graph = problem["graph"]
    start = problem["start"]
    n = len(graph)
    
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in enumerate(graph[u]):
            if w > 0 and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return {"distances": dist}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_kruskal_mst_gold(self):
        """Test Kruskal's MST gold solution."""
        engine = _AlgoTuneEngine("kruskal_mst", n=32)
        code = """
def solve(problem):
    edges = problem["edges"]
    n = problem["num_vertices"]
    
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    edges.sort(key=lambda e: e[2])
    mst = []
    
    for u, v, w in edges:
        if union(u, v):
            mst.append([u, v, w])
    
    return {"mst": mst}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    # Dynamic Programming Tasks
    @pytest.mark.asyncio
    async def test_knapsack_gold(self):
        """Test 0/1 knapsack gold solution."""
        engine = _AlgoTuneEngine("knapsack", n=32)
        code = """
def solve(problem):
    weights = problem["weights"]
    values = problem["values"]
    capacity = problem["capacity"]
    n = len(weights)
    
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find items
    w = capacity
    items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1)
            w -= weights[i-1]
    
    return {"max_value": dp[n][capacity], "items": sorted(items)}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_longest_common_subsequence_gold(self):
        """Test LCS gold solution."""
        engine = _AlgoTuneEngine("longest_common_subsequence", n=64)
        code = """
def solve(problem):
    s1 = problem["string1"]
    s2 = problem["string2"]
    m, n = len(s1), len(s2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return {"lcs": ''.join(reversed(lcs)), "length": dp[m][n]}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    # Optimization Tasks
    @pytest.mark.asyncio
    async def test_linear_programming_gold(self):
        """Test linear programming gold solution."""
        engine = _AlgoTuneEngine("linear_programming", n=16)
        code = """
import numpy as np
from scipy.optimize import linprog
def solve(problem):
    c = problem["c"]
    A_ub = problem["A_ub"]
    b_ub = problem["b_ub"]
    bounds = problem.get("bounds", None)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    return {
        "x": result.x.tolist() if result.success else None,
        "optimal_value": float(result.fun) if result.success else None,
        "success": result.success
    }
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    # Numerical Computation Tasks
    @pytest.mark.asyncio
    async def test_fft_gold(self):
        """Test FFT gold solution."""
        engine = _AlgoTuneEngine("fft", n=128)
        code = """
import numpy as np
def solve(problem):
    signal = problem["signal"]
    fft_result = np.fft.fft(signal)
    return {"fft": [{"real": float(x.real), "imag": float(x.imag)} for x in fft_result]}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True

    @pytest.mark.asyncio
    async def test_convolution_1d_gold(self):
        """Test 1D convolution gold solution."""
        engine = _AlgoTuneEngine("convolution_1d", n=64)
        code = """
import numpy as np
def solve(problem):
    signal = problem["signal"]
    kernel = problem["kernel"]
    result = np.convolve(signal, kernel, mode='same')
    return {"convolved": result.tolist()}
"""
        ok, err, (_, pub) = await engine.step(code)
        assert ok == True
        assert pub["ok"] == True


@pytest.mark.asyncio
async def test_task_variety():
    """Test that we can access various types of AlgoTune tasks."""
    task_samples = [
        # Linear algebra
        "matrix_multiplication",
        "qr_factorization",
        "eigenvalues",
        # Sorting/searching
        "sorting",
        "binary_search",
        "quicksort",
        # Graph algorithms
        "dijkstra",
        "bellman_ford",
        "kruskal_mst",
        # Dynamic programming
        "knapsack",
        "edit_distance",
        "longest_common_subsequence",
        # Optimization
        "linear_programming",
        "lasso",
        # Signal processing
        "fft",
        "convolution_1d",
        # Combinatorial
        "traveling_salesman",
        "graph_coloring",
    ]

    # Just verify we can create engines for various tasks
    for task_name in task_samples:
        try:
            engine = _AlgoTuneEngine(task_name, n=16)
            assert engine.baseline_time > 0
        except ImportError:
            # Some tasks might require additional dependencies
            pytest.skip(f"Task {task_name} requires additional dependencies")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
