"""
Comprehensive verification of AlgoTune gold solutions across all task categories.
Run from repository root: python src/synth_env/examples/algotune/verify_gold_solutions.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

import asyncio
import logging
from typing import Dict, List, Tuple
from synth_env.examples.algotune.environment import _AlgoTuneEngine

# Suppress AlgoTune logging
logging.getLogger().setLevel(logging.ERROR)


# Define gold solutions for various tasks
GOLD_SOLUTIONS = {
    # Linear Algebra
    "matrix_multiplication": '''
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    return np.dot(A, B).tolist()
''',
    
    "qr_factorization": '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    Q, R = np.linalg.qr(A, mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
''',
    
    "cholesky_factorization": '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    L = np.linalg.cholesky(A)
    return {"Cholesky": {"L": L.tolist()}}
''',
    
    "lu_factorization": '''
import numpy as np
import scipy.linalg
def solve(problem):
    A = problem["matrix"]
    P, L, U = scipy.linalg.lu(A)
    return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
''',
    
    "eigenvalues": '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Sort by eigenvalue magnitude for consistency
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return {
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist()
    }
''',
    
    # Computational Geometry
    "convex_hull": '''
import numpy as np
from scipy.spatial import ConvexHull
def solve(problem):
    points = problem["points"]
    hull = ConvexHull(points)
    return {
        "hull_vertices": hull.vertices.tolist(),
        "hull_points": np.array(points)[hull.vertices].tolist()
    }
''',
    
    # Graph Algorithms
    "dijkstra": '''
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
''',
    
    "bellman_ford": '''
def solve(problem):
    edges = problem["edges"]
    num_vertices = problem["num_vertices"]
    source = problem["source"]
    
    # Initialize distances
    dist = [float('inf')] * num_vertices
    dist[source] = 0
    
    # Relax edges V-1 times
    for _ in range(num_vertices - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return {"distances": None, "has_negative_cycle": True}
    
    return {"distances": dist, "has_negative_cycle": False}
''',
    
    "kruskal_mst": '''
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
''',
    
    # Dynamic Programming
    "knapsack": '''
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
''',
    
    "edit_distance": '''
def solve(problem):
    s1 = problem["string1"]
    s2 = problem["string2"]
    m, n = len(s1), len(s2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return {"distance": dp[m][n]}
''',
    
    "longest_common_subsequence": '''
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
''',
    
    # Signal Processing
    "fft": '''
import numpy as np
def solve(problem):
    signal = problem["signal"]
    fft_result = np.fft.fft(signal)
    return {"fft": [{"real": float(x.real), "imag": float(x.imag)} for x in fft_result]}
''',
    
    "convolve_1d": '''
import numpy as np
def solve(problem):
    signal = problem["signal"]
    kernel = problem["kernel"]
    return np.convolve(signal, kernel, mode='same').tolist()
''',
    
    # Optimization
    "linear_programming": '''
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
''',
    
    # Cryptography
    "aes_gcm_encryption": '''
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def solve(problem):
    plaintext = problem["plaintext"].encode()
    key = bytes.fromhex(problem["key"])
    nonce = bytes.fromhex(problem["nonce"])
    aad = problem.get("aad", b"").encode() if isinstance(problem.get("aad", ""), str) else problem.get("aad", b"")
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    if aad:
        encryptor.authenticate_additional_data(aad)
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    
    return {
        "ciphertext": ciphertext.hex(),
        "tag": encryptor.tag.hex()
    }
''',
}


async def test_task(task_name: str, solution_code: str, n: int = 32) -> Tuple[bool, float, str]:
    """Test a single task with its gold solution."""
    try:
        engine = _AlgoTuneEngine(task_name, n=n, random_seed=42)
        ok, err, (priv, pub) = await engine.step(solution_code)
        
        if ok:
            return True, pub['speedup_vs_baseline'], ""
        else:
            return False, 0.0, err or "Solution incorrect"
    except Exception as e:
        return False, 0.0, str(e)


async def verify_all_gold_solutions():
    """Verify gold solutions for all defined tasks."""
    print("AlgoTune Gold Solutions Verification")
    print("=" * 70)
    print(f"Testing {len(GOLD_SOLUTIONS)} tasks...\n")
    
    results = []
    categories = {
        "Linear Algebra": ["matrix_multiplication", "qr_factorization", "cholesky_factorization", 
                          "lu_factorization", "eigenvalues"],
        "Graph Algorithms": ["dijkstra", "bellman_ford", "kruskal_mst"],
        "Dynamic Programming": ["knapsack", "edit_distance", "longest_common_subsequence"],
        "Signal Processing": ["fft", "convolve_1d"],
        "Optimization": ["linear_programming"],
        "Cryptography": ["aes_gcm_encryption"],
        "Computational Geometry": ["convex_hull"],
    }
    
    for category, tasks in categories.items():
        print(f"\n{category}:")
        print("-" * 50)
        
        for task_name in tasks:
            if task_name not in GOLD_SOLUTIONS:
                print(f"  ❓ {task_name:<30} - No gold solution defined")
                continue
                
            # Adjust problem size for certain tasks
            n = 32
            if task_name in ["convex_hull", "fft", "convolve_1d"]:
                n = 64
            elif task_name in ["knapsack", "dijkstra", "linear_programming"]:
                n = 16
                
            success, speedup, error = await test_task(task_name, GOLD_SOLUTIONS[task_name], n)
            
            if success:
                status = "✅ PASS"
                speedup_str = f"Speedup: {speedup:.2f}x"
            else:
                status = "❌ FAIL"
                speedup_str = f"Error: {error[:50]}..."
                
            results.append((task_name, success, speedup, error))
            print(f"  {status} {task_name:<30} {speedup_str}")
    
    # Summary
    passed = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - passed
    
    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{len(results)} tests passed")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for task_name, success, _, error in results:
            if not success:
                print(f"  - {task_name}: {error}")
    
    return passed == len(results)


async def test_additional_tasks():
    """Test additional AlgoTune tasks that exist in the repository."""
    print("\n\nTesting Additional Available Tasks")
    print("=" * 70)
    
    # Try to test some additional tasks that might exist
    additional_tasks = [
        ("svd", 32),
        ("pagerank", 16),
        ("k_means", 32),
        ("pca", 32),
        ("sha256_hashing", 16),
        ("binary_search", 128),
        ("merge_sort", 128),
        ("quick_sort", 128),
    ]
    
    for task_name, n in additional_tasks:
        try:
            engine = _AlgoTuneEngine(task_name, n=n, random_seed=42)
            print(f"\n{task_name} exists! Baseline time: {engine.baseline_time:.6f}s")
        except Exception as e:
            if "could not be imported" not in str(e):
                print(f"\n{task_name}: Unexpected error - {e}")


if __name__ == "__main__":
    asyncio.run(verify_all_gold_solutions())
    asyncio.run(test_additional_tasks())