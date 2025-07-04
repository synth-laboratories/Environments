"""
Verify gold solutions for actual AlgoTune tasks that exist.
Run from repository root: python src/synth_env/examples/algotune/verify_actual_tasks.py
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


# Define gold solutions for tasks that actually exist
ACTUAL_GOLD_SOLUTIONS = {
    # Linear Algebra (verified to exist)
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
    
    "svd": '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return {"SVD": {"U": U.tolist(), "S": S.tolist(), "Vt": Vt.tolist()}}
''',
    
    "eigenvalues_real": '''
import numpy as np
def solve(problem):
    A = problem["matrix"]
    eigenvalues = np.linalg.eigvals(A)
    # Sort by real part then imaginary for consistency
    eigenvalues = sorted(eigenvalues, key=lambda x: (x.real, x.imag))
    return {"eigenvalues": [{"real": float(e.real), "imag": float(e.imag)} for e in eigenvalues]}
''',
    
    "linear_system_solver": '''
import numpy as np
def solve(problem):
    A = problem["A"]
    b = problem["b"]
    x = np.linalg.solve(A, b)
    return {"x": x.tolist()}
''',
    
    # Computational Geometry (verified to exist)
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
    
    "delaunay": '''
import numpy as np
from scipy.spatial import Delaunay
def solve(problem):
    points = problem["points"]
    tri = Delaunay(points)
    return {"simplices": tri.simplices.tolist()}
''',
    
    "voronoi_diagram": '''
import numpy as np
from scipy.spatial import Voronoi
def solve(problem):
    points = problem["points"]
    vor = Voronoi(points)
    return {
        "vertices": vor.vertices.tolist(),
        "ridge_vertices": vor.ridge_vertices,
        "ridge_points": vor.ridge_points.tolist()
    }
''',
    
    # Graph Algorithms (using actual task names)
    "shortest_path_dijkstra": '''
import heapq
def solve(problem):
    adjacency_matrix = problem["adjacency_matrix"]
    source = problem["source"]
    n = len(adjacency_matrix)
    
    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(n):
            w = adjacency_matrix[u][v]
            if w > 0 and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return {"distances": dist}
''',
    
    "minimum_spanning_tree": '''
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
    
    edges_list = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges[i])):
            if edges[i][j] > 0:
                edges_list.append((i, j, edges[i][j]))
    
    edges_list.sort(key=lambda e: e[2])
    mst = []
    total_weight = 0
    
    for u, v, w in edges_list:
        if union(u, v):
            mst.append([u, v, w])
            total_weight += w
    
    return {"mst": mst, "total_weight": total_weight}
''',
    
    "pagerank": '''
import numpy as np
def solve(problem):
    adjacency_matrix = np.array(problem["adjacency_matrix"])
    damping_factor = problem.get("damping_factor", 0.85)
    max_iterations = problem.get("max_iterations", 100)
    tolerance = problem.get("tolerance", 1e-6)
    
    n = len(adjacency_matrix)
    M = adjacency_matrix.copy()
    
    # Normalize columns
    col_sums = M.sum(axis=0)
    col_sums[col_sums == 0] = 1
    M = M / col_sums
    
    # Initialize PageRank vector
    pr = np.ones(n) / n
    
    for _ in range(max_iterations):
        pr_new = (1 - damping_factor) / n + damping_factor * M @ pr
        if np.abs(pr_new - pr).max() < tolerance:
            break
        pr = pr_new
    
    return {"pagerank": pr.tolist()}
''',
    
    # Optimization (verified to exist)
    "lasso": '''
import numpy as np
from sklearn.linear_model import Lasso as SkLasso
def solve(problem):
    X = np.array(problem["X"])
    y = np.array(problem["y"])
    alpha = problem["alpha"]
    
    lasso = SkLasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasso.fit(X, y)
    
    return {"beta": lasso.coef_.tolist()}
''',
    
    "least_squares": '''
import numpy as np
def solve(problem):
    A = np.array(problem["A"])
    b = np.array(problem["b"])
    
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return {"x": x.tolist()}
''',
    
    # Signal Processing (verified to exist)
    "fft_real_scipy_fftpack": '''
import numpy as np
from scipy import fftpack
def solve(problem):
    signal = problem["signal"]
    result = fftpack.fft(signal)
    return {"fft": [{"real": float(x.real), "imag": float(x.imag)} for x in result]}
''',
    
    "convolve_1d": '''
import numpy as np
def solve(problem):
    signal = problem["signal"]
    kernel = problem["kernel"]
    return np.convolve(signal, kernel, mode='same').tolist()
''',
    
    "correlate_1d": '''
import numpy as np
def solve(problem):
    signal1 = problem["signal1"]
    signal2 = problem["signal2"]
    return np.correlate(signal1, signal2, mode='same').tolist()
''',
    
    # Machine Learning (verified to exist)
    "kmeans": '''
import numpy as np
from sklearn.cluster import KMeans as SkKMeans
def solve(problem):
    data = np.array(problem["data"])
    n_clusters = problem["n_clusters"]
    random_state = problem.get("random_state", 42)
    
    kmeans = SkKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(data)
    
    return {
        "labels": kmeans.labels_.tolist(),
        "centers": kmeans.cluster_centers_.tolist()
    }
''',
    
    "pca": '''
import numpy as np
from sklearn.decomposition import PCA as SkPCA
def solve(problem):
    data = np.array(problem["data"])
    n_components = problem["n_components"]
    
    pca = SkPCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    
    return {
        "transformed_data": transformed.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist()
    }
''',
    
    # Cryptography (verified to exist)
    "sha256_hashing": '''
import hashlib
def solve(problem):
    data = problem["data"]
    if isinstance(data, str):
        data = data.encode()
    
    hash_obj = hashlib.sha256(data)
    return {"hash": hash_obj.hexdigest()}
''',
    
    "aes_gcm_encryption": '''
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def solve(problem):
    plaintext = problem["plaintext"].encode()
    key = bytes.fromhex(problem["key"])
    nonce = bytes.fromhex(problem["nonce"])
    aad = problem.get("aad", b"")
    if isinstance(aad, str):
        aad = aad.encode()
    
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


async def verify_actual_gold_solutions():
    """Verify gold solutions for actual AlgoTune tasks."""
    print("AlgoTune Gold Solutions Verification (Actual Tasks)")
    print("=" * 70)
    print(f"Testing {len(ACTUAL_GOLD_SOLUTIONS)} tasks...\n")
    
    results = []
    categories = {
        "Linear Algebra": ["matrix_multiplication", "qr_factorization", "cholesky_factorization", 
                          "lu_factorization", "svd", "eigenvalues_real", "linear_system_solver"],
        "Computational Geometry": ["convex_hull", "delaunay", "voronoi_diagram"],
        "Graph Algorithms": ["shortest_path_dijkstra", "minimum_spanning_tree", "pagerank"],
        "Optimization": ["lasso", "least_squares"],
        "Signal Processing": ["fft_real_scipy_fftpack", "convolve_1d", "correlate_1d"],
        "Machine Learning": ["kmeans", "pca"],
        "Cryptography": ["sha256_hashing", "aes_gcm_encryption"],
    }
    
    for category, tasks in categories.items():
        print(f"\n{category}:")
        print("-" * 50)
        
        for task_name in tasks:
            if task_name not in ACTUAL_GOLD_SOLUTIONS:
                continue
                
            # Adjust problem size for certain tasks
            n = 32
            if task_name in ["convex_hull", "delaunay", "voronoi_diagram"]:
                n = 50
            elif task_name in ["pagerank", "shortest_path_dijkstra"]:
                n = 16
            elif task_name in ["kmeans", "pca"]:
                n = 100
                
            success, speedup, error = await test_task(task_name, ACTUAL_GOLD_SOLUTIONS[task_name], n)
            
            if success:
                status = "✅ PASS"
                speedup_str = f"Speedup: {speedup:.2f}x"
            else:
                status = "❌ FAIL"
                speedup_str = f"Error: {error[:40]}..."
                
            results.append((task_name, success, speedup, error))
            print(f"  {status} {task_name:<28} {speedup_str}")
    
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


if __name__ == "__main__":
    asyncio.run(verify_actual_gold_solutions())