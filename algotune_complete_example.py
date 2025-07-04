#!/usr/bin/env python3
"""
Complete example of using AlgoTune: 
- Load a task
- Get baseline solution
- Create and test a custom optimized solution
"""

import sys
import os
import time
import numpy as np

# Add the AlgoTune source to Python path
sys.path.insert(0, '/Users/joshuapurtell/Documents/GitHub/Environments/src/synth_env/examples/algotune/algotune_repo')

from AlgoTuneTasks.factory import TaskFactory

def main():
    print("="*60)
    print("Complete AlgoTune Example: Matrix Multiplication")
    print("="*60)
    
    # 1. Create task instance
    task = TaskFactory('matrix_multiplication')
    print("✓ Created matrix_multiplication task")
    
    # 2. Generate a problem
    n = 50  # Matrix size
    problem = task.generate_problem(n=n, random_seed=123)
    print(f"\n✓ Generated problem with n={n}")
    print(f"  Problem contains: {list(problem.keys())}")
    if 'A' in problem and 'B' in problem:
        # Convert to numpy arrays to check shape
        A = np.array(problem['A'])
        B = np.array(problem['B'])
        print(f"  Matrix A shape: {A.shape}")
        print(f"  Matrix B shape: {B.shape}")
    
    # 3. Get baseline solution and time it
    print("\n" + "-"*40)
    print("Baseline Solution:")
    print("-"*40)
    
    start = time.perf_counter()
    baseline_solution = task.solve(problem)
    baseline_time = time.perf_counter() - start
    
    print(f"✓ Baseline solution computed in {baseline_time:.6f} seconds")
    
    # Check what the solution looks like
    if isinstance(baseline_solution, dict):
        print(f"  Solution type: dict with keys {list(baseline_solution.keys())}")
    elif isinstance(baseline_solution, np.ndarray):
        print(f"  Solution type: numpy array with shape {baseline_solution.shape}")
    elif isinstance(baseline_solution, list):
        print(f"  Solution type: list with length {len(baseline_solution)}")
    
    # 4. Verify baseline solution
    is_valid = task.is_solution(problem, baseline_solution)
    print(f"✓ Baseline solution valid: {is_valid}")
    
    # 5. Create a custom optimized solution
    print("\n" + "-"*40)
    print("Custom Optimized Solution:")
    print("-"*40)
    
    def custom_solve(problem):
        """Custom matrix multiplication using float32 for speed."""
        # Convert lists to numpy arrays
        A = np.array(problem['A'])
        B = np.array(problem['B'])
        
        # Convert to float32 for faster computation
        A_float32 = A.astype(np.float32)
        B_float32 = B.astype(np.float32)
        
        # Perform multiplication
        C_float32 = np.matmul(A_float32, B_float32)
        
        # Convert back to float64 to match expected precision
        C = C_float32.astype(np.float64)
        
        # Return in the expected format
        # Matrix multiplication just returns the result matrix as a list
        return C.tolist()
    
    # Time the custom solution
    start = time.perf_counter()
    custom_solution = custom_solve(problem)
    custom_time = time.perf_counter() - start
    
    print(f"✓ Custom solution computed in {custom_time:.6f} seconds")
    
    # 6. Validate custom solution
    is_valid_custom = task.is_solution(problem, custom_solution)
    print(f"✓ Custom solution valid: {is_valid_custom}")
    
    # 7. Calculate speedup
    if custom_time > 0:
        speedup = baseline_time / custom_time
        print(f"\n{'='*40}")
        print(f"PERFORMANCE RESULTS:")
        print(f"{'='*40}")
        print(f"Baseline time: {baseline_time:.6f} seconds")
        print(f"Custom time:   {custom_time:.6f} seconds")
        print(f"Speedup:       {speedup:.2f}x")
        
        if speedup > 1:
            print(f"✓ SUCCESS: Custom solution is {speedup:.2f}x faster!")
        else:
            print(f"✗ Custom solution is slower than baseline")
    
    # 8. Show all available tasks (first 30)
    print(f"\n{'='*60}")
    print("Other Available AlgoTune Tasks:")
    print("="*60)
    
    tasks_dir = '/Users/joshuapurtell/Documents/GitHub/Environments/src/synth_env/examples/algotune/algotune_repo/AlgoTuneTasks'
    tasks = []
    for item in sorted(os.listdir(tasks_dir)):
        if os.path.isdir(os.path.join(tasks_dir, item)) and not item.startswith('_'):
            task_file = os.path.join(tasks_dir, item, f"{item}.py")
            if os.path.exists(task_file):
                tasks.append(item)
    
    # Group tasks by category
    categories = {
        'Linear Algebra': ['matrix_multiplication', 'qr_factorization', 'cholesky_factorization', 
                          'lu_factorization', 'svd', 'eigenvalues_real', 'eigenvectors_real'],
        'Optimization': ['lasso', 'group_lasso', 'markowitz', 'lp_box', 'qp'],
        'Graph Algorithms': ['shortest_path_dijkstra', 'minimum_spanning_tree', 'pagerank',
                           'graph_coloring_assign', 'max_clique_cpsat'],
        'Signal Processing': ['fft_convolution', 'convolve_1d', 'convolve2d_full_fill'],
        'Cryptography': ['aes_gcm_encryption', 'chacha_encryption', 'sha256_hashing'],
        'Machine Learning': ['kmeans', 'pca', 'svm', 'nmf']
    }
    
    for category, task_list in categories.items():
        matching = [t for t in task_list if t in tasks]
        if matching:
            print(f"\n{category}:")
            for t in matching[:5]:
                print(f"  - {t}")

if __name__ == "__main__":
    main()