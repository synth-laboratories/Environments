#!/usr/bin/env python3
"""
Script to explore AlgoTune tasks and understand the API.
"""

import sys
import os
import importlib

# Add the AlgoTune source to Python path
sys.path.insert(0, '/Users/joshuapurtell/Documents/GitHub/Environments/src/synth_env/examples/algotune/algotune_repo')

# Import AlgoTune components
from AlgoTuneTasks.factory import TaskFactory
from AlgoTuneTasks.base import TASK_REGISTRY

def list_available_tasks():
    """List all available AlgoTune tasks by scanning the directory."""
    tasks_dir = '/Users/joshuapurtell/Documents/GitHub/Environments/src/synth_env/examples/algotune/algotune_repo/AlgoTuneTasks'
    tasks = []
    
    for item in os.listdir(tasks_dir):
        item_path = os.path.join(tasks_dir, item)
        if os.path.isdir(item_path) and not item.startswith('__') and not item.startswith('.'):
            # Check if it has a .py file with the same name (task module)
            task_module = os.path.join(item_path, f"{item}.py")
            if os.path.exists(task_module):
                tasks.append(item)
    
    return sorted(tasks)

def explore_task(task_name):
    """Explore a specific task: generate problem, solve it, and understand the API."""
    print(f"\n{'='*60}")
    print(f"Exploring task: {task_name}")
    print('='*60)
    
    try:
        # Create task instance
        task = TaskFactory(task_name)
        print(f"✓ Successfully created task instance")
        
        # Generate a problem instance
        n = 10  # Small size for testing
        problem = task.generate_problem(n=n, random_seed=42)
        print(f"\n✓ Generated problem with n={n}")
        print(f"  Problem keys: {list(problem.keys())}")
        
        # Get the baseline/gold solution
        solution = task.solve(problem)
        print(f"\n✓ Generated baseline/gold solution")
        print(f"  Solution keys: {list(solution.keys())}")
        
        # Verify the solution
        is_valid = task.is_solution(problem, solution)
        print(f"\n✓ Solution validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Show task API methods
        print(f"\nTask API methods:")
        methods = [m for m in dir(task) if not m.startswith('_') and callable(getattr(task, m))]
        for method in sorted(methods):
            print(f"  - {method}")
            
        # Try to access task description if available
        desc_file = f'/Users/joshuapurtell/Documents/GitHub/Environments/src/synth_env/examples/algotune/algotune_repo/AlgoTuneTasks/{task_name}/description.txt'
        if os.path.exists(desc_file):
            with open(desc_file, 'r') as f:
                desc = f.read().strip()
            print(f"\nTask Description:")
            print(f"  {desc[:200]}..." if len(desc) > 200 else f"  {desc}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error exploring task: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # List all available tasks
    print("Discovering available AlgoTune tasks...")
    tasks = list_available_tasks()
    print(f"\nFound {len(tasks)} tasks:")
    for i, task in enumerate(tasks[:20]):  # Show first 20
        print(f"  {i+1:2d}. {task}")
    if len(tasks) > 20:
        print(f"  ... and {len(tasks) - 20} more")
    
    # Explore a few example tasks
    example_tasks = ['qr_factorization', 'matrix_multiplication', 'convex_hull']
    
    for task_name in example_tasks:
        if task_name in tasks:
            explore_task(task_name)
    
    # Show how to use the gold solution
    print("\n" + "="*60)
    print("SUMMARY: How to access gold solutions")
    print("="*60)
    print("""
1. Import and create a task:
   from AlgoTuneTasks.factory import TaskFactory
   task = TaskFactory('task_name')

2. Generate a problem instance:
   problem = task.generate_problem(n=size, random_seed=seed)

3. Get the baseline/gold solution:
   gold_solution = task.solve(problem)

4. Validate any solution:
   is_valid = task.is_solution(problem, candidate_solution)

The task.solve() method provides the baseline implementation that serves
as the gold standard for correctness and the baseline for timing comparisons.
""")

if __name__ == "__main__":
    main()