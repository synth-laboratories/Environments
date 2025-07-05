"""
React-style agent demo for AlgoTune optimization tasks.
This demonstrates how an LLM agent might approach algorithm optimization.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import asyncio
from typing import Dict, Any, List
from synth_ai.zyk import LM
from synth_env.examples.algotune.environment import AlgoTuneEnvironment
from synth_env.examples.algotune.taskset import create_algotune_task_instance


class AlgoTuneReactAgent:
    """Simple React-style agent for algorithm optimization."""

    def __init__(self, model_name: str = "gpt-4"):
        self.lm = LM(model_name)
        self.max_attempts = 3

    def format_observation(self, obs: Dict[str, Any], attempt_num: int) -> str:
        """Format observation for the LLM."""
        public_state = obs.get("public", {})

        if attempt_num == 0:
            # Initial observation
            return f"""You are optimizing the {public_state.get("task", "unknown")} algorithm.

Current state:
- Problem size: n={public_state.get("n", "unknown")}
- Baseline time: {public_state.get("baseline_time", 0):.4f}s
- Target speedup: {public_state.get("target_speedup", 1.5)}x
- Attempts so far: {public_state.get("attempts", 0)}

Your goal is to write an optimized solve(problem) function that:
1. Is correct (passes all tests)
2. Runs faster than the baseline implementation
3. Achieves at least the target speedup if possible

The problem format and expected solution format will depend on the specific task.
Think about algorithmic optimizations, data structure choices, and numerical tricks."""

        else:
            # After submission
            msg = f"\nAttempt {attempt_num} results:\n"

            if public_state.get("ok", False):
                msg += f"✅ Solution correct!\n"
                msg += f"- Time: {public_state.get('elapsed', 0):.4f}s\n"
                msg += f"- Speedup: {public_state.get('speedup_vs_baseline', 0):.2f}x\n"
                msg += f"- Best speedup so far: {public_state.get('best_speedup', 0):.2f}x\n"

                if public_state.get("speedup_vs_baseline", 0) < 1.0:
                    msg += "\n⚠️ Your solution is slower than baseline. Consider:\n"
                    msg += "- Using more efficient algorithms\n"
                    msg += "- Reducing unnecessary operations\n"
                    msg += "- Using optimized libraries\n"
            else:
                msg += f"❌ Solution incorrect!\n"
                if "error" in public_state:
                    msg += f"Error: {public_state['error']}\n"
                msg += "\nMake sure your solution:\n"
                msg += "- Returns the correct format\n"
                msg += "- Handles all edge cases\n"
                msg += "- Produces numerically accurate results\n"

            return msg

    def generate_prompt(self, task_name: str, obs_history: List[str]) -> str:
        """Generate prompt for the LLM."""
        prompt = f"You are optimizing the {task_name} algorithm.\n\n"

        # Add observation history
        for obs in obs_history:
            prompt += obs + "\n"

        # Add task-specific hints
        if task_name == "matrix_multiplication":
            prompt += "\nHints for matrix multiplication:\n"
            prompt += "- Consider using numpy's optimized BLAS routines\n"
            prompt += "- Think about memory layout and cache efficiency\n"
            prompt += "- For small matrices, overhead might dominate\n"
        elif task_name == "qr_factorization":
            prompt += "\nHints for QR factorization:\n"
            prompt += "- Consider precision vs speed tradeoffs\n"
            prompt += "- Look into specialized QR algorithms\n"
            prompt += "- Think about whether you need full or reduced QR\n"
        elif task_name == "convex_hull":
            prompt += "\nHints for convex hull:\n"
            prompt += (
                "- Consider algorithmic complexity (Graham scan, QuickHull, etc)\n"
            )
            prompt += "- Think about preprocessing steps\n"
            prompt += "- Consider special cases (collinear points, etc)\n"

        prompt += "\nGenerate a complete solve(problem) function. Include all necessary imports.\n"
        prompt += "Respond with ONLY the Python code, no explanations.\n"

        return prompt

    async def optimize_task(self, task_name: str, problem_size: int = 128):
        """Attempt to optimize a specific task."""
        print(f"\n{'=' * 60}")
        print(f"React Agent optimizing {task_name} (n={problem_size})")
        print("=" * 60)

        # Create task instance
        task_instance = create_algotune_task_instance(
            task_name=task_name,
            problem_size=problem_size,
            random_seed=42,
            target_speedup=1.5,
        )

        # Create environment
        env = AlgoTuneEnvironment(task_instance)

        # Initialize
        obs = await env.initialize()
        obs_history = [self.format_observation(obs, 0)]

        print(obs_history[0])

        # Make optimization attempts
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🤔 Thinking about attempt {attempt}...")

            # Generate solution
            prompt = self.generate_prompt(task_name, obs_history)
            response = self.lm(prompt)

            # Extract code (simple extraction - in practice would be more robust)
            code = response.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            print(f"📝 Generated solution:")
            print("-" * 40)
            for line in code.split("\n")[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(code.split("\n")) > 10:
                print("  ...")
            print("-" * 40)

            # Submit solution
            obs = await env.step({"code": code})
            obs_msg = self.format_observation(obs, attempt)
            obs_history.append(obs_msg)
            print(obs_msg)

            # Check if we achieved target
            if (
                obs["public"].get("speedup_vs_baseline", 0)
                >= task_instance.metadata.target_speedup
            ):
                print(f"\n🎉 Target speedup achieved in {attempt} attempts!")
                break

        await env.terminate()

        return obs["public"].get("best_speedup", 0)


async def main():
    """Run the React agent on example tasks."""
    print("AlgoTune React Agent Demo")
    print("Demonstrating LLM-based algorithm optimization")

    # Note: This uses a mock LM that returns simple solutions
    # In practice, you'd use a real LLM that can generate optimized code
    agent = AlgoTuneReactAgent(model_name="mock")

    # Example 1: Matrix Multiplication
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Matrix Multiplication")
    best_speedup = await agent.optimize_task("matrix_multiplication", problem_size=32)

    # Example 2: Convex Hull
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Convex Hull")
    best_speedup = await agent.optimize_task("convex_hull", problem_size=100)

    print("\n" + "=" * 60)
    print("React Agent Demo Complete!")
    print("\nIn a real scenario, the LLM would:")
    print("- Analyze the baseline algorithm")
    print("- Research optimization techniques")
    print("- Iteratively improve based on feedback")
    print("- Learn from failed attempts")


# Mock solutions for demo purposes
MOCK_SOLUTIONS = {
    "matrix_multiplication": [
        # Attempt 1: Basic numpy
        """
import numpy as np
def solve(problem):
    A = problem["A"]
    B = problem["B"]
    return np.dot(A, B).tolist()
""",
        # Attempt 2: Try float32
        """
import numpy as np
def solve(problem):
    A = np.array(problem["A"], dtype=np.float32)
    B = np.array(problem["B"], dtype=np.float32)
    result = np.dot(A, B)
    return result.astype(np.float64).tolist()
""",
        # Attempt 3: Use @ operator
        """
import numpy as np
def solve(problem):
    A = np.asarray(problem["A"])
    B = np.asarray(problem["B"])
    return (A @ B).tolist()
""",
    ],
    "convex_hull": [
        # Attempt 1: Basic scipy
        """
import numpy as np
from scipy.spatial import ConvexHull
def solve(problem):
    points = problem["points"]
    hull = ConvexHull(points)
    return {
        "hull_vertices": hull.vertices.tolist(),
        "hull_points": np.array(points)[hull.vertices].tolist()
    }
""",
        # Attempt 2: Try to optimize with numpy arrays
        """
import numpy as np
from scipy.spatial import ConvexHull
def solve(problem):
    points = np.asarray(problem["points"])
    hull = ConvexHull(points, qhull_options='QJ')
    return {
        "hull_vertices": hull.vertices.tolist(),
        "hull_points": points[hull.vertices].tolist()
    }
""",
    ],
}

# Monkey-patch the LM class for demo
attempt_counters = {}


class MockLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, prompt):
        # Extract task name from prompt
        for task_name in MOCK_SOLUTIONS:
            if task_name in prompt:
                if task_name not in attempt_counters:
                    attempt_counters[task_name] = 0

                solutions = MOCK_SOLUTIONS[task_name]
                idx = min(attempt_counters[task_name], len(solutions) - 1)
                attempt_counters[task_name] += 1

                return solutions[idx]

        # Default solution
        return """
import numpy as np
def solve(problem):
    # Default implementation
    return {}
"""


# Replace LM with mock for demo
LM = MockLM


if __name__ == "__main__":
    import logging

    logging.getLogger().setLevel(logging.WARNING)

    asyncio.run(main())
