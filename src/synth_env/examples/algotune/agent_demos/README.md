# AlgoTune Agent Demos

This directory contains demonstration scripts for testing agents on the AlgoTune environment - an algorithmic optimization challenge where agents must write faster implementations of computational algorithms.

## 🎯 Quick Start

**For basic algorithm optimization demo:**
```bash
python algotune_basic_demo.py
```

**For ReAct agent demonstration:**
```bash
python algotune_react_agent.py
```

## 📁 Files Overview

### Demo Scripts

#### Basic Optimization Demo
- **`algotune_basic_demo.py`** (3.9KB, 123 lines) - **Simple optimization demonstrations**
  - Shows how to optimize specific AlgoTune tasks with hardcoded solutions
  - Demonstrates proper task instance creation and environment usage
  - Covers 3 algorithms: QR factorization, matrix multiplication, convex hull
  - **Use for:** Understanding AlgoTune basics, testing environment setup

#### ReAct Agent Demo
- **`algotune_react_agent.py`** (9.9KB, 285 lines) - **LLM-based optimization agent**
  - Implements ReAct (Reasoning + Acting) pattern for algorithm optimization
  - Shows iterative improvement based on performance feedback
  - Includes mock LLM for demonstration purposes
  - **Use for:** Understanding agent-based optimization, ReAct pattern implementation

## 🚀 Usage Examples

### 1. Basic Optimization Demo

```bash
# Run the basic demo to see hardcoded optimizations
python algotune_basic_demo.py
```

**Expected Output:**
```
AlgoTune Environment Demo
Demonstrating algorithm optimization with proper task instances

============================================================
Optimizing qr_factorization (n=128)
============================================================

Initial state:
  Task: qr_factorization
  Problem size: 128
  Baseline time: 0.0234s
  Target speedup: 1.5x

Submitting optimization attempt...
✅ Solution correct!
  Time: 0.0156s
  Speedup: 1.50x
  🎯 Target speedup achieved!
```

### 2. ReAct Agent Demo

```bash
# Run the ReAct agent demo (uses mock LLM)
python algotune_react_agent.py
```

**Expected Output:**
```
AlgoTune React Agent Demo
Demonstrating LLM-based algorithm optimization

============================================================
React Agent optimizing matrix_multiplication (n=32)
============================================================

You are optimizing the matrix_multiplication algorithm.

Current state:
- Problem size: n=32
- Baseline time: 0.0012s
- Target speedup: 1.5x
- Attempts so far: 0

🤔 Thinking about attempt 1...
📝 Generated solution:
----------------------------------------
  import numpy as np
  def solve(problem):
      A = problem["A"]
      B = problem["B"]
      return np.dot(A, B).tolist()
  ...
----------------------------------------

Attempt 1 results:
✅ Solution correct!
- Time: 0.0008s
- Speedup: 1.50x
- Best speedup so far: 1.50x

🎉 Target speedup achieved in 1 attempts!
```

## 🧪 Algorithm Optimization Tasks

AlgoTune includes several computational challenges:

### Matrix Multiplication
- **Task**: Optimize matrix multiplication for various sizes
- **Challenge**: Balance numerical precision with computational speed
- **Techniques**: BLAS optimization, memory layout, precision tradeoffs

### QR Factorization
- **Task**: Optimize QR decomposition of matrices
- **Challenge**: Choose between full/reduced QR, precision vs speed
- **Techniques**: Specialized algorithms, numerical stability considerations

### Convex Hull
- **Task**: Find convex hull of 2D point sets
- **Challenge**: Handle various point distributions and sizes
- **Techniques**: Graham scan, QuickHull, preprocessing optimizations

## 🔧 Environment Interface

### Task Instance Creation
```python
from synth_env.examples.algotune.taskset import create_algotune_task_instance

task_instance = create_algotune_task_instance(
    task_name="matrix_multiplication",
    problem_size=128,
    random_seed=42,
    target_speedup=1.5
)
```

### Environment Usage
```python
from synth_env.examples.algotune.environment import AlgoTuneEnvironment

env = AlgoTuneEnvironment(task_instance)
obs = await env.initialize()

# Submit optimization
obs = await env.step({"code": optimization_code})

# Check results
if obs['public']['ok']:
    print(f"Speedup: {obs['public']['speedup_vs_baseline']:.2f}x")
```

### Code Format
Optimization code must define a `solve(problem)` function:

```python
import numpy as np

def solve(problem):
    # Extract problem parameters
    A = problem["A"]
    B = problem["B"]
    
    # Implement optimized algorithm
    result = np.dot(A, B)
    
    # Return in expected format
    return result.tolist()
```

## 📊 Performance Metrics

### Evaluation Criteria
- **Correctness**: Solution must pass all test cases
- **Speed**: Measured execution time on test problems
- **Speedup**: Ratio of baseline time to optimized time
- **Target Achievement**: Whether speedup meets the target threshold

### Observation Format
```python
{
    'public': {
        'task': 'matrix_multiplication',
        'n': 128,
        'baseline_time': 0.0234,
        'target_speedup': 1.5,
        'attempts': 1,
        'ok': True,                    # Correctness
        'elapsed': 0.0156,            # Execution time
        'speedup_vs_baseline': 1.50,  # Performance improvement
        'best_speedup': 1.50          # Best achieved so far
    }
}
```

## 🤖 ReAct Agent Architecture

### Agent Components
1. **Observation Formatter**: Converts environment state to LLM-readable format
2. **Prompt Generator**: Creates task-specific prompts with optimization hints
3. **Code Extractor**: Parses LLM response to extract executable code
4. **Feedback Loop**: Uses performance results to guide next attempts

### Optimization Strategy
1. **Initial Analysis**: Understand the task and constraints
2. **Baseline Assessment**: Analyze current performance metrics
3. **Iterative Improvement**: Generate optimizations based on feedback
4. **Performance Tracking**: Monitor speedup improvements across attempts

### Mock LLM Behavior
The demo includes a mock LLM that:
- Returns predefined optimization attempts for each task
- Simulates realistic optimization progression
- Demonstrates agent learning from feedback

## 🔍 Development Tips

### Adding New Algorithms
1. **Extend task instances**: Add new algorithm types to taskset
2. **Update optimization attempts**: Include example solutions in demos
3. **Add task-specific hints**: Provide optimization guidance in prompts
4. **Test correctness**: Ensure solutions pass validation

### Optimization Techniques
- **Algorithmic improvements**: Better algorithms (O(n²) → O(n log n))
- **Numerical optimizations**: Precision tradeoffs, specialized libraries
- **Memory optimizations**: Cache-friendly access patterns
- **Parallelization**: Multi-threading, vectorization

### Common Pitfalls
- **Numerical precision**: Optimizations may affect accuracy
- **Edge cases**: Handle degenerate inputs (empty arrays, single points)
- **Import statements**: Include all necessary dependencies
- **Return format**: Match expected output structure exactly

## 🧪 Extending the Demos

### Adding Real LLM Support
Replace the mock LLM with actual model calls:

```python
from synth_ai.zyk import LM

class AlgoTuneReactAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.lm = LM(model_name)  # Real LLM instead of mock
        
    async def generate_solution(self, prompt: str) -> str:
        response = await self.lm.respond_async(prompt)
        return self.extract_code(response)
```

### Creating Evaluation Scripts
Build comprehensive evaluation systems:

```python
async def evaluate_agent(agent, tasks, num_episodes=10):
    results = []
    for task_name in tasks:
        for episode in range(num_episodes):
            result = await agent.optimize_task(task_name)
            results.append({
                'task': task_name,
                'episode': episode,
                'best_speedup': result
            })
    return results
```

### Adding Trace Generation
Integrate with the trace viewer system:

```python
from synth_sdk.tracing.decorators import trace_calls

@trace_calls
async def optimize_with_tracing(self, task_name: str):
    # Optimization logic with automatic tracing
    pass
```

## 📚 Related Documentation

- [AlgoTune Environment Guide](../README.md) - Environment setup and usage
- [Task Instance Creation](../taskset.py) - Creating custom optimization tasks
- [Agent Development Guide](../../../docs/agent_development.md) - Building optimization agents
- [Performance Benchmarking](../../../docs/benchmarking.md) - Evaluation best practices

## 🤝 Contributing

When extending the AlgoTune demos:

1. **Follow naming conventions**: Use descriptive names like `algotune_*_demo.py`
2. **Update this README**: Document new files and their purposes
3. **Include error handling**: Graceful handling of optimization failures
4. **Add comprehensive examples**: Cover various algorithm types and optimization strategies
5. **Test thoroughly**: Verify with different problem sizes and algorithms

## 📈 Future Enhancements

Planned improvements for the demo system:

- **Multi-algorithm evaluation**: Compare optimization across different tasks
- **Performance profiling**: Detailed analysis of optimization bottlenecks
- **Curriculum learning**: Progressive difficulty in optimization challenges
- **Collaborative optimization**: Multiple agents working together
- **Real-world benchmarks**: Industry-standard algorithm optimization tasks

## 🎯 Learning Objectives

These demos help you understand:

- **Algorithm optimization principles**: Speed vs accuracy tradeoffs
- **Environment interaction patterns**: Task creation, code submission, feedback loops
- **Agent design patterns**: ReAct methodology for iterative improvement
- **Performance evaluation**: Metrics and benchmarking for optimization tasks
- **Code generation challenges**: Producing correct, efficient implementations

Run the demos to see these concepts in action! 🚀 