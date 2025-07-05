# Testing Guide

This guide explains how to run tests for the Synth-Env framework.

## Quick Start

Run all tests:
```bash
./run_tests.sh
```

Or using Python (cross-platform):
```bash
python run_tests.py
```

## Test Options

### Test Types

- **Unit tests**: Test individual components in isolation
  ```bash
  ./run_tests.sh --unit
  ```

- **Integration tests**: Test agent demos and full environment runs
  ```bash
  ./run_tests.sh --integration
  ```

### Specific Environment

Test a single environment:
```bash
./run_tests.sh --env tictactoe
./run_tests.sh --env sokoban
./run_tests.sh --env verilog
./run_tests.sh --env crafter_classic
```

### Service Check

The test runner checks if the Synth-Env service is running on `http://localhost:6532`. To skip this check:
```bash
./run_tests.sh --skip-service
```

### Verbose Output

For detailed test output:
```bash
./run_tests.sh --verbose
```

### Combined Options

You can combine options:
```bash
# Run only unit tests for sokoban with verbose output
./run_tests.sh --unit --env sokoban --verbose

# Run integration tests without service check
./run_tests.sh --integration --skip-service
```

## What Gets Tested

### Unit Tests
- Engine functionality (state management, actions, rewards)
- Environment wrapper behavior
- TaskSet generation and serialization
- Core framework components

### Integration Tests
- Agent demos (ReAct agents)
- Full episode runs
- Environment-agent interactions
- Tool usage patterns

### Type Checking
- Runs `uvx ty check` on each environment
- Type errors are reported but non-blocking
- Ensures type annotations are correct

## Test Structure

Each environment has tests in:
```
src/synth_env/examples/<env_name>/
├── units/              # Unit tests
│   ├── test_<env>_engine.py
│   ├── test_<env>_environment.py
│   └── test_<env>_taskset.py
└── agent_demos/        # Integration tests
    └── test_synth_react.py
```

## Running Tests Manually

You can also run tests directly with pytest:

```bash
# Run all tests for an environment
PYTHONPATH=src python -m pytest src/synth_env/examples/tictactoe/

# Run specific test file
PYTHONPATH=src python -m pytest src/synth_env/examples/tictactoe/units/test_tictactoe_engine.py

# Run with coverage
PYTHONPATH=src python -m pytest --cov=synth_env src/synth_env/examples/tictactoe/
```

## Continuous Integration

The test scripts are designed to work in CI environments:
- Exit code 0 on success, 1 on failure
- Colored output for readability
- Service check can be skipped for CI
- Verbose mode for debugging failures

## Troubleshooting

### Service Not Running
If you see "Service is not running", start it with:
```bash
cd src && python -m synth_env.service.app
```

### Missing Dependencies
Install test dependencies:
```bash
pip install pytest pytest-asyncio
```

### Type Checker Not Found
Install `uv` for type checking:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Permission Denied
Make scripts executable:
```bash
chmod +x run_tests.sh run_tests.py
```