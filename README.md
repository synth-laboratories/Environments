# Environments
Synthetic Environments / Long Horizon Tasks / Digital Control Agents

![Coverage](https://img.shields.io/badge/coverage-42.3%25-yellow)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

# Motivation
- We're building Environments to have key desiderata for long-horizon language agents
- Snapshotting and reproducibility
- Statefulness as a first-class citizen
- Consistent abstractions for engine interaction and state reads across settings
- Environment observability and tracing
- HTTP access for simplified training and evaluation
- Strong data abstractions to enable easily-configurable filtering and curriculum learning
- Simple abstractions for agent-originated tool edits, etc.

# User Guide
Note - this repo is under extremely active development. Hic sunt dracones, if not contributing it may be more useful as a reference or development resource than as core code for your production systems.

# Supported Environments
[] Sokoban (maturing, not active dev)
    - This environment is not of our own making! Please credit the great researchers at [Deepmind](https://deepmind.google/discover/blog/agents-that-imagine-and-plan/) and [Ragen](https://ragen-ai.github.io), among others.
    - Checkout test_synth_react.py for a hello world example!
[] Hendryks Math [] (maturing)
[] Crafter-Classic (maturing)
[] EnronBench (active dev)
[] SWE-Bench (active dev - do not attempt)
[] NMMO (active dev - do not attempt)
[] Red (active dev)
[] Verilog (maturing)

...

# Development

To use the Astral suite of devtools:
- Package management: `uv sync`, `uv add`, `uv remove`
- Linting: `ruff format .`
- Type checking: `uvx ty check`

## Testing and Coverage

Run tests with coverage:
```bash
# Run all tests with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test files
pytest tests/unit/test_registry.py

# Run with verbose output
pytest tests/ -v
```

Current test coverage: **42.3%**

The test suite includes:
- **Unit tests** for core components (registry, environments, engines)
- **Integration tests** for service API endpoints
- **Q* solver tests** for Sokoban environment
- No AI agent demos in tests (algorithmic solving only)

