# Environments
Synthetic Environments / Long Horizon Tasks / Digital Control Agents

![Coverage](https://img.shields.io/badge/coverage-0.0%25-red)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-7/19 failing-red)
![Type Check Examples](https://img.shields.io/badge/types (examples)-98%25 (5 errors)-green)-98%25 (5 errors)-green)-98%25 (5 errors)-green)
![Type Check Core](https://img.shields.io/badge/types (core)-100%25 (0 errors)-green)-100%25 (0 errors)-green)-100%25 (0 errors)-green)
![PyPI Dev](https://img.shields.io/badge/PyPI dev-0.0.1.dev2-orange)

uvicorn src.synth_env.service.app:app --reload --port 8000
uv run python src/synth_env/examples/nethack/agent_demos/test_synth_react.py

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

For development setup, testing, and contribution guidelines, see:
- **[Contributing Guide](dev/contributing.md)** - Complete development setup and workflow
- **[Environment Contribution Guide](docs/env_contribution_guide.md)** - Step-by-step guide for adding new environments

## Quick Development Commands

```bash
# Install dependencies
uv sync

# Run tests
./scripts/run_tests.sh

# Update README metrics
./scripts/run_dev_helpers.sh

# Format code
ruff format .

# Check codebase health
python scripts/check_health.py

# Release to PyPI (increment dev version)
python scripts/release.py

# Release with version selection
python scripts/release.py --minor
python scripts/release.py --patch

# Dry run to see what would happen
python scripts/release.py --dry-run

# Publish to TestPyPI instead
python scripts/release.py --test-pypi
```

