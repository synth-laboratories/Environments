# Contributing to Synth-Env

Welcome to the Synth-Env project! This guide will help you get started with contributing to our synthetic environments framework for long-horizon language agents.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Setup](#development-setup)
3. [Contributing New Environments](#contributing-new-environments)
4. [Testing](#testing)
5. [Code Quality](#code-quality)
6. [Development Workflow](#development-workflow)
7. [Package Publishing](#package-publishing)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

## Related Documentation

- **[Testing Guide](testing.md)** - Comprehensive testing documentation
- **[Environment Contribution Guide](../docs/env_contribution_guide.md)** - Step-by-step environment implementation

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- Git for version control

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/Environments.git
   cd Environments
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the service:**
   ```bash
   uvicorn src.synth_env.service.app:app --reload --port 8000
   ```

4. **Test an environment:**
   ```bash
   uv run python src/synth_env/examples/nethack/agent_demos/test_synth_react.py
   ```

## Development Setup

### Package Management

We use the Astral suite of devtools:
- **Package management**: `uv sync`, `uv add`, `uv remove`
- **Linting**: `ruff format .`
- **Type checking**: `uvx ty check`

### Development Dependencies

Install development dependencies:
```bash
uv add --dev pytest pytest-cov pytest-asyncio requests
```

### Environment Variables

Create a `.env` file in the project root with any necessary environment variables.

## Contributing New Environments

We have a comprehensive guide for implementing new environments. See [Environment Contribution Guide](../docs/env_contribution_guide.md) for the complete step-by-step process.

### Quick Overview

Each environment consists of:
1. **Engine** - Core game/task logic and state management
2. **Environment** - Wrapper that provides the standardized interface
3. **TaskSet** - Task instance generation and configuration
4. **Agent Demos** - Example agents (typically ReAct) that solve tasks
5. **Unit Tests** - Comprehensive testing of all components

### Directory Structure

```
src/synth_env/examples/your_env/
├── __init__.py                    # Module init
├── engine.py                      # Core logic + state management
├── environment.py                 # StatefulEnvironment wrapper
├── taskset.py                     # Task/TaskInstance generator
├── agent_demos/
│   └── test_synth_react.py       # ReAct agent evaluation
└── units/                         # Unit tests
    ├── test_your_env_engine.py
    ├── test_your_env_environment.py
    └── test_your_env_taskset.py
```

### Implementation Checklist

- [ ] Engine implements all required methods
- [ ] Environment wraps engine correctly
- [ ] TaskSet generates valid instances
- [ ] Agent demo runs successfully
- [ ] Unit tests pass with good coverage
- [ ] Type checker reports no errors
- [ ] Registered in service app.py
- [ ] Documentation/comments added

## Testing

For comprehensive testing documentation, see [Testing Guide](testing.md).

### Quick Testing Commands

**Run all tests:**
```bash
./scripts/run_tests.sh
```

**Test specific environment:**
```bash
./scripts/run_tests.sh --env sokoban
```

**Unit tests only:**
```bash
./scripts/run_tests.sh --unit
```

**Integration tests only:**
```bash
./scripts/run_tests.sh --integration
```

Current test coverage: **12.5%** (8/29 tests passing)

## Code Quality

### Type Checking

We use `uvx ty check` for type checking:

```bash
# Check examples directory
uvx ty check src/synth_env/examples

# Check core synth_env (excluding examples)
uvx ty check src/synth_env --exclude src/synth_env/examples
```

Current type checking status:
- **Examples**: 98% clean (5 errors in 201 files)
- **Core**: 100% clean (0 errors in 1 files)

### Linting and Formatting

Format code with ruff:
```bash
ruff format .
```

### Code Style Guidelines

- Use `from __future__ import annotations` at the top of files
- Use absolute imports from synth_env
- All functions should have type hints
- Use `InternalObservation` for observation returns
- Intent.rubric is `Dict[str, Any]` (usually `{"goal": "..."}`)
- Use `Optional` for nullable fields

### Important Conventions

- All interface methods must be async
- Use `await` when calling other async methods
- Engine methods like `_step_engine` must be async
- Public state = what agent sees
- Private state = internal bookkeeping
- Always implement `diff()` methods for states
- States should be immutable (use copies)

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding guidelines and implement your changes.

### 3. Run Quality Checks

```bash
# Update README metrics
./scripts/run_dev_helpers.sh

# Run tests
./scripts/run_tests.sh

# Check types
uvx ty check src/synth_env/examples
uvx ty check src/synth_env --exclude src/synth_env/examples

# Format code
ruff format .

# Publish package (dev version)
python scripts/publish.py

# Publish with version selection
python scripts/publish.py --minor
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new environment XYZ"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Package Publishing

We use `uv` for building and publishing packages to PyPI. The publishing script handles version management, building, testing, and uploading.

### Publishing Workflow

**1. Increment dev version (default):**
```bash
python scripts/publish.py
```

**2. Interactive version selection:**
```bash
# Minor version increment with TUI
python scripts/publish.py --minor

# Major version increment with TUI  
python scripts/publish.py --major
```

**3. Set specific version:**
```bash
python scripts/publish.py --version 1.0.0
```

**4. Test before publishing:**
```bash
# Dry run to see what would happen
python scripts/publish.py --dry-run

# Publish to Test PyPI first
python scripts/publish.py --test-pypi
```

### Publishing Requirements

**Environment Variables:**
- `UV_PUBLISH_TOKEN` - PyPI API token for production
- `UV_PUBLISH_TOKEN_TEST` - Test PyPI API token (optional)

**Get PyPI API Token:**
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create a new API token
3. Set the environment variable:
   ```bash
   export UV_PUBLISH_TOKEN=pypi-*****
   ```

### Build System

The project uses `uv_build` as the build backend for fast, reliable builds:

```toml
[build-system]
requires = ["uv_build>=0.7.19,<0.8.0"]
build-backend = "uv_build"
```

### Version Management

Version format: `MAJOR.MINOR.PATCH[.dev|rc|alpha|beta]NUMBER`

Examples:
- `0.1.0` - Stable release
- `0.1.0.dev1` - Development version
- `0.1.0.rc1` - Release candidate
- `1.0.0` - Major release

The script automatically:
- Increments dev versions by default
- Provides interactive selection for minor/major
- Validates version format
- Updates `pyproject.toml`
- Builds with `uv build --no-sources`
- Tests the wheel locally
- Publishes to PyPI with `uv publish`

## Project Structure

```
Environments/
├── src/
│   ├── synth_env/
│   │   ├── examples/              # Environment implementations
│   │   │   ├── sokoban/
│   │   │   ├── tictactoe/
│   │   │   ├── nethack/
│   │   │   └── ...
│   │   ├── environment/           # Core environment framework
│   │   ├── stateful/             # Stateful engine abstractions
│   │   ├── tasks/                # Task and TaskSet abstractions
│   │   └── service/              # HTTP service
│   ├── viewer/                   # Trace visualization
│   └── evals/                    # Evaluation results
├── tests/                        # Test suite
├── docs/                         # Documentation
│   └── env_contribution_guide.md # Environment implementation guide
├── dev/                          # Development tools
│   ├── contributing.md           # This file
│   ├── testing.md               # Testing documentation
│   └── update_readme_metrics.py  # README metrics updater
├── scripts/                      # Build and deployment scripts
│   ├── run_tests.sh             # Test runner
│   ├── run_dev_helpers.sh       # Development helpers
│   ├── publish.py               # Package publishing script
│   └── *.sh                     # Other utility scripts
└── README.md                     # Project overview
```

## Troubleshooting

### Service Not Running

If you see "Service is not running", start it with:
```bash
cd src && python -m synth_env.service.app
```

### Missing Dependencies

Install test dependencies:
```bash
uv add --dev pytest pytest-asyncio pytest-cov requests
```

### Type Checker Not Found

Install `uv` for type checking:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Permission Denied

Make scripts executable:
```bash
chmod +x run_tests.sh run_dev_helpers.sh
```

### Import Errors

Make sure you're using the correct Python path:
```bash
PYTHONPATH=src python -m pytest tests/
```

### Coverage Issues

If coverage reports are missing, install coverage:
```bash
uv add --dev coverage pytest-cov
```

## Development Philosophy

Note: This repo is under extremely active development. *Hic sunt dracones* - if not contributing, it may be more useful as a reference or development resource than as core code for your production systems.

### Key Principles

We're building Environments to have key desiderata for long-horizon language agents:

- **Snapshotting and reproducibility** - Full state serialization/deserialization
- **Statefulness as a first-class citizen** - Rich state management
- **Consistent abstractions** - Uniform interface across environments
- **Environment observability and tracing** - Comprehensive logging
- **HTTP access** - Simplified training and evaluation
- **Strong data abstractions** - Configurable filtering and curriculum learning
- **Simple tool abstractions** - Agent-originated tool edits

### Supported Environments

- **Sokoban** (maturing, not active dev) - Credit [Deepmind](https://deepmind.google/discover/blog/agents-that-imagine-and-plan/) and [Ragen](https://ragen-ai.github.io)
- **Hendryks Math** (maturing)
- **Crafter-Classic** (maturing)
- **NetHack** (active dev)
- **TicTacToe** (reference implementation)
- **Verilog** (maturing)
- **EnronBench** (active dev)
- **SWE-Bench** (active dev - do not attempt)
- **NMMO** (active dev - do not attempt)
- **Red** (active dev)

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the [Environment Contribution Guide](../docs/env_contribution_guide.md)
- **Examples**: Look at existing environments like `tictactoe` or `sokoban`

## Contributing Guidelines

1. **Follow the coding style** - Use ruff formatting and type hints
2. **Write tests** - Aim for good coverage of your changes
3. **Update documentation** - Keep docs in sync with code changes
4. **Run quality checks** - Use `./run_dev_helpers.sh` before submitting
5. **Be descriptive** - Write clear commit messages and PR descriptions

Thank you for contributing to Synth-Env! 🚀
