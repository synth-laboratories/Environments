# Synth Environments

**Synthetic Environments for Long-Horizon Language Agents**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.0.2.dev1-orange)](https://pypi.org/project/synth-env/)

A comprehensive framework for building and managing synthetic environments designed specifically for training and evaluating long-horizon language agents.

## 🎯 Key Features

- **🔄 Snapshotting & Reproducibility** - Full state capture and replay
- **🏗️ Statefulness First** - Built-in state management across environments  
- **🔌 Consistent APIs** - Unified interface for all environment types
- **📊 Observability** - Built-in tracing and monitoring
- **🌐 HTTP Access** - RESTful API for remote training and evaluation
- **📚 Curriculum Learning** - Configurable filtering and progression
- **🛠️ Agent Tools** - Simple abstractions for agent-environment interaction

## 🚀 Quick Start

### Installation

```bash
pip install synth-env
```

### Basic Usage

```python
from synth_env import Environment

# Create environment
env = Environment("sokoban")

# Run agent
state = env.reset()
while not env.done:
    action = agent.act(state)
    state = env.step(action)
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/synth-env.git
cd synth-env

# Install dependencies
uv sync

# Run tests
python dev/update_readme_metrics.py --fast
```

## 🎮 Supported Environments

| Environment | Status | Description |
|-------------|---------|-------------|
| **Sokoban** | ✅ Stable | Classic puzzle game for planning |
| **Hendryks Math** | ✅ Stable | Mathematical reasoning tasks |
| **Crafter** | ✅ Stable | Minecraft-like survival environment |
| **Verilog** | 🔄 Beta | Hardware description language tasks |
| **Red Team** | 🚧 Development | Security testing scenarios |
| **SWE-Bench** | 🚧 Development | Software engineering tasks |

## 📖 Documentation

- **[API Reference](docs/api.md)** - Complete API documentation
- **[Environment Guide](docs/environments.md)** - Detailed environment descriptions
- **[Contributing](dev/contributing.md)** - Development setup and guidelines

## 🔧 Development

### Health Check
```bash
# Check codebase health
python scripts/check_health.py
```

### Testing
```bash
# Fast tests (~3 seconds)
python dev/update_readme_metrics.py --fast

# Full test suite
python dev/update_readme_metrics.py
```

### Code Quality
```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
uvx ty check
```

### Release
```bash
# Increment version and publish
python scripts/release.py

# Dry run
python scripts/release.py --dry-run
```

### Pre-Merge Checklist
Before creating a PR, see **[dev/pr_checklist.md](dev/pr_checklist.md)** for the complete checklist.

## 🤝 Contributing

We welcome contributions! Please see our **[Contributing Guide](dev/contributing.md)** for:
- Development setup
- Code style guidelines  
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the research teams at DeepMind, Ragen AI, and other contributors to the environments included in this framework.

---

**⚠️ Development Status**: This project is under active development. While stable environments are production-ready, newer environments may have breaking changes.

