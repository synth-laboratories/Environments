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

### Running Evaluation Scripts

The framework includes ReAct agent evaluation scripts for testing language models on various environments. These scripts provide comprehensive metrics and shaped rewards for training.

#### Prerequisites
1. Start the synth service on port 8901:
   ```bash
   # In your service directory
   python -m uvicorn main:app --host 0.0.0.0 --port 8901
   ```

2. Ensure your model is available (OpenAI, Anthropic, etc.)

#### TicTacToe Evaluation
```bash
cd Environments
uvpm synth_env.examples.tictactoe.agent_demos.test_tictactoe_react_agent
```

**Features:**
- Tests strategic gameplay against random opponent  
- Provides win/loss/draw statistics
- Validates coordinate parsing and legal moves
- Supports multiple models (gpt-4.1-mini, o3, etc.)

#### NetHack Evaluation  
```bash
cd Environments
uvpm synth_env.examples.nethack.agent_demos.test_nethack_react_agent
```

**Features:**
- Comprehensive dungeon exploration evaluation
- 26+ shaped reward signals for training
- Balrog scoring system integration
- Progress bars for multi-trajectory runs
- Separates relevant vs. irrelevant metrics

#### Sokoban Evaluation
```bash
cd Environments  
uvpm synth_env.examples.sokoban.agent_demos.test_sokoban_react_agent
```

**Features:**
- Classic puzzle-solving evaluation
- Box-pushing logic validation
- Step efficiency analysis
- Multiple difficulty levels

#### Configuration
Edit the script configuration at the top of each file:
```python
MODEL_NAME = "gpt-4.1-mini"  # or "o3", "claude-sonnet-4", etc.
NUM_INSTANCES = 5            # Number of test episodes
MAX_TURNS = 100             # Maximum steps per episode  
DIFFICULTY = "beginner"     # Environment-specific difficulty
```

All scripts provide detailed rubric results, progress metrics, and shaped rewards suitable for reinforcement learning applications.

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

