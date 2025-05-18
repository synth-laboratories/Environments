# Environments
Synthetic Environments / Long Horizon Tasks / Digital Control Agents

# Motivation
- We're building Environments to have key desiderata for long-horizon language agents
- Snapshotting and reproducibility
- Statefulness as a first-class citizen
- Consistent abstractions for engine interaction and state reads across settings
- Environment observability and tracing
- HTTP access for simplified training and evaluation
- Strong data abstractions to enable easily-configurable filtering and curriculum learning

# User Guide
Note - this repo is under extremely active development. Hic sunt dracones, if not contributing it may be more useful as a reference or development resource than as core code for your production systems.

# Supported Environments
[] Sokoban (in dev)
...

# Development

To use the Astral suite of devtools:
package management: uv sync, uv add, uv remove
linting: ruff format .
type(hint) checking: uvx ty check

