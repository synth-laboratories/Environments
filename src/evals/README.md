# Evaluation System

This directory contains the evaluation infrastructure for testing agents across different environments.

## Directory Structure

```
src/evals/
├── configs/                    # Configuration files for evaluations
│   └── *.example.toml         # Example configuration templates
└── {env_name}/                # Results directory (created at runtime)
    └── run_{timestamp}/       # Individual evaluation runs
        ├── evaluation_summary.json
        ├── traces/           # Individual trace files
        └── viewer/          # Standalone trace viewer
```

## Configuration Files

Configuration files use TOML format and define:
- Environment to evaluate
- Model to use
- Number of episodes
- Evaluation parameters

Example usage:
```bash
# Copy example config
cp src/evals/configs/crafter.example.toml src/evals/configs/my_eval.toml

# Edit configuration as needed
# Run evaluation (implementation depends on your evaluation script)
```

## Results

All evaluation results are automatically excluded from git via `.gitignore`. This includes:
- Trace files (can be large)
- Generated viewer files
- Evaluation summaries
- Runtime artifacts

Only example configuration files are committed to version control.

## Trace Viewer

Each evaluation run generates a standalone trace viewer in the `viewer/` subdirectory, allowing you to:
- Browse individual traces
- View agent reasoning steps
- Analyze environment interactions
- Compare performance across episodes

The main trace viewer at `src/viewer/` can also discover and display traces from this directory structure. 